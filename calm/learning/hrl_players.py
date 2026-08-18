
import pygame
import sys
import time

import torch.nn.functional

from . import gamepad

import copy
from gym import spaces
import numpy as np
import os
import torch 
import yaml

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext

import learning.common_player as common_player
import learning.calm_players as calm_players
import learning.calm_models as calm_models
import learning.calm_network_builder as calm_network_builder

from .distill_network import SingleDistilModel, SingleMoEDistill


#############################
# for check param, model size
#############################

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb), param_size)

############################

class HRLPlayer(common_player.CommonPlayer):
    def __init__(self, config):
        self._joystick = gamepad.Joystick(joystick_idx=0)
        self._joystick_coroutine = self._joystick.command() if self._joystick.is_available() else None

        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
            self._latent_dim = llc_config_params['config']['latent_dim']
        
        super().__init__(config)
        
        self._task_size = self.env.task.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        
        assert(llc_checkpoint != "")
        self._build_llc(llc_config_params, llc_checkpoint)

        self._target_motion_index = torch.zeros((self.env.task.num_envs, 1), dtype=torch.long, device=self.device)

        self._MODE = config['mode']
        self._exp_path = config['exp_path']
        self._student_ckpt = config['student_ckpt']
        self._collect_n_games = config['collect_n_games']
        self._collect_max_steps = config['collect_max_steps']
        self._input_size = config['input_size']
        self._db_name = config['db_name']
        self._db_size = config['db_size']
        self.NEAREST_MOTION_TEST = config['NEAREST_MOTION_TEST']
        self._debug_hlp_obs = config.get('debug_hlp_obs', False)
        self._debug_hlp_obs_interval = max(1, int(config.get('debug_hlp_obs_interval', 30)))
        self._debug_hlp_obs_step = 0
        self._for_normal = config.get('for_normal', '')

        ############
        if self._MODE == 'COLLECT_TEACHER':
            # for collect teacher data
            self._obs_record = list()
            self._latent_mu_record = list()
            self._latent_sigma_record = list()
            self._value_record = list()
        ############
            
        if self._MODE == 'COLLECT_MATCHING_DB':
            self._amp_obs = list()
            self._latent = list()
        
        if self._MODE == 'COLLECT_INIT_CONDITION':
            self._init_ball_states = list()
            self._init_dof_pos = list()
            self._init_dof_vel = list()
            self._init_root_states = list()
            self._init_rb_pos = list()
            self._init_rb_rot = list()
            self._init_rb_vel = list()
            self._init_rb_ang_vel = list()

        if self.NEAREST_MOTION_TEST:
            path = '/workspace/CALM/latents_90_new.npy'
            self._motions = torch.tensor(np.load(path), device=self.device)
            self._current_latent = torch.zeros_like(self._motions)

            with open('/workspace/CALM/blackhole/motions/soccer_90.yaml') as f:
                data = yaml.load(f)
            self._motion_names = list(data['motions'])
        return

    def _maybe_debug_hlp_obs(self, obs):
        if not self._debug_hlp_obs or self._task_size <= 0:
            return

        if self._debug_hlp_obs_step % self._debug_hlp_obs_interval != 0:
            self._debug_hlp_obs_step += 1
            return

        task_obs = obs[..., -self._task_size:]
        task_obs_0 = task_obs[0].detach().float().cpu().numpy()
        task_name = self.env.task.__class__.__name__
        progress = None
        if hasattr(self.env.task, 'progress_buf'):
            progress = int(self.env.task.progress_buf[0].item())

        print(f"[HLP obs debug] step={self._debug_hlp_obs_step} task={task_name} env=0 progress={progress} obs_dim={obs.shape[-1]} task_dim={self._task_size}")

        if task_name == 'HumanoidDribble' and self._task_size == 14:
            print(
                "  local_tar_dir={} tar_speed={:.3f} local_tar_face_dir={}".format(
                    np.array2string(task_obs_0[0:2], precision=3, suppress_small=True),
                    float(task_obs_0[2]),
                    np.array2string(task_obs_0[3:5], precision=3, suppress_small=True),
                )
            )
            print(
                "  local_ball_pos={} local_ball_vel={} local_ball_ang_vel={}".format(
                    np.array2string(task_obs_0[5:8], precision=3, suppress_small=True),
                    np.array2string(task_obs_0[8:11], precision=3, suppress_small=True),
                    np.array2string(task_obs_0[11:14], precision=3, suppress_small=True),
                )
            )
        elif task_name == 'HumanoidToeStop' and self._task_size == 11:
            print(
                "  local_ball_pos={} local_ball_vel={} local_ball_ang_vel={} target_foot_one_hot={}".format(
                    np.array2string(task_obs_0[0:3], precision=3, suppress_small=True),
                    np.array2string(task_obs_0[3:6], precision=3, suppress_small=True),
                    np.array2string(task_obs_0[6:9], precision=3, suppress_small=True),
                    np.array2string(task_obs_0[9:11], precision=3, suppress_small=True),
                )
            )
        else:
            print("  task_obs={}".format(np.array2string(task_obs_0, precision=3, suppress_small=True)))

        self._debug_hlp_obs_step += 1
        return
    
    def restore(self, fn):
        super().restore(fn)
        self._override_self_obs_normalization()
        return

    def _override_self_obs_normalization(self):
        if not self._for_normal:
            return

        if not self.normalize_input:
            raise RuntimeError("--for_normal requires normalize_input=True for the HLP player.")

        source_checkpoint = torch_ext.load_checkpoint(self._for_normal)
        source_rms = source_checkpoint.get('running_mean_std')
        if source_rms is None:
            raise KeyError(f"--for_normal checkpoint has no running_mean_std: {self._for_normal}")

        current_rms = self.running_mean_std.state_dict()
        self_obs_size = self.obs_shape[0] - self._task_size
        if self_obs_size <= 0:
            raise RuntimeError(f"Invalid self obs size: obs_shape={self.obs_shape}, task_size={self._task_size}")

        copied_keys = []
        for key in ['running_mean', 'running_var']:
            if key not in current_rms:
                raise KeyError(f"Current running_mean_std has no key '{key}'. Available keys: {list(current_rms.keys())}")
            if key not in source_rms:
                raise KeyError(f"Source running_mean_std has no key '{key}'. Available keys: {list(source_rms.keys())}")

            current_tensor = current_rms[key].clone()
            source_tensor = source_rms[key].to(device=current_tensor.device, dtype=current_tensor.dtype)

            current_flat = current_tensor.reshape(-1).clone()
            source_flat = source_tensor.reshape(-1)
            if source_flat.numel() < self_obs_size:
                raise RuntimeError(
                    f"--for_normal source obs dim is too small for self obs override: "
                    f"source_dim={source_flat.numel()}, self_obs_size={self_obs_size}, source={self._for_normal}"
                )
            if current_flat.numel() < self_obs_size:
                raise RuntimeError(
                    f"Current obs dim is too small for self obs override: "
                    f"current_dim={current_flat.numel()}, self_obs_size={self_obs_size}"
                )

            current_flat[:self_obs_size] = source_flat[:self_obs_size]
            current_rms[key] = current_flat.reshape_as(current_tensor)
            copied_keys.append(key)

        self.running_mean_std.load_state_dict(current_rms)
        source_dim = source_rms[copied_keys[0]].reshape(-1).numel()
        current_dim = current_rms[copied_keys[0]].reshape(-1).numel()
        print(
            f"[HLP normal override] source={self._for_normal} "
            f"self_obs_size={self_obs_size} current_dim={current_dim} source_dim={source_dim} "
            f"copied={copied_keys}"
        )
        return


    def get_action(self, obs_dict, is_determenistic=False):
        obs = obs_dict['obs']

        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        self._maybe_debug_hlp_obs(obs)
        proc_obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : proc_obs,
            'rnn_states' : self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = torch.squeeze(current_action.detach())
        # print('z', current_action.max(), current_action.min(), current_action.norm(dim=1))
        clamped_actions = torch.clamp(current_action, -1.0, 1.0)
        
        # experiences
        if self._MODE == 'COLLECT_TEACHER':
            self._obs_record.append(proc_obs)
            self._latent_mu_record.append(clamped_actions)
            self._latent_sigma_record.append(res_dict['sigmas'])
            self._value_record.append(res_dict['values'])
        elif self._MODE  == 'COLLECT_INIT_CONDITION':
            ########################################################################
            ######### initial states for dribble skill tuning - from trapping
            ######### need contact condition change!!
            ########################################################################
            
            # for trapping collect
            # self._init_ball_states.extend(self.env.task._ball_states[self.env.task._touched==2].cpu().numpy()) if self.env.task._ball_states[self.env.task._touched==2].shape[0] > 0 else None
            # self._init_dof_pos.extend(self.env.task._dof_pos[self.env.task._touched==2].cpu().numpy()) if self.env.task._dof_pos[self.env.task._touched==2].shape[0] > 0 else None
            # self._init_dof_vel.extend(self.env.task._dof_vel[self.env.task._touched==2].cpu().numpy()) if self.env.task._dof_vel[self.env.task._touched==2].shape[0] > 0 else None
            # self._init_root_states.extend(self.env.task._humanoid_root_states[self.env.task._touched==2].cpu().numpy()) if self.env.task._humanoid_root_states[self.env.task._touched==2].shape[0] > 0 else None
            # self._init_rb_pos.extend(self.env.task._rigid_body_pos[self.env.task._touched==2, :15].cpu().numpy()) if self.env.task._rigid_body_pos[self.env.task._touched==2].shape[0] > 0 else None
            # self._init_rb_vel.extend(self.env.task._rigid_body_vel[self.env.task._touched==2, :15].cpu().numpy()) if self.env.task._rigid_body_vel[self.env.task._touched==2].shape[0] > 0 else None
            # self._init_rb_rot.extend(self.env.task._rigid_body_rot[self.env.task._touched==2, :15].cpu().numpy()) if self.env.task._rigid_body_rot[self.env.task._touched==2].shape[0] > 0 else None
            # self._init_rb_ang_vel.extend(self.env.task._rigid_body_ang_vel[self.env.task._touched==2, :15].cpu().numpy()) if self.env.task._rigid_body_ang_vel[self.env.task._touched==2].shape[0] > 0 else None

            # # for dribble collect
            # self._init_ball_states.extend(self.env.task._ball_states[self.env.task.progress_buf%10==0].cpu().numpy()) if self.env.task._ball_states[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            # self._init_dof_pos.extend(self.env.task._dof_pos[self.env.task.progress_buf%10==0].cpu().numpy()) if self.env.task._dof_pos[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            # self._init_dof_vel.extend(self.env.task._dof_vel[self.env.task.progress_buf%10==0].cpu().numpy()) if self.env.task._dof_vel[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            # self._init_root_states.extend(self.env.task._humanoid_root_states[self.env.task.progress_buf%10==0].cpu().numpy()) if self.env.task._humanoid_root_states[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            # self._init_rb_pos.extend(self.env.task._rigid_body_pos[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_pos[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            # self._init_rb_vel.extend(self.env.task._rigid_body_vel[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_vel[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            # self._init_rb_rot.extend(self.env.task._rigid_body_rot[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_rot[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            # self._init_rb_ang_vel.extend(self.env.task._rigid_body_ang_vel[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_ang_vel[self.env.task.progress_buf%10==0].shape[0] > 0 else None

            # for move collect
            self._init_dof_pos.extend(self.env.task._dof_pos[self.env.task.progress_buf%10==0].cpu().numpy()) if self.env.task._dof_pos[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            self._init_dof_vel.extend(self.env.task._dof_vel[self.env.task.progress_buf%10==0].cpu().numpy()) if self.env.task._dof_vel[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            self._init_root_states.extend(self.env.task._humanoid_root_states[self.env.task.progress_buf%10==0].cpu().numpy()) if self.env.task._humanoid_root_states[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            self._init_rb_pos.extend(self.env.task._rigid_body_pos[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_pos[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            self._init_rb_vel.extend(self.env.task._rigid_body_vel[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_vel[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            self._init_rb_rot.extend(self.env.task._rigid_body_rot[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_rot[self.env.task.progress_buf%10==0].shape[0] > 0 else None
            self._init_rb_ang_vel.extend(self.env.task._rigid_body_ang_vel[self.env.task.progress_buf%10==0, :15].cpu().numpy()) if self.env.task._rigid_body_ang_vel[self.env.task.progress_buf%10==0].shape[0] > 0 else None

        return clamped_actions


    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None


        if self._MODE == 'EVALUATE':
            n_games = 1
            self.max_steps = self.env.task.max_episode_length
            print(f"\n\nngames and max_steps set to {n_games}\n\n")

        #####################
        if self._MODE == 'COLLECT_TEACHER':
            n_games = self._collect_n_games
            self.max_steps = self._collect_max_steps
        #####################_init_conditions

        ###############
        if self._MODE == 'RUN_STUDENT':
            # for student model
            if self.model:
                del self.model
            self.model = SingleDistilModel(input_size=self._input_size, output_size=64, rnn=False)
            # self.model = MoEDistillModel(output_size=64, rnn=False)
            
            # 저장된 파일 로드
            loaded = torch.load(self._student_ckpt)

            # 로드된 모델 구조와 state_dict 사용
            self.model.load_state_dict(loaded['state_dict'])
            self.running_mean_std.load_state_dict(loaded['running_mean_std'])
        if self._MODE == 'COLLECT_INIT_CONDITION':
            n_games = self._collect_n_games
            self.max_steps = self._collect_max_steps
        
        #################

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        # n_games = 10
        for _ in range(n_games):
            if games_played >= n_games:
                print(f"game exceeded!: {games_played} >= {n_games}")
                break

            obs_dict = self.env_reset()
            batch_size = 1
            if len(obs_dict['obs'].size()) > len(self.obs_shape):
                batch_size = obs_dict['obs'].size()[0]
            self.batch_size = batch_size

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            done_indices = []

            for n in range(self.max_steps):
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                
                # force latent output
                # action = self._motions[70].repeat([2, 1]) + .0 * torch.randn(size=(action.shape[0], 1), device='cuda')
                
                obs_dict, r, done, info = self.env_step(self.env, obs_dict, action)
                
                #########################################33
                if self._MODE == 'COLLECT_MATCHING_DB':
                    action = torch.nn.functional.normalize(action)
                    self._amp_obs.append(info['amp_obs'][0, :125].cpu().numpy())
                    self._latent.append(action[0].cpu().numpy())

                    if len(self._amp_obs) % 100 == 0:
                        print(f'collect status : {len(self._amp_obs)} / {self._db_size}, {100 * len(self._amp_obs) / self._db_size} %')
                    if len(self._amp_obs) > self._db_size:
                        self._amp_obs = np.array(self._amp_obs)
                        self._latent = np.array(self._latent)
                        np.save(f'{self._db_name}_s.npy', self._amp_obs)
                        np.save(f'{self._db_name}_z.npy', self._latent)
                        import sys
                        sys.exit(0)

                #########################################33
                if self.NEAREST_MOTION_TEST:
                    time.sleep(0.05)
                    self._current_latent[:] = action[0:1, :]
                    sim = torch.nn.functional.cosine_similarity(self._motions, self._current_latent)
                    val, idx = torch.topk(sim, 5)
                    print('nearest:', self._motion_names[idx[0]]['file'][17:-9], val)
                
                cr += r
                steps += 1
  
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        print(f"game exceeded!: {games_played} >= {n_games}")
                        break
        
                done_indices = done_indices[:, 0]

        print(f"total reward: {sum_rewards}")
        # if print_game_res:
        #     print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        # else:
        #     print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        if self._MODE == 'COLLECT_TEACHER':
            print('> total', len(self._obs_record) * self._obs_record[0].shape[0], 'samples are saved', f'at {self._exp_path}.pkl')

            import pickle
            with open(f'{self._exp_path}.pkl', 'wb') as f:
                pickle.dump({'obs_record': self._obs_record, 
                            'latent_mu_record': self._latent_mu_record,
                            'latent_sigma_record': self._latent_sigma_record,
                            'value_record': self._value_record,
                            'running_mean_std': self.running_mean_std}, 
                            f, protocol=pickle.HIGHEST_PROTOCOL)
        elif self._MODE == 'COLLECT_INIT_CONDITION':
            self._init_ball_states = np.array(self._init_ball_states)
            self._init_dof_pos = np.array(self._init_dof_pos)
            self._init_dof_vel = np.array(self._init_dof_vel)
            self._init_root_states = np.array(self._init_root_states)
            self._init_rb_pos = np.array(self._init_rb_pos)
            self._init_rb_rot = np.array(self._init_rb_rot)
            self._init_rb_vel = np.array(self._init_rb_vel)
            self._init_rb_ang_vel = np.array(self._init_rb_ang_vel)
            
            # under construction
            import pickle
            with open(f'{self._exp_path}_ball_state.pkl', 'wb') as f:
                pickle.dump({'ball_state': self._init_ball_states,
                             'dof_pos': self._init_dof_pos,
                             'dof_vel': self._init_dof_vel,
                             'root_state': self._init_root_states,
                             'rb_pos': self._init_rb_pos,
                             'rb_rot': self._init_rb_rot,
                             'rb_vel': self._init_rb_vel,
                             'rb_ang_vel': self._init_rb_ang_vel,
                            },
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            print('saved', len(self._init_dof_pos))
        return

    def env_step(self, env, obs_dict, action):
        ###############################
        # for pygame input control
        # 평소에는 랜덤으로 모션 따라하다가 특정 키 입력 들어오면 그 모션 재생
        # 평소에는 그냥 idle 모션을 모방하도록 변경하기 (FBC 새로 학습 필요)

        if self._joystick_coroutine:
            buttons, axes, hat = next(self._joystick_coroutine)
            env.task.set_command(buttons, axes, hat)
        ###############################

        if not self.is_tensor_obses:
            action = action.cpu().numpy()

        obs = obs_dict['obs']
        rewards = 0.0
        done_count = 0.0
        disc_rewards = 0.0
        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, action)
            obs, curr_rewards, curr_dones, infos = env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones

            amp_obs = infos['amp_obs']
            curr_disc_reward = self._calc_disc_reward(amp_obs)
            curr_disc_reward = curr_disc_reward[0, 0].cpu().numpy()
            disc_rewards += curr_disc_reward

        rewards /= self._llc_steps
        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0

        disc_rewards /= self._llc_steps

        if isinstance(obs, dict):
            obs = obs['obs']
        if obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return torch.from_numpy(obs).to(self.device), torch.from_numpy(rewards), torch.from_numpy(dones), infos
    
    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']

        network_builder = calm_network_builder.CALMBuilder()

        network_builder.load(network_params)

        network = calm_models.ModelCALMContinuous(network_builder)

        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = calm_players.CALMPlayer(llc_agent_config)

        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])
        llc_env_info['amp_observation_space'] = self.env.amp_observation_space.shape
        llc_env_info['num_envs'] = self.env.task.num_envs
        llc_env_info['num_amp_obs_steps'] = self.env.task._num_amp_obs_steps

        config = config_params['config']
        config['network'] = network
        config['env_info'] = llc_env_info
        config['env'] = self.env

        return config

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim
        return

    def _compute_llc_action(self, obs, actions):
        llc_obs = self._extract_llc_obs(obs)
        processed_obs = self._llc_agent._preproc_obs(llc_obs)

        z = torch.nn.functional.normalize(actions, dim=-1)
        mu, _ = self._llc_agent.model.a2c_network.eval_actor(processed_obs, z)
        # mu, _ = self._llc_agent.model.a2c_network.eval_actor(processed_obs, torch.unsqueeze(z, 0))
        llc_action = players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(mu, -1.0, 1.0))

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs
    
    def _calc_disc_reward(self, amp_obs):
        disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward
