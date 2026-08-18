from . import gamepad

##########################################


import torch

from isaacgym.torch_utils import *
from rl_games.algos_torch import players

from learning import amp_players

import numpy as np
import time

query_freq = 1
virtual_mode = False
agility_test = False

COLLECT_LATENT_TRAJ = False
latent_traj = list()
traj_len = 200

IS_MOTION_STYLE = True
MOTION = 70


class CALMPlayer(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        self._latent_dim = config['latent_dim']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)
        self._sample_latent_only_on_reset = config.get('latents_on_reset', None)  # if interpolate latents -- this should be set to false
        self._interpolate_latents = config.get('interpolate_latents', None)
        self._get_latents_from_data = config.get('latents_from_data', None)

        self._conditional_disc_reward_scale = config['conditional_disc_reward_scale']
        self._amp_batch_size = int(config['amp_batch_size'])

        if 'env' in config:
            self.env = config['env']

        super().__init__(config)

        if hasattr(self, 'env'):
            batch_size = self.env.task.num_envs
        else:
            batch_size = self.env_info['num_envs']
        self._calm_latents = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)

        if self._interpolate_latents is True:
            self._latents0 = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)
            self._latents1 = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)

            self._interpolation_alpha = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

            self._interpolation_steps = 100
            self.env.task.max_episode_length = 500

        return

    def run(self):
        self._reset_latent_step_count()

        self.model.a2c_network._env = self.env
        super().run()
        return

    def _fetch_amp_obs_demo(self, num_samples):
        motion_ids, _, enc_amp_obs_demo_flat, _, _ = self.env.fetch_amp_obs_demo_enc_pair(num_samples)
        return motion_ids, enc_amp_obs_demo_flat

    def get_action(self, obs_dict, is_determenistic=False):
        if not self._sample_latent_only_on_reset:
            self._update_latents()

        obs = obs_dict['obs']
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        calm_latents = self._calm_latents

        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states,
            'calm_latents': calm_latents
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
        return players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)

        self._reset_latents(env_ids)
        return obs

    def _build_net_config(self):
        config = super()._build_net_config()
        config['calm_latent_shape'] = (self._latent_dim,)
        print(config)
        return config

    def _reset_latents(self, done_env_ids=None):
        if done_env_ids is None:
            num_envs = self.env.task.num_envs
            done_env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)

        num_samples = len(done_env_ids)

        if num_samples > 0:
            if self._interpolate_latents:
                num_samples *= 2

            if self._get_latents_from_data:
                z, motion_ids = self._sample_latents(num_samples)
            else:
                z = torch.normal(torch.zeros([num_samples, self._latent_dim], device=self.device))

            z = torch.nn.functional.normalize(z, dim=-1)

            if self._interpolate_latents:
                z, z1 = torch.chunk(z, 2)
                self._latents0[done_env_ids] = z
                self._latents1[done_env_ids] = z1
                self._interpolation_alpha[done_env_ids] = 1.

            self._calm_latents[done_env_ids] = z
            self._change_char_color(done_env_ids)
            # print(self._calm_latents)

        return

    def _sample_latents(self, n):
        # 1. sample motion id
        # 2. create obs from entire sequence
        # 3. encode

        motion_ids, new_enc_amp_obs_demo = self._fetch_amp_obs_demo(n)

        proc_new_enc_amp_obs = self._preproc_amp_obs(new_enc_amp_obs_demo)
        with torch.no_grad():
            encoded_demo_amp_obs = self.model.a2c_network.eval_enc(proc_new_enc_amp_obs)

        # if we're interpolating from data, let's make it visually appealing by forcing the main character to have
        # two different motion files to interpolavte from.
        motion_indices = motion_ids
        if self._interpolate_latents and motion_ids[0].item() == motion_ids[n // 2].item():
            for idx in range(1, n):
                if motion_ids[idx].item() != motion_ids[0].item():
                    encoded_demo_amp_obs[n // 2] = encoded_demo_amp_obs[idx]
                    break

        return encoded_demo_amp_obs, motion_indices

    def _update_latents(self):
        if self._latent_step_count <= 0:
            if self._interpolate_latents:
                self._interpolation_alpha -= 1. / self._interpolation_steps

                clamped_alpha = torch.clamp_min(self._interpolation_alpha, min=0)
                self._calm_latents = torch.nn.functional.normalize(self._latents0 * clamped_alpha + self._latents1 * (1 - clamped_alpha), dim=1)
            else:
                self._reset_latents()
            self._reset_latent_step_count()

            if self.env.task.viewer:
                print("Sampling new calm latents------------------------------")
                num_envs = self.env.task.num_envs
                env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)
                self._change_char_color(env_ids)
        else:
            self._latent_step_count -= 1
        return

    def _reset_latent_step_count(self):
        if self._interpolate_latents:
            self._latent_step_count = self.env.task.max_episode_length // self._interpolation_steps
        else:
            self._latent_step_count = np.random.randint(self._latent_steps_min, self._latent_steps_max)
        return

    def _calc_amp_rewards(self, amp_obs, calm_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        cdisc_r = self._calc_conditional_disc_rewards(amp_obs, calm_latents)
        output = {
            'disc_rewards': disc_r,
            'conditional_disc_rewards': cdisc_r
        }
        return output

    def _calc_conditional_disc_rewards(self, amp_obs, calm_latents):
        with torch.no_grad():
            disc_logits = self._eval_conditional_disc(amp_obs, calm_latents)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._conditional_disc_reward_scale

        return disc_r

    def _eval_conditional_disc(self, amp_obs, calm_latents):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_conditional_disc(proc_amp_obs, calm_latents)

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            calm_latents = self._calm_latents
            disc_pred = self._eval_conditional_disc(amp_obs, calm_latents)
            amp_rewards = self._calc_amp_rewards(amp_obs, calm_latents)
            disc_reward = amp_rewards['disc_rewards']
            cdisc_reward = amp_rewards['conditional_disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            cdisc_reward = cdisc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward, cdisc_reward)
        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.env.task.set_char_color(rand_col, env_ids)
        return


class CALMPlayerLab(amp_players.AMPPlayerContinuous):
    
    motion_mapper = {
        'run': 9,
        'jump': 44,
        'pass': 14,
        'chest': 22,
        'shoot': 15,
        'dribble': 31,
        'cmove_r': 21,
        'cmove_l': 50,
        'back': 4,
        'jog': 23

    }
    def __init__(self, config):
        # if virtual mode, emulate joystick.
        # if not, defualt motion assign would be used.
        self._joystick = gamepad.Joystick(joystick_idx=0, virtual_mode=virtual_mode)
        self._joystick_coroutine = self._joystick.command() if self._joystick.is_available() or self._joystick.is_virtual_mode() else None
        
        # for only agility test
        if agility_test:
            self._tester_coroutine = self._joystick.test_agility()
            next(self._tester_coroutine)

        self._latent_dim = config['latent_dim']
        self._latent_steps_min = 0 #config.get('latent_steps_min', np.inf)
        self._latent_steps_max = 1000 #config.get('latent_steps_max', np.inf)
        self._sample_latent_only_on_reset = config.get('latents_on_reset', None)  # if interpolate latents -- this should be set to false
        self._interpolate_latents = config.get('interpolate_latents', None)
        self._get_latents_from_data = config.get('latents_from_data', None)

        self._conditional_disc_reward_scale = config['conditional_disc_reward_scale']
        self._amp_batch_size = int(config['amp_batch_size'])

        if not IS_MOTION_STYLE:
            self._motion_alias = self.motion_mapper[config.get('motion_alias', None)]
        else:
            self._motion_alias = MOTION

        if 'env' in config:
            self.env = config['env']

        super().__init__(config)

        if hasattr(self, 'env'):
            batch_size = self.env.task.num_envs
        else:
            batch_size = self.env_info['num_envs']
        self._calm_latents = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)

        if self._interpolate_latents is True:
            self._latents0 = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)
            self._latents1 = torch.zeros((batch_size, self._latent_dim), dtype=torch.float32, device=self.device)

            self._interpolation_alpha = torch.ones((batch_size, 1), dtype=torch.float32, device=self.device)

            self._interpolation_steps = 10
            self.env.task.max_episode_length = 200

        ###########################
        from env.feature_db_2 import MotionDB_2
        # from env.feature_db import MotionDB

        self._MODE = config['mode']
        if self._MODE in ['ORIGINAL_MATCHING', 'SKILL_MATCHING']:
            feature_path = config['feature_path']
            latent_path = config['latent_path']
            self._feature_path = feature_path
            if self._MODE == 'SKILL_MATCHING':
                self._latent_path = latent_path
            else:
                self._latent_path = None
            
            self._mdb = MotionDB_2(feature_path, latent_path, mode=self._MODE)

        return

    def run(self):
        self._reset_latent_step_count()

        self.model.a2c_network._env = self.env
        super().run()
        return

    def _fetch_amp_obs_demo(self, num_samples):
        motion_ids, _, enc_amp_obs_demo_flat, _, _ = self.env.fetch_amp_obs_demo_enc_pair(num_samples)
        return motion_ids, enc_amp_obs_demo_flat

    def _fetch_amp_obs_demo_lab(self, motion_ids):
        motion_ids, _, enc_amp_obs_demo_flat, _, _ = self.env.fetch_amp_obs_demo_enc_pair_lab(motion_ids)
        return motion_ids, enc_amp_obs_demo_flat

    def _fetch_amp_obs_demo_lab_query(self, motion_ids):
        motion_ids, _, enc_amp_obs_demo_flat = self.env.fetch_amp_obs_demo_enc_pair_lab_query(motion_ids)
        return motion_ids, enc_amp_obs_demo_flat

    def get_action(self, obs_dict, is_determenistic=False):
        ###############################
        # for pygame input control
        # 평소에는 랜덤으로 모션 따라하다가 특정 키보드 입력 들어오면 그 모션 재생

        if self._joystick_coroutine:
            ret = next(self._joystick_coroutine)
            if ret and len(ret) == 1:
                print(ret)
                if ret == 'up':
                    self._motion_alias = 9
                elif ret == 'down':
                    self._motion_alias = 4
                elif ret == 'right':
                    self._motion_alias = 36
                elif ret == 'left':
                    self._motion_alias = 40
                elif ret == 'q':
                    self._motion_alias = 33
                elif ret == 'w':
                    self._motion_alias = 13
                elif ret == 'e':
                    self._motion_alias = 45
                elif ret == 'r':
                    self._motion_alias = 8
                elif ret == 'a':
                    self._motion_alias = 51
                elif ret == 's':
                    self._motion_alias = 70
                elif ret == 'd':
                    self._motion_alias = 64
                elif ret == 'f':
                    self._motion_alias = 67
            
                z, motion_ids = self._sample_latents(1)
                z = torch.nn.functional.normalize(z, dim=-1)
                self._calm_latents[:] = z
            ###############################


        if not self._sample_latent_only_on_reset:
            self._update_latents()

        obs = obs_dict['obs']
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)
        obs = self._preproc_obs(obs)
        calm_latents = self._calm_latents

        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states,
            'calm_latents': calm_latents
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

        # current_action += torch.normal(0, 0.05, size=current_action.shape, device=self.device)

        return players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        
        ###############################
        # for joystick control
        start = time.time()
        if self._joystick_coroutine:
            buttons, axes, hat = next(self._joystick_coroutine)
            env.task.set_command(buttons, axes, hat)
        ###############################
        # print('joystick:', time.time() - start)

        
        #######################################
        # Excute Latent Matching
        if (self._joystick.is_available() or self._joystick.is_virtual_mode()) and env.task.progress_buf[0] % query_freq == 0:
            if self._joystick.is_virtual_mode() and env.task.progress_buf[0] % 1 == 0:
                
                # for only agility test
                if agility_test:
                    # next(self._tester_coroutine)
                    self._tester_coroutine.send(self.env.task._root_states)
                # for no agility test
                else:
                    self._joystick.update_virtual_command()

            start = time.time()
            runtime_feature = env.task.build_feature_single_2() # env.task.build_feature_single()
            # print('build feature:', time.time() - start)

            start = time.time()
            nearest_motion, nearest_frame = self._mdb.query(runtime_feature)
            # print('query:', time.time() - start)

            # print('convert time:', time.time() - start)
            if self._MODE == 'ORIGINAL_MATCHING':
                start = time.time()
                
                z = self._mdb.get_latent_code(nearest_motion.cpu().numpy())
                
                # print('sample:', time.time() - start)
                
                # for multiple latents weighted sum. rearest_frame = 유클리디안 거리
                start = time.time()

                nearest_frame = 1 / (nearest_frame * nearest_frame)
                s = torch.sum(nearest_frame)
                norm = torch.tensor(nearest_frame / s, device=self.device).unsqueeze(dim=1)
                self._calm_latents[:] = torch.sum(z * norm, dim=0)

                #######
                # for recording latent traj
                ########
                if COLLECT_LATENT_TRAJ:
                    latent_traj.append(self._calm_latents.cpu().numpy())
                    if len(latent_traj) >= traj_len:
                        import sys
                        import pickle
                        
                        with open('./latent_traj.pkl', 'wb') as f:
                            pickle.dump(latent_traj, f, protocol=pickle.HIGHEST_PROTOCOL)

                        sys.exit(0)
                
                # print('IDW:', time.time() - start)

                # self._calm_latents[:] = z
                # print('nearest motion:', self._mdb._motion_names[nearest_motion])
            elif self._MODE == 'SKILL_MATCHING':
                z = self._mdb.get_latent_code(nearest_frame)
                z = torch.nn.functional.normalize(z, dim=-1)
                self._calm_latents[:] = z

        #######################################

        start = time.time()
        obs, rewards, dones, infos = env.step(actions)
        # print('step:', time.time() - start)

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)

        self._reset_latents(env_ids)
        return obs

    def _build_net_config(self):
        config = super()._build_net_config()
        config['calm_latent_shape'] = (self._latent_dim,)

        return config

    def _reset_latents(self, done_env_ids=None):
        if done_env_ids is None:
            num_envs = self.env.task.num_envs
            done_env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)

        num_samples = len(done_env_ids)

        if num_samples > 0:
            if self._interpolate_latents:
                num_samples *= 2

            if self._get_latents_from_data:
                z, motion_ids = self._sample_latents(num_samples)
            else:
                z = torch.normal(torch.zeros([num_samples, self._latent_dim], device=self.device))

            # for save latents
            # np.save('latents_90_new_unnorm.npy', z.cpu().numpy())

            # for noise add test
            # z += torch.randn(1, 64, device='cuda') * .01
            
            z = torch.nn.functional.normalize(z, dim=-1)

            if self._interpolate_latents:
                z, z1 = torch.chunk(z, 2)
                self._latents0[done_env_ids] = z
                self._latents1[done_env_ids] = z1
                self._interpolation_alpha[done_env_ids] = 1.

            # self._calm_latents[done_env_ids] = z
            self._calm_latents[done_env_ids] = torch.mean(z, dim=0)
            self._change_char_color(done_env_ids)

        return

    def _sample_latents(self, n):
        # 1. sample motion id
        # 2. create obs from entire sequence
        # 3. encode

        if self._interpolate_latents:
            motion_ids = torch.tensor([34, 9], device='cuda:0')
        else:
            motion_ids = torch.tensor([self._motion_alias], device='cuda:0')
        
        # for save latents
        # motion_ids = torch.arange(90, device=self.device)
        
        # motion_ids, new_enc_amp_obs_demo = self._fetch_amp_obs_demo(n)
        motion_ids, new_enc_amp_obs_demo = self._fetch_amp_obs_demo_lab(motion_ids=motion_ids)

        proc_new_enc_amp_obs = self._preproc_amp_obs(new_enc_amp_obs_demo)
        with torch.no_grad():
            encoded_demo_amp_obs = self.model.a2c_network.eval_enc(proc_new_enc_amp_obs)

        # if we're interpolating from data, let's make it visually appealing by forcing the main character to have
        # two different motion files to interpolate from.
        motion_indices = motion_ids
        if self._interpolate_latents and motion_ids[0].item() == motion_ids[n // 2].item():
            for idx in range(1, n):
                if motion_ids[idx].item() != motion_ids[0].item():
                    encoded_demo_amp_obs[n // 2] = encoded_demo_amp_obs[idx]
                    break

        return encoded_demo_amp_obs, motion_indices

    def _sample_latents_query(self):
        motion_ids = self._motion_alias
        
        motion_ids, new_enc_amp_obs_demo = self._fetch_amp_obs_demo_lab_query(motion_ids=motion_ids)

        proc_new_enc_amp_obs = self._preproc_amp_obs(new_enc_amp_obs_demo)
        with torch.no_grad():
            encoded_demo_amp_obs = self.model.a2c_network.eval_enc(proc_new_enc_amp_obs)

        return encoded_demo_amp_obs, motion_ids

    def _update_latents(self):
        if self._latent_step_count <= 0:
            if self._interpolate_latents:
                self._interpolation_alpha -= 1. / self._interpolation_steps

                clamped_alpha = torch.clamp_min(self._interpolation_alpha, min=0)
                self._calm_latents = torch.nn.functional.normalize(self._latents0 * clamped_alpha + self._latents1 * (1 - clamped_alpha), dim=1)
            else:
                self._reset_latents()
            self._reset_latent_step_count()

            if self.env.task.viewer:
                # print("Sampling new calm latents------------------------------")
                num_envs = self.env.task.num_envs
                env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)
                self._change_char_color(env_ids)
        else:
            self._latent_step_count -= 1
        return

    def _reset_latent_step_count(self):
        if self._interpolate_latents:
            self._latent_step_count = self.env.task.max_episode_length // self._interpolation_steps
        else:
            self._latent_step_count = self._latent_steps_max #np.random.randint(self._latent_steps_min, self._latent_steps_max)
        return

    def _calc_amp_rewards(self, amp_obs, calm_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        cdisc_r = self._calc_conditional_disc_rewards(amp_obs, calm_latents)
        output = {
            'disc_rewards': disc_r,
            'conditional_disc_rewards': cdisc_r
        }
        return output

    def _calc_conditional_disc_rewards(self, amp_obs, calm_latents):
        with torch.no_grad():
            disc_logits = self._eval_conditional_disc(amp_obs, calm_latents)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._conditional_disc_reward_scale

        return disc_r

    def _eval_conditional_disc(self, amp_obs, calm_latents):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_conditional_disc(proc_amp_obs, calm_latents)

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            calm_latents = self._calm_latents
            disc_pred = self._eval_conditional_disc(amp_obs, calm_latents)
            amp_rewards = self._calc_amp_rewards(amp_obs, calm_latents)
            disc_reward = amp_rewards['disc_rewards']
            cdisc_reward = amp_rewards['conditional_disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            cdisc_reward = cdisc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward, cdisc_reward)
        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.env.task.set_char_color(rand_col, env_ids)
        return

