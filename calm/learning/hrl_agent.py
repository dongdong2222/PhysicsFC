
import copy
from gym import spaces
import numpy as np
import os
import yaml

from rl_games.common import a2c_common

import torch
from torch.nn.functional import cosine_similarity

import learning.common_agent as common_agent
import learning.calm_agent as calm_agent
import learning.calm_models as calm_models
import learning.calm_network_builder as calm_network_builder

from .distill_network import SingleDistilModel


class HRLAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']
            self._latent_dim = llc_config_params['config']['latent_dim']
        
        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.task.get_task_obs_size()
        
        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != "")
        self._build_llc(llc_config_params, llc_checkpoint)

        self._MODE = config['mode']
        self._student_ckpt = config['student_ckpt']
        self._input_size = config['input_size']

        ##########################
        if self._MODE == 'TRAIN_REFINE':

            # for refine distilled student model
            self.model = SingleDistilModel(input_size=self._input_size, output_size=64, rnn=False)
            
            # 저장된 파일 로드
            loaded = torch.load(self._student_ckpt)

            # 로드된 모델 구조와 state_dict 사용
            self.model.load_state_dict(loaded['state_dict'])
            
            # setting
            # self.model.a2c_network = self.model
            self.model.to(self.ppo_device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=0.01)

        #################


        return

    def env_step(self, actions):
        #########################################
        # give Full Body latent to env.
        #########################################
        # self.vec_env.env.task.set_latent(actions)
        #########################################

        actions = self.preprocess_actions(actions)
        obs = self.obs['obs']

        rewards = 0.0
        disc_rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0
        for t in range(self._llc_steps):
            # z = torch.nn.functional.normalize(actions, dim=-1)
            
            # for force latent assign - basic HLC
            # self.vec_env.env.task.set_latent_action(actions)
            llc_actions = self._compute_llc_action(obs, actions)
            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)
            
            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos['terminate']
            
            amp_obs = infos['amp_obs']
            curr_disc_reward = self._calc_disc_reward(amp_obs)
            disc_rewards += curr_disc_reward

        rewards /= self._llc_steps
        disc_rewards /= self._llc_steps

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0
        infos['terminate'] = terminate
        infos['disc_rewards'] = disc_rewards

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    # little change
    def _eval_critic(self, obs_dict):
        if self._MODE == 'TRAIN_REFINE':
            self.model.eval()
            obs = obs_dict['obs']
            processed_obs = self._preproc_obs(obs)
            value = self.model.eval_critic(processed_obs) # 

            if self.normalize_value:
                value = self.value_mean_std(value, True)
            return value
        else:
            value = super()._eval_critic(obs_dict)
            return value

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def preprocess_actions(self, actions):
        # print('range', actions.max(), actions.min(), actions.norm(dim=1))
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        clamped_actions = actions
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def play_steps(self):
        self.set_eval()
        
        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            self.experience_buffer.update_data('disc_rewards', n, infos['disc_rewards'])

            style_rewards = self._calc_style_reward(res_dict['actions'])
            self.experience_buffer.update_data('style_rewards', n, style_rewards)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

            # print('style:', style_rewards.sum())

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_disc_rewards = self.experience_buffer.tensor_dict['disc_rewards']
        mb_style_rewards = self.experience_buffer.tensor_dict['style_rewards']

        mb_rewards = self._combine_rewards(mb_rewards, mb_disc_rewards, mb_style_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        
        return batch_dict
    
    def train_epoch(self):
        train_info = super().train_epoch()
        try:
            self.vec_env.env.task.set_difficulty(self.epoch_num)
        except:
            ...
        
        return train_info
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        self._style_reward_w = config['style_reward_w']
        return

    def _get_mean_rewards(self):
        rewards = super()._get_mean_rewards()
        rewards *= self._llc_steps
        return rewards

    def _setup_action_space(self):
        super()._setup_action_space()
        self.actions_num = self._latent_dim
        return

    def init_tensors(self):
        super().init_tensors()

        del self.experience_buffer.tensor_dict['actions']
        del self.experience_buffer.tensor_dict['mus']
        del self.experience_buffer.tensor_dict['sigmas']

        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                    dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['mus'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (self._latent_dim,),
                                                                   dtype=torch.float32, device=self.ppo_device)
        
        self.experience_buffer.tensor_dict['disc_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.experience_buffer.tensor_dict['style_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.tensor_list += ['disc_rewards', 'style_rewards']

        return

    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']

        network_builder = calm_network_builder.CALMBuilder()

        network_builder.load(network_params)

        network = calm_models.ModelCALMContinuous(network_builder)

        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = calm_agent.CALMAgent('llc', llc_agent_config)

        self._llc_agent.restore(checkpoint_file)
        print("Loaded LLC checkpoint from {:s}".format(checkpoint_file))
        self._llc_agent.set_eval()

        enc_amp_obs = self._llc_agent._fetch_amp_obs_demo(128)
        if len(enc_amp_obs) == 2:
            enc_amp_obs = enc_amp_obs[0]

        preproc_enc_amp_obs = self._llc_agent._preproc_amp_obs(enc_amp_obs)
        self.encoded_motion = self._llc_agent.model.a2c_network.eval_enc(amp_obs=preproc_enc_amp_obs).unsqueeze(0)

        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        config = config_params['config']
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer': self.algo_observer}
        config['env_info'] = llc_env_info
        config['minibatch_size'] = 1
        config['amp_batch_size'] = 32
        config['amp_minibatch_size'] = 1
        config['enable_eps_greedy'] = False
        config['vec_env'] = self.vec_env

        return config

    def _compute_llc_action(self, obs, actions):
        llc_obs = self._extract_llc_obs(obs)
        processed_obs = self._llc_agent._preproc_obs(llc_obs)

        z = torch.nn.functional.normalize(actions, dim=-1)
        mu, _ = self._llc_agent.model.a2c_network.eval_actor(processed_obs, z)
        llc_action = mu
        llc_action = self._llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs

    def _calc_disc_reward(self, amp_obs):
        disc_reward = self._llc_agent._calc_disc_rewards(amp_obs)
        return disc_reward

    def _calc_style_reward(self, action):
        z = torch.nn.functional.normalize(action, dim=-1)

        tmp = self.encoded_motion.squeeze()
        tmp = tmp.repeat((self.num_actors // 128, 1))
        tmp = tmp.unsqueeze(1)

        style_reward = torch.max((cosine_similarity(z.unsqueeze(1), tmp, dim=-1) + 1) / 2, dim=1)[0]
        # style_reward = torch.max((cosine_similarity(z.unsqueeze(1), self.encoded_motion, dim=-1) + 1) / 2, dim=1)[0]
        return style_reward.unsqueeze(-1)

    def _combine_rewards(self, task_rewards, disc_rewards, style_rewards):
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_rewards + self._style_reward_w * style_rewards

        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        train_info['style_rewards'] = batch_dict['style_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)

        style_reward_std, style_reward_mean = torch.std_mean(train_info['style_rewards'])
        self.writer.add_scalar('info/style_reward_mean', style_reward_mean.item(), frame)
        self.writer.add_scalar('info/style_reward_std', style_reward_std.item(), frame)
        return
