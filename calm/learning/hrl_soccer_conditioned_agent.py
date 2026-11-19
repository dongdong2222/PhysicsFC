import numpy as np

import torch
from torch.nn.functional import cosine_similarity

from learning.hrl_agent import HRLAgent


class HRLSoccerConditionedAgent(HRLAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        # for strong regularize
        # self.optimizer = optim.AdamW(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=0.1)

        return

    def _build_llc(self, config_params, checkpoint_file):
        super()._build_llc(config_params, checkpoint_file)
        
        # 사용할 모션들 미리 뽑아놓기. 쓸 것들만 yaml 파일 따로 만들 필요 있음
        self.encoded_motion = self._get_motion_encoding()

    def _get_motion_encoding(self):
        all_encoded_demo_amp_obs = []
        for motion_id in range(self.vec_env.env.task._motion_lib._motion_weights.shape[0]):
            motion_amp_obs = self.vec_env.env.task.fetch_amp_obs_demo_per_id(32, motion_id)[-1].view(32, self.vec_env.env.task._num_amp_obs_enc_steps, self.vec_env.env.task._num_amp_obs_per_step)

            preproc_amp_obs = self._llc_agent._preproc_amp_obs(motion_amp_obs)
            encoded_demo_amp_obs = self._llc_agent.model.a2c_network.eval_enc(preproc_amp_obs)
            all_encoded_demo_amp_obs.append(encoded_demo_amp_obs)
        all_encoded_demo_amp_obs = torch.stack(all_encoded_demo_amp_obs, dim=0)

        return all_encoded_demo_amp_obs

    def _calc_style_reward(self, action):
        # env 에서 지정된 모션 가져와서 비교할 거
        # situation 이 지정된 env 에만 적용해야함
        requested_style = self.vec_env.env.task._tar_locomotion_index.view(-1)
        style_reward = torch.zeros((action.shape[0]), device=self.ppo_device, dtype=torch.float32)
        z = torch.nn.functional.normalize(action, dim=-1)

        # maybe more than 3
        # 여기서 style 지정 안 된 env 의 style 은 무조건 1 로 설정하기
        # 하드코딩 된 스타일 종류
        for motion_type in range(len(self.vec_env.env.task._motion_ids)):
            motion_mask = requested_style == motion_type
            
            tmp = self.encoded_motion[motion_type][0].repeat((torch.sum(motion_mask), 1)).view(-1, 1, self._latent_dim)

            # style_reward[motion_mask] = torch.max((cosine_similarity(z[motion_mask].view(-1, 1, self._latent_dim), self.encoded_motion[motion_type].view(1, -1, self._latent_dim), dim=-1) + 1) / 2, dim=1)[0]
            style_reward[motion_mask] = torch.max((cosine_similarity(z[motion_mask].view(-1, 1, self._latent_dim), tmp, dim=-1) + 1) / 2, dim=1)[0]

        # style 컨디션이 없는 경우에 style_reward 를 0 으로 맞춰놓기. 대신 task reward 는 2 배로
        style_reward[requested_style==-1] = 0
        
        return style_reward.unsqueeze(-1)

    def _calc_disc_reward(self, amp_obs):
        requested_motion_indices = self.vec_env.env.task._tar_locomotion_index.view(-1)

        disc_reward = torch.zeros((amp_obs.shape[0], 1), device=self.ppo_device, dtype=torch.float32)

        latent_index = np.random.randint(low=0, high=32)

        for motion_type in range(3):  # TODO: currently hardcoded to 3 motions.
            motion_mask = requested_motion_indices == motion_type
            disc_reward[motion_mask] = self._llc_agent._calc_conditional_disc_rewards(amp_obs[motion_mask], self.encoded_motion[0][latent_index].view(1, self._latent_dim).expand(amp_obs.shape[0], -1)[motion_mask])

        return disc_reward
