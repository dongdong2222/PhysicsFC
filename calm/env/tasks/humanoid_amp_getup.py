# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from utils import torch_utils

from env.tasks.humanoid_amp import HumanoidAMP, dof_to_obs
from isaacgym.torch_utils import *

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2

################################################
from env.feature_db import MotionDB
from env.feature_db_2 import MotionDB_2
################################################


class HumanoidAMPGetup(HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        
        self._recovery_episode_prob = cfg["env"]["recoveryEpisodeProb"]
        self._recovery_steps = cfg["env"]["recoverySteps"]
        self._fall_init_prob = cfg["env"]["fallInitProb"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]

        self._reset_fall_env_ids = []

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._recovery_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        #####################################
        # extract and save for motion DB
        #####################################
        # self._extract_features()
        self._feature = torch.ones(size=(34,), device=self.device) # torch.zeros(size=(21,))

        # for marker
        if not self.headless:
            self._build_marker_state_tensors()
        self._tar_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_dir[..., 0] = 1.0

        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

        self._generate_fall_states()

        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        self._update_recovery_count()
        return

    def _generate_fall_states(self):
        max_steps = 150
        
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        root_states = self._initial_humanoid_root_states[env_ids].clone()
        root_states[..., 3:7] = torch.randn_like(root_states[..., 3:7])
        root_states[..., 3:7] = torch.nn.functional.normalize(root_states[..., 3:7], dim=-1)
        self._humanoid_root_states[env_ids] = root_states
        
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        rand_actions = np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()])
        rand_actions = to_torch(rand_actions, device=self.device)
        self.pre_physics_step(rand_actions)

        # step physics and render each frame
        for i in range(max_steps):
            self.render()
            self.gym.simulate(self.sim)
            
        self._refresh_sim_tensors()
        
        self._fall_root_states = self._humanoid_root_states.clone()
        self._fall_root_states[:, 7:13] = 0
        self._fall_dof_pos = self._dof_pos.clone()
        self._fall_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        return

    def _reset_actors(self, env_ids):
        num_envs = env_ids.shape[0]
        recovery_probs = to_torch(np.array([self._recovery_episode_prob] * num_envs), device=self.device)
        recovery_mask = torch.bernoulli(recovery_probs) == 1.0
        terminated_mask = (self._terminate_buf[env_ids] == 1)
        recovery_mask = torch.logical_and(recovery_mask, terminated_mask)

        recovery_ids = env_ids[recovery_mask]
        if len(recovery_ids) > 0:
            self._reset_recovery_episode(recovery_ids)
            

        nonrecovery_ids = env_ids[torch.logical_not(recovery_mask)]
        fall_probs = to_torch(np.array([self._fall_init_prob] * nonrecovery_ids.shape[0]), device=self.device)
        fall_mask = torch.bernoulli(fall_probs) == 1.0
        fall_ids = nonrecovery_ids[fall_mask]
        if len(fall_ids) > 0:
            self._reset_fall_episode(fall_ids)
            

        nonfall_ids = nonrecovery_ids[torch.logical_not(fall_mask)]
        if len(nonfall_ids) > 0:
            super()._reset_actors(nonfall_ids)
            self._recovery_counter[nonfall_ids] = 0

        return

    def _reset_recovery_episode(self, env_ids):
        self._recovery_counter[env_ids] = self._recovery_steps
        return
    
    def _reset_fall_episode(self, env_ids):
        fall_state_ids = torch.randint_like(env_ids, low=0, high=self._fall_root_states.shape[0])
        self._humanoid_root_states[env_ids] = self._fall_root_states[fall_state_ids]
        self._dof_pos[env_ids] = self._fall_dof_pos[fall_state_ids]
        self._dof_vel[env_ids] = self._fall_dof_vel[fall_state_ids]
        self._recovery_counter[env_ids] = self._recovery_steps
        self._reset_fall_env_ids = env_ids
        return
    
    def _reset_envs(self, env_ids):
        self._reset_fall_env_ids = []
        super()._reset_envs(env_ids)
        return

    def _init_amp_obs(self, env_ids):
        super()._init_amp_obs(env_ids)

        if len(self._reset_fall_env_ids) > 0:
            self._init_amp_obs_default(self._reset_fall_env_ids)

        return

    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return

    def _compute_reset(self):
        super()._compute_reset()

        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        return

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _extract_features(self):
        motion_ids = torch.tensor(list(range(49)), device=self.device)
        dt = self.dt
        _num_amp_obs_steps = 120 # tmp. if motion has under 120 frames, need to be handled
        # time must be start from zero
        motion_times = torch.zeros_like(motion_ids, device=self.device)

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, _num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        # time must me positive
        time_steps = dt * (torch.arange(0, _num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        amp_obs_demo = build_amp_observations_2(root_pos, root_rot, root_vel, root_ang_vel, 
                                        dof_pos, dof_vel, key_pos, 
                                        self._local_root_obs, self._root_height_obs, 
                                        self._dof_obs_size, self._dof_offsets)
        print('*** features for Latent Matching are saved! ***', amp_obs_demo.shape)
        np.save('motion_features_locomotion_only_jump.npy', amp_obs_demo.cpu().numpy())

    def build_feature_single(self):
        #######################################
        # joystick input handle
        #######################################
        # if self._axes:
        left_x, left_y, left_trigger, right_x, right_y, right_trigger = self._axes
        left_x, left_y, left_trigger, right_x, right_y, right_trigger = torch.tensor(left_x, device=self.device), torch.tensor(left_y, device=self.device), torch.tensor(left_trigger, device=self.device), torch.tensor(right_x, device=self.device), torch.tensor(right_y, device=self.device), torch.tensor(right_trigger, device=self.device)
        magnitude_left = torch.clamp(torch.sqrt(torch.pow(left_x, 2) + torch.pow(left_y, 2)), 0, 1)
        magnitude_right = torch.sqrt(torch.pow(right_x, 2) + torch.pow(right_y, 2))
        left_axis = torch.tensor((left_x, -left_y, 0), dtype=torch.float) * self._tar_speed_max
        right_axis = torch.tensor((left_x, -left_y, 0), dtype=torch.float)
        
        rand_theta = torch.atan2(-left_y, left_x)
        if magnitude_right > .1:
            right_axis = torch.tensor((right_x, -right_y, 0), dtype=torch.float)
            rand_face_theta = torch.atan2(torch.tensor(-right_y, device=self.device), torch.tensor(right_x, device=self.device))
        else:
            rand_face_theta = rand_theta
        # 왼쪽 조이스틱 값으로 각도 계산

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)
        self._tar_dir = tar_dir
        self._tar_facing_dir = face_tar_dir
        
        # fixed input control
        # else:
        # left_axis = torch.tensor((5, 0, 0), dtype=torch.float)# * self._tar_speed_max
        # right_axis = torch.tensor((1, 0, 0), dtype=torch.float)
        #
        #######################################
        # joystick input handle
        #######################################
        offset = 10
        interp_ratio_1 = 1.0
        interp_ratio_2 = 1.0
        key_body_pos = self._rigid_body_pos[:1, self._key_body_ids, :]
        obs = build_amp_observations_2(self._rigid_body_pos[:1, 0, :],
                                        self._rigid_body_rot[:1, 0, :],
                                        self._rigid_body_vel[:1, 0, :],
                                        self._rigid_body_ang_vel[:1, 0, :],
                                        self._dof_pos[:1], self._dof_vel[:1], key_body_pos,
                                        self._local_root_obs, self._root_height_obs, 
                                        self._dof_obs_size, self._dof_offsets)
        obs = obs.cpu()
        self._feature[:12] = obs[0, -12:]
        self._feature[12:15] = obs[0, 7:10]
        rotation = torch.tensor(obs[0, 3:7]).unsqueeze(0)
        heading_rot = torch_utils.calc_heading_quat_inv(torch.tensor(rotation)).cpu()
        heading_rot_inv = torch_utils.calc_heading_quat(torch.tensor(rotation)).cpu()
        
        # only joystick control input affects future pos, rot
        vel_joystick = torch_utils.quat_rotate(heading_rot, left_axis.unsqueeze(0))
        vel_sum = (1-interp_ratio_1) * obs[0, 7:10] + interp_ratio_1 * vel_joystick # (obs[0, 7:10] + vel_joystick) / 2
        # print(vel_joystick, vel_sum)
        future_pos = vel_sum * self.dt * offset
        self._feature[15:17] = future_pos[0, :2]

        rot = np.arctan2(right_axis[1], right_axis[0]) * 180 / np.pi
        quat_joystick = torch_utils.quat_from_angle_axis((rot.clone() * np.pi / 180).unsqueeze(0), torch.tensor([0, 0, 1], dtype=torch.float).unsqueeze(0))
        target_rot = torch_utils.quat_mul(heading_rot, quat_joystick)
        target_rot_compare = torch_utils.quat_mul(heading_rot, heading_rot_inv)
        target_rot = torch_utils.slerp(target_rot_compare, target_rot, torch.tensor(interp_ratio_2)) #
        # a = torch_utils.quat_to_angle_axis(target_rot)[0] * 180 / torch.pi
        # b = torch_utils.quat_to_angle_axis(target_rot_compare)[0] * 180 / torch.pi
        futures_local = calc_heading_xy(target_rot)
        self._feature[17] = futures_local[0]
        self._feature[18] = futures_local[1]

        # self._feature[19] = vel_joystick[0, 0]
        # self._feature[20] = vel_joystick[0, 1]

        return self._feature


    # 점프를 고려한 feature
    def build_feature_single_2(self):
        #######################################
        # joystick input handle
        #######################################
        # if self._axes:

        left_x, left_y, left_trigger, right_x, right_y, right_trigger = self._axes
        jump = self._buttons[0]
        left_x, left_y, left_trigger, right_x, right_y, right_trigger = torch.tensor(left_x, device=self.device), torch.tensor(left_y, device=self.device), torch.tensor(left_trigger, device=self.device), torch.tensor(right_x, device=self.device), torch.tensor(right_y, device=self.device), torch.tensor(right_trigger, device=self.device)
        magnitude_left = torch.clamp(torch.sqrt(torch.pow(left_x, 2) + torch.pow(left_y, 2)), 0, 1)
        magnitude_right = torch.sqrt(torch.pow(right_x, 2) + torch.pow(right_y, 2))
        left_axis = torch.tensor((left_x, -left_y, 0), dtype=torch.float) * self._tar_speed_max
        right_axis = torch.tensor((left_x, -left_y, 0), dtype=torch.float)

        rand_theta = torch.atan2(-left_y, left_x)
        if magnitude_right > .1:
            right_axis = torch.tensor((right_x, -right_y, 0), dtype=torch.float)
            rand_face_theta = torch.atan2(torch.tensor(-right_y, device=self.device), torch.tensor(right_x, device=self.device))
        else:
            rand_face_theta = rand_theta
        # 왼쪽 조이스틱 값으로 각도 계산
        
        # print(left_axis, right_axis, jump)

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)
        self._tar_dir = tar_dir
        self._tar_facing_dir = face_tar_dir
        
        # fixed input control
        # else:
        # left_axis = torch.tensor((5, 0, 0), dtype=torch.float)# * self._tar_speed_max
        # right_axis = torch.tensor((1, 0, 0), dtype=torch.float)
        #
        #######################################
        # joystick input handle
        #######################################
        offset = 3
        interp_ratio_1 = .5
        interp_ratio_2 = .7
        interp_ratio_3 = .99
        key_body_pos = self._rigid_body_pos[:1, self._key_body_ids, :]
        obs = build_amp_observations_2(self._rigid_body_pos[:1, 0, :],
                                        self._rigid_body_rot[:1, 0, :],
                                        self._rigid_body_vel[:1, 0, :],
                                        self._rigid_body_ang_vel[:1, 0, :],
                                        self._dof_pos[:1], self._dof_vel[:1], key_body_pos,
                                        self._local_root_obs, self._root_height_obs, 
                                        self._dof_obs_size, self._dof_offsets)
        obs = obs.cpu()
        self._feature[:12] = obs[0, -12:]
        self._feature[12:16] = obs[0, 3:7]
        self._feature[16:19] = obs[0, 7:10]
        rotation = torch.tensor(obs[0, 3:7]).unsqueeze(0)
        heading_rot = torch_utils.calc_heading_quat_inv(torch.tensor(rotation)).cpu()
        heading_rot_inv = torch_utils.calc_heading_quat(torch.tensor(rotation)).cpu()
        
        # only joystick control input affects future pos, rot
        vel_joystick = torch_utils.quat_rotate(heading_rot, left_axis.unsqueeze(0))
        vel_z = torch.tensor([[0, 0, 3]]) if jump else torch.tensor([[0, 0, 0]])
        vel_joystick += vel_z
        vel_sum = (1-interp_ratio_1) * obs[0, 7:10] + interp_ratio_1 * vel_joystick # (obs[0, 7:10] + vel_joystick) / 2
        vel_sum_2 = (1-interp_ratio_2) * obs[0, 7:10] + interp_ratio_2 * vel_joystick # (obs[0, 7:10] + vel_joystick) / 2
        vel_sum_3 = (1-interp_ratio_3) * obs[0, 7:10] + interp_ratio_3 * vel_joystick # (obs[0, 7:10] + vel_joystick) / 2
        future_pos = vel_sum * self.dt * offset * .8
        future_pos_2 = future_pos + vel_sum_2 * self.dt * offset * 1.
        future_pos_3 = future_pos_2 + vel_sum_3 * self.dt * offset * 1.2
        self._feature[19:22] = future_pos[0, :3]
        self._feature[22:25] = future_pos_2[0, :3]
        self._feature[25:28] = future_pos_3[0, :3]

        rot = np.arctan2(right_axis[1], right_axis[0]) * 180 / np.pi
        quat_joystick = torch_utils.quat_from_angle_axis((rot.clone() * np.pi / 180).unsqueeze(0), torch.tensor([0, 0, 1], dtype=torch.float).unsqueeze(0))
        target_rot = torch_utils.quat_mul(heading_rot, quat_joystick)
        target_rot_compare = torch_utils.quat_mul(heading_rot, heading_rot_inv)
        target_rot_1 = torch_utils.slerp(target_rot_compare, target_rot, torch.tensor(interp_ratio_1))
        target_rot_2 = torch_utils.slerp(target_rot_compare, target_rot, torch.tensor(interp_ratio_2))
        target_rot_3 = torch_utils.slerp(target_rot_compare, target_rot, torch.tensor(interp_ratio_3))

        futures_local = calc_heading_xy(target_rot_1)
        futures_local_2 = calc_heading_xy(target_rot_2)
        futures_local_3 = calc_heading_xy(target_rot_3)

        self._feature[28] = futures_local[0]
        self._feature[29] = futures_local[1]
        self._feature[30] = futures_local_2[0]
        self._feature[31] = futures_local_2[1]
        self._feature[32] = futures_local_3[0]
        self._feature[33] = futures_local_3[1]

        return self._feature

    def set_command(self, buttons, axes, hat):
        self._buttons = buttons
        self._axes = axes
        self._hat = hat

        
    ##################################
    # for markers
    ##################################
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = []
            self._face_marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "calm/data/assets/mjcf/"
        asset_file = "heading_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if not self.headless:
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0b11
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        default_pose.p.z = 0.0
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

        face_marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "face_marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, face_marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8))
        self._face_marker_handles.append(face_marker_handle)
        
        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_ACTOR_ID, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self._humanoid_actor_ids + TAR_ACTOR_ID

        self._face_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_FACING_ACTOR_ID, :]
        self._face_marker_pos = self._face_marker_states[..., :3]
        self._face_marker_rot = self._face_marker_states[..., 3:7]
        self._face_marker_actor_ids = self._humanoid_actor_ids + TAR_FACING_ACTOR_ID

        return

    def _update_marker(self):
        humanoid_root_pos = self._humanoid_root_states[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_dir
        self._marker_pos[..., 2] = 0.0

        heading_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        heading_q = quat_from_angle_axis(heading_theta.unsqueeze(0), heading_axis)
        self._marker_rot[:] = heading_q

        self._face_marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_facing_dir
        self._face_marker_pos[..., 2] = -0.01

        face_theta = torch.atan2(self._tar_facing_dir[..., 1], self._tar_facing_dir[..., 0])
        face_axis = torch.zeros_like(self._marker_pos)
        face_axis[..., -1] = 1.0
        face_q = quat_from_angle_axis(face_theta.unsqueeze(0), heading_axis)
        self._face_marker_rot[:] = face_q

        marker_ids = torch.cat([self._marker_actor_ids, self._face_marker_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(marker_ids), len(marker_ids))
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return
    
    def _draw_task(self):
        self._update_marker()

        vel_scale = 0.2
        heading_cols = np.array([[0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        root_pos = self._humanoid_root_states[..., 0:3]
        prev_root_pos = self._prev_root_pos
        sim_vel = (root_pos - prev_root_pos) / self.dt
        sim_vel[..., -1] = 0

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_amp_observations_2(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    root_rot_obs = root_rot
    # root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    # just need return quaternion of root_rot as global
    
    root_pos_obs = root_pos
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_pos_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

def calc_heading_xy(q):
    # calculate heading direction from quaternion
    # the heading is the direction on the xy plane
    # q must be normalized
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    return rot_dir[..., 0], rot_dir[..., 1]
