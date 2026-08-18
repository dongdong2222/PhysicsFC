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
from torch import Tensor
from torch.nn.functional import normalize

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np

# hyper paramaters
VEL_ERR_SCALE = None
TANGENT_ERR_W = None
DIR_REWARD_W = None
FACING_REWARD_W = None

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2


class HumanoidMove(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        global VEL_ERR_SCALE
        global TANGENT_ERR_W
        global DIR_REWARD_W
        global FACING_REWARD_W

        VEL_ERR_SCALE = cfg["env"]["VEL_ERR_SCALE"]
        TANGENT_ERR_W = cfg["env"]["TANGENT_ERR_W"]
        DIR_REWARD_W = cfg["env"]["DIR_REWARD_W"]
        FACING_REWARD_W = cfg["env"]["FACING_REWARD_W"]

        self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]
        self._heading_change_steps_min = cfg["env"]["headingChangeStepsMin"]
        self._heading_change_steps_max = cfg["env"]["headingChangeStepsMax"]
        self._enable_rand_heading = cfg["env"]["enableRandHeading"]

        #########################################
        self._is_play = not cfg['args'].train
        self._buttons = None
        self._axes = None
        self._hat = None
        #########################################
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._heading_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_speed = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self._tar_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_dir[..., 0] = 1.0
        
        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

        if not self.headless:
            self._build_obj_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 5
        return obs_size

    def _update_obj_actors(self):
        humanoid_root_pos = self._humanoid_root_states[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_dir
        self._marker_pos[..., 2] = 0.0

        heading_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        heading_q = quat_from_angle_axis(heading_theta, heading_axis)
        self._marker_rot[:] = heading_q

        self._face_marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_facing_dir
        self._face_marker_pos[..., 2] = 0.0

        face_theta = torch.atan2(self._tar_facing_dir[..., 1], self._tar_facing_dir[..., 0])
        face_axis = torch.zeros_like(self._marker_pos)
        face_axis[..., -1] = 1.0
        face_q = quat_from_angle_axis(face_theta, heading_axis)
        self._face_marker_rot[:] = face_q

        marker_ids = torch.cat([self._marker_actor_ids, self._face_marker_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(marker_ids), len(marker_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = []
            self._face_marker_handles = []
            self._load_obj_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_obj_asset(self):
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
            self._build_obj_actors(env_id, env_ptr)

        return

    def _build_obj_actors(self, env_id, env_ptr):
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

    def _build_obj_state_tensors(self):
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

    def _update_task(self):
        #####################################################
        # if play mode, always update per every llc_steps
        #####################################################

        if self._is_play:
            if self._axes or self._is_eval: # update in every step
                rest_env_ids = self.progress_buf >= 0
            else:
                reset_task_mask = self.progress_buf >= self._heading_change_steps
                rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
            self._reset_task(rest_env_ids)
        else:
            reset_task_mask = self.progress_buf >= self._heading_change_steps
            rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
            if len(rest_env_ids) > 0:
                self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)
        
        if self._is_play and self._axes:
            left_x, left_y, left_trigger, right_x, right_y, right_trigger = self._axes
            # left_x, left_y, left_trigger, right_x, right_y, right_trigger, trigger_l, trigger_r = self._axes
            left_x, left_y, left_trigger, right_x, right_y, right_trigger = torch.tensor(left_x, device=self.device), torch.tensor(left_y, device=self.device), torch.tensor(left_trigger, device=self.device), torch.tensor(right_x, device=self.device), torch.tensor(right_y, device=self.device), torch.tensor(right_trigger, device=self.device)
            magnitude_left = torch.clamp(torch.sqrt(torch.pow(left_x, 2) + torch.pow(left_y, 2)), 0, 1)
            magnitude_right = torch.sqrt(torch.pow(right_x, 2) + torch.pow(right_y, 2))
            
            # 왼쪽 조이스틱 값으로 각도 계산
            rand_theta = torch.atan2(-left_y, left_x)

            # 오른쪽 조이스틱으로 heading direction 조작할 때
            if magnitude_right > .1:
                rand_face_theta = torch.atan2(torch.tensor(-right_y, device=self.device), torch.tensor(right_x, device=self.device))
            # 오른쪽 조이스틱은 그대로 둘 때. heading direction = moving direction
            else:
                rand_face_theta = rand_theta
            # 왼쪽 조이스틱 magnitude 에 해당하는 속력으로 목표 속력을 설정
            tar_speed = self._tar_speed_max * magnitude_left
        elif self._enable_rand_heading:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        else:
            rand_theta = torch.zeros(n, device=self.device)
            rand_face_theta = torch.zeros(n, device=self.device)
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min

        # # for restrict face angle for init direc
        # rand_face_theta = (2 * np.pi * torch.rand(n, device=self.device) - np.pi) / 16
        # rand_theta = (2 * np.pi * torch.rand(n, device=self.device) - np.pi) / 1
        # ######
        
        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        # tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                    size=(n,), device=self.device, dtype=torch.int64)
        
        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return
    
    # ################EVALUATION Code Start##############
    def _eval_tgt_update(self):
        testcase_index = torch.div(self.progress_buf[0], self._eval_change_steps, rounding_mode='floor').item()
        target_speeds = torch.from_numpy(self._eval_testcase_dict["target_speeds"]).to(device=self.device, dtype=torch.float)[:,testcase_index]
        # print(f"\n\n{target_speeds.shape}\n\n")
        self._tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.from_numpy(self._eval_testcase_dict["target_speeds"]).to(device=self.device, dtype=torch.float)[:,testcase_index] + self._tar_speed_min
        self._tar_dir = torch.from_numpy(self._eval_testcase_dict["target_move_directions"]).to(device=self.device, dtype=torch.float)[:,testcase_index]
        self._tar_facing_dir = torch.from_numpy(self._eval_testcase_dict["target_facing_directions"]).to(device=self.device, dtype=torch.float)[:,testcase_index]


    def _eval_process(self):
        super()._eval_process()
                
        current_frame = self.progress_buf[0].item() % self._eval_change_steps
        testcase_index = torch.div(self.progress_buf[0], self._eval_change_steps, rounding_mode='floor').item()
        
        root_state = self._humanoid_root_states.clone()
        root_pos = root_state[..., 0:3]
        root_rot = root_state[..., 3:7]
        
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        facing_dir = torch.zeros_like(root_pos)
        facing_dir[..., 0] = 1.0
        facing_dir = quat_rotate(heading_rot, facing_dir)
        
        char_forward_dir_diff = torch.rad2deg(torch_utils.get_angle_between(
            normalize(facing_dir[..., 0:2]),
            normalize(self._tar_facing_dir)
        )).cpu().numpy()
        
        self._eval_current_result.get("char_tgt_forward_diff")[
            :, 
            testcase_index, 
            current_frame
        ] = char_forward_dir_diff  
              
        if current_frame >= self._eval_gain_criteria_frame:
            
            self._eval_check_and_update_gain_time(
                metric="char_tgt_forward",
                testcase_index=testcase_index,
                current_frame=current_frame,
                criteria_type="angle"
            )
        
    # ################EVALUATION Code End##############
        
    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_facing_dir
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_facing_dir[env_ids]
        
        obs = compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir)
        
        # speed test
        # root_pos = self._humanoid_root_states[..., 0:3]
        # delta_root_pos = root_pos - self._prev_root_pos
        # root_vel = delta_root_pos / self.dt
        # print(torch.pow(root_states[0, 7:9], 2).sum().sqrt(), torch.pow(root_vel[0, :2], 2).sum().sqrt(), tar_speed)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:] = compute_heading_reward(root_pos, self._prev_root_pos,  root_rot,
                                                 self._tar_dir, self._tar_speed,
                                                 self._tar_facing_dir, self.dt)
        # print('reward:', sum(self.rew_buf).item())
        return

    ########################################
    # Set Command
    ########################################

    def set_command(self, buttons, axes, hat):
        self._buttons = buttons
        self._axes = axes
        self._hat = hat
    
    def _draw_task(self):
        self._update_obj_actors()

        vel_scale = 0.2
        heading_cols = np.array([[0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        root_pos = self._humanoid_root_states[..., 0:3]
        prev_root_pos = self._prev_root_pos
        sim_vel = (root_pos - prev_root_pos) / self.dt
        sim_vel[..., -1] = 0

        starts = root_pos
        tar_ends = torch.clone(starts)
        tar_ends[..., 0:2] += vel_scale * self._tar_speed.unsqueeze(-1) * self._tar_dir
        sim_ends = starts + vel_scale * sim_vel

        verts = torch.cat([starts, tar_ends, starts, sim_ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i:i+1]
            curr_verts = curr_verts.reshape([2, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, heading_cols)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)
    
    tar_face_dir3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)
    local_tar_face_dir = quat_rotate(heading_rot, tar_face_dir3d)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]

    obs = torch.cat([local_tar_dir, tar_speed, local_tar_face_dir], dim=-1)
    return obs


#@torch.jit.script
def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, tar_face_dir, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    vel_err_scale = VEL_ERR_SCALE
    tangent_err_w = TANGENT_ERR_W

    dir_reward_w = DIR_REWARD_W
    facing_reward_w = FACING_REWARD_W

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = (tar_speed - tar_dir_speed) / tar_speed # for norm
    tangent_vel_err = tangent_speed / tar_speed # for norm
    dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err + 
                        tangent_err_w * tangent_vel_err * tangent_vel_err))

    speed_mask = tar_dir_speed <= 0
    dir_reward[speed_mask] = 0

    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward

