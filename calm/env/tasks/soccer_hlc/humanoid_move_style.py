import torch

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

# hyper paramaters
VEL_ERR_SCALE = None
TANGENT_ERR_W = None
DIR_REWARD_W = None
FACING_REWARD_W = None

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2


class HumanoidMoveStyle(humanoid_amp_task.HumanoidAMPTask):
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
        # for joystick
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

        # for style
        # walk_f_r, walk_f_l, walk_b_r, walk_b_l
        # run_f_r, run_f_l
        # cmove_r, cmove_l, cmove_f_r, cmove_f_l, cmove_b_r, cmove_b_l
        # jog_b_r, jog_b_l, jog_f_r, jog_f_l
        self._motion_ids = torch.arange(16, device=self.device, dtype=torch.float32)
        self._tar_locomotion_index = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._corresponding_speeds_max = torch.tensor([1, 1, .5, .5,
                                                       5, 5,
                                                       2, 2, 2, 2, 2, 2,
                                                       2, 2, 2, 2], device=self.device, dtype=torch.float32) * 1.2
        self._corresponding_speeds_min = torch.tensor([1, 1, .5, .5,
                                                       5, 5,
                                                       2, 2, 2, 2, 2, 2,
                                                       2, 2, 2, 2], device=self.device, dtype=torch.float32) * .8
        self._angle = np.pi * torch.tensor([0, 0, 180, 180,
                                    0, 0,
                                    -90, 90, 55, -55, -125, 125,
                                    180, 180, 0, 0], device=self.device, dtype=torch.float32) / 180
        self._angle_range = np.pi * torch.tensor(20, device=self.device, dtype=torch.float32) / 180 # 전후 합쳐서 20 도 만큼

        if not self.headless:
            self._build_obj_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 5# + 1
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

        # just for ball test
        # asset_root = "calm/data/assets/mjcf/"
        # asset_file = "soccer_ball.urdf"

        # asset_options = gymapi.AssetOptions()
        # asset_options.angular_damping = .2
        # asset_options.linear_damping = .5
        # asset_options.max_angular_velocity = 100.0
        # asset_options.override_inertia = True
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # shape_property = self.gym.get_asset_rigid_shape_properties(self._target_asset)
        # shape_property[0].restitution = 1
        # shape_property[0].friction = .5
        # shape_property[0].rolling_friction = .5
        # self.gym.set_asset_rigid_shape_properties(self._target_asset, shape_property)


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
        
        # just for ball test
        col_group = env_id
        col_filter = 0b10
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 2.0
        default_pose.p.y = 2.0

        # target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group,
        #                                       col_filter, segmentation_id)
        
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
            if self._axes:
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

    def _reset_task(self, env_ids):
        n = len(env_ids)

        # inference 시 조이스틱 제어
        if self._is_play and self._axes:
            left_x, left_y, left_trigger, right_x, right_y, right_trigger = self._axes
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
        # style 지정을 위한 situtaion 생성
        elif self._enable_rand_heading and len(env_ids) > len(self._motion_ids) * 10:
            rand_theta, rand_face_theta, tar_speed = self.make_situation_state(env_ids)
        # style 지정할 조건이 안되면 (env_ids 가 부족할 때) + 추론 시 게임패드 없으면 랜덤 입력
        else:
            # just for debugging situation creation
            # rand_theta, rand_face_theta, tar_speed = self.make_situation_state_debug(env_ids)

            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min

        
        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)
        # change_steps = self._heading_change_steps_min

        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_facing_dir
            tar_locomotion_index = self._tar_locomotion_index

        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_facing_dir[env_ids]
            tar_locomotion_index = self._tar_locomotion_index[env_ids]

        # added tar_style_motion
        # for un relavent situtaion, need to set zero style rewrad
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
        self.rew_buf[self._tar_locomotion_index==-1] *= 2

        print('reward:', sum(self.rew_buf).item())
        return

    ########################################
    # Set Command
    ########################################

    def set_command(self, buttons, axes, hat):
        self._buttons = buttons
        self._axes = axes
        self._hat = hat

    def make_situation_state(self, env_ids):
        # -1 로 초기화
        n = len(env_ids)
        print(n)
        self._tar_locomotion_index = torch.ones([n], device=self.device, dtype=torch.long) * -1

        # set direction, speed, style
        # 여기서 확룰에 따라 모든 상황 만들기. 아무것도 아닌 상황 포함. maybe no need anymore
        # self._motion_ids_test = self._motion_ids[6:7]

        env_per_situation = int(n * .05) # 
        indices = torch.randint(0, n, (len(self._motion_ids) * env_per_situation,), device=self.device)

        # 랜덤하게 선택한 인덱스에 해당하는 요소 선택
        corresponding_envs = [indices[i * env_per_situation:(i+1) * env_per_situation] for i in range(len(self._motion_ids))]
        rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        rand_face_theta = 2 * np.pi * torch.rand(size=(n, ), device=self.device) - np.pi
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n,device=self.device) + self._tar_speed_min
        rand = torch.rand(env_per_situation, device=self.device)
        for i in range(len(self._motion_ids)):
            self._tar_locomotion_index[corresponding_envs[i]] = i
            rand_face_theta[corresponding_envs[i]] = rand_theta[corresponding_envs[i]] + self._angle[i] + (rand - .5) * self._angle_range
            tar_speed[corresponding_envs[i]] = (self._corresponding_speeds_max[i] - self._corresponding_speeds_min[i]) * rand + \
                                                self._corresponding_speeds_min[i]

        # 스타일 지정 안된 env 들은 랜덤으로 설정 해주기
        # speed cap 도 지정해주기
        # style reward 없도록 지정해주기
        rand_face_theta[self._tar_locomotion_index==-1] = 2 * np.pi * rand_face_theta[self._tar_locomotion_index==-1] - np.pi
        tar_speed[self._tar_locomotion_index==-1] = tar_speed[self._tar_locomotion_index==-1] * .5

        return rand_theta, rand_face_theta, tar_speed
    
    # situation 만드는게 정상적인지 디버깅
    def make_situation_state_debug(self, env_ids):
        i = 10
        n = len(env_ids)
        
        if n <= 0:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min

            return rand_theta, rand_face_theta, tar_speed

        # set direction, speed, style
        # 여기서 확룰에 따라 모든 상황 만들기. 아무것도 아닌상황 포함
        env_per_situation = 2
        indices = torch.ones(size=(n,), device=self.device, dtype=torch.long)

        # 랜덤하게 선택한 인덱스에 해당하는 요소 선택
        corresponding_envs = indices[:]
        rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        rand_face_theta = 2 * np.pi * torch.rand(size=(n, ), device=self.device) - np.pi
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n,device=self.device) + self._tar_speed_min
        rand = torch.rand(env_per_situation, device=self.device)
        
        rand_face_theta = rand_theta + self._angle[i] + (rand - .5) * self._angle_range
        tar_speed = (self._corresponding_speeds_max[i] - self._corresponding_speeds_min[i]) * rand + \
                                            self._corresponding_speeds_min[i]
        
        return rand_theta, rand_face_theta, tar_speed

    
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
