import torch
from torch.nn.functional import normalize as torch_norm

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
from env.tasks.soccer_hlc.ball_physics import BallPhysicsMixin

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from collections import deque, defaultdict
import json
import os
import pickle

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2
TAR_BALL_ID = 3
TAR_ID = 4

IS_TARGET_Z = 0

IS_INIT = True # for initialization from trapping contact state
IS_SIMPLE = True
IS_TARGET_FALL = False
IS_TARGET_PART = True # one-hot body part
IS_CONTACT_OBS = False
IS_GROUND_BALL = False

IS_WALL = False


if IS_INIT:
    with open('blackhole/init_states/move_ball_state.pkl', 'rb') as f:
        init_states = pickle.load(f)
        init_ball_states = init_states['ball_state']
        init_dof_pos = init_states['dof_pos']
        init_dof_vel = init_states['dof_vel']
        init_root_states = init_states['root_state']
        init_rb_pos = init_states['rb_pos']
        init_rb_rot = init_states['rb_rot']
        init_rb_vel = init_states['rb_vel']
        init_rb_ang_vel = init_states['rb_ang_vel']

#####

def add_reward(*args):
    global r1_buf, r2_buf, r3_buf
    r1, r2, r3 = args
    r1_buf.append(float(r1))
    r2_buf.append(float(r2))
    r3_buf.append(float(r3))

    if len(r1_buf) > 10:
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            data = {'r1':list(), 'r2':list(), 'r3':list()}  # 초기 빈 딕셔너리
        else:
            # 파일이 비어있지 않은 경우, 데이터 로드
            with open(file_path, 'r') as f:
                data = json.load(f)
        with open(file_path, 'w') as f:
            data['r1'].extend(r1_buf)
            data['r2'].extend(r2_buf)
            data['r3'].extend(r3_buf)
            json.dump(data, f)

        r1_buf, r2_buf, r3_buf = list(), list(), list()

######
        
class HumanoidTrapping_3(BallPhysicsMixin, humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # cfg["env"]["ballProfile"] 가 있으면 공 / 바닥 물성을 프로필로, 없으면 아래 하드코딩 값 그대로
        self._ball_profile = None
        if "ballProfile" in cfg["env"]:
            self._setup_ball_profile(cfg)
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
        
        #### 공 초기화 셋팅
        self._init_ball_dist_min = cfg['env']['initialBallDistMin']
        self._init_ball_dist_max = cfg['env']['initialBallDistMax']
        self._init_ball_speed_min = cfg['env']['initialBallSpeedMin']
        self._init_ball_speed_max = cfg['env']['initialBallSpeedMax']
        self._init_ball_h_min = cfg['env']['initialBallHeightMin']
        self._init_ball_h_max = cfg['env']['initialBallHeightMax']
        self._init_ball_speed_z_min = cfg['env']['initialBallSpeedZMin']
        self._init_ball_speed_z_max = cfg['env']['initialBallSpeedZMax']
        self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]

        ###
        self._ball_contact_flag = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        
        self._tar_facing_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._tar_facing_dir[..., 0] = 1.0

        self._ball_target = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)

        # if not self.headless:
        self._build_obj_state_tensors()

        # ball prev states
        self._prev_ball_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        num_actors = self.get_num_actors_per_env()
        self._ball_root_states = self._root_states.view(self.num_envs, num_actors, 13)[..., TAR_BALL_ID, :]
        self._touched = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._fall_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._target_fall_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._ori_dist = torch.zeros([self.num_envs], device=self.device)
        self._nearest = torch.zeros([self.num_envs], device=self.device)
        self._counter = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._prev_ball_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._target_z = torch.zeros([self.num_envs, 1], device=self.device, dtype=torch.float)
        self._launch_direc = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)
        self._is_ground = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self._initial_dist = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._initial_ball_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._moved_dist = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._onehot_body_part = torch.zeros([self.num_envs, 6], device=self.device, dtype=torch.long)
        self._onehot_contact_state = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.long)

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            if IS_SIMPLE:
                obs_size = 3 + 3 + 3 # move + 공 위치 + 공 속력 + (ang vel)
                if IS_TARGET_FALL:
                    obs_size += 3
                if IS_TARGET_PART:
                    obs_size += 6
                if IS_CONTACT_OBS:
                    obs_size += 2
            else:
                obs_size = 5 + 3 + 3 + 3 + 2 + IS_TARGET_Z # move + 공 위치 + 공 속력 + (ang vel)

        return obs_size

    def _update_obj_actors(self):
        # 이 함수는 physics_step 즉 중간 시점에 일어남
        # reward 계산은 post_physics_step 에 일어나므로, 여기에 target 을 업데이트해도 문제없음

        humanoid_root_pos = self._humanoid_root_states[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_dir
        self._marker_pos[..., 2] = -10.0

        heading_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        heading_q = quat_from_angle_axis(heading_theta, heading_axis)
        self._marker_rot[:] = heading_q

        self._face_marker_pos[..., 2] = -1

        # face_theta = torch.atan2(self._tar_facing_dir[..., 1], self._tar_facing_dir[..., 0])
        # face_axis = torch.zeros_like(self._marker_pos)
        # face_axis[..., -1] = 1.0
        # face_q = quat_from_angle_axis(face_theta, heading_axis)
        # self._face_marker_rot[:] = face_q

        # update target by user control
        # delta = torch.stack([self._tar_dir[..., 0] * self._tar_speed, self._tar_dir[..., 1] * self._tar_speed], dim=-1) * self.dt * 5
        self._target_pos[..., :2] = self._target_fall_pos[..., :2]
        self._target_pos[..., 2] = 0
        # self._ball_target = self._target_pos[..., :2]
        
        # marker, ball, target 전부 업데이트. face marker 는 필요없음!
        marker_ids = torch.cat([self._marker_actor_ids, self._face_marker_actor_ids, self._target_actor_ids, self._ball_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(marker_ids), len(marker_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        # if not self.headless:
        self._marker_handles = []
        self._face_marker_handles = []
        self._ball_handles = []
        self._target_handles = []
        self._wall_handles = []
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

        ## ball
        if self._ball_profile is not None:
            self._ball_asset = self._create_ball_asset()
        else:
            asset_root = "calm/data/assets/mjcf/"
            asset_file = "soccer_ball.urdf"

            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.05 #.5
            asset_options.linear_damping = 0.1 #.2
            asset_options.max_angular_velocity = 100.0
            asset_options.density = 80.71
            asset_options.override_inertia = True
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            shape_property = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
            shape_property[0].restitution = 0.8 #0.8
            shape_property[0].friction = .2
            shape_property[0].rolling_friction = .2
            self.gym.set_asset_rigid_shape_properties(self._ball_asset, shape_property)

        ## sphere
        asset_root = "calm/data/assets/mjcf/"
        asset_file = "soccer_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._sphere_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # create box asset
        if IS_WALL:
            asset_options = gymapi.AssetOptions()

            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.fix_base_link = True
            self._box_asset = self.gym.create_box(self.sim, 10000, 1, 1000, asset_options)
            # shape_property = self.gym.get_asset_rigid_shape_properties(self._box_asset)
            # print(shape_property[0].friction)
            # print(shape_property[0].rolling_friction)
            # print(shape_property[0].restitution)
            # print(asset_options.linear_damping)
            # print(asset_options.angular_damping)
            # import sys
            # sys.exit(0)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        # if not self.headless:
        self._build_obj_actors(env_id, env_ptr)

        # for body part color highlight
        # handle = self.humanoid_handles[env_id]
        # self.gym.set_rigid_body_color(env_ptr, handle, 14, gymapi.MESH_VISUAL,
        #                                 gymapi.Vec3(1, 0, 0))

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

        # ball
        col_group = env_id
        col_filter = 0b10
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        ball_handle = self.gym.create_actor(env_ptr, self._ball_asset, default_pose, "ball", col_group,
                                              col_filter, segmentation_id)
        
        body_property = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
        self.gym.set_actor_rigid_body_properties(env_ptr, ball_handle, body_property)

        self._ball_handles.append(ball_handle)

        # desired ball pos
        col_group = env_id
        col_filter = 0b11
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 0.0

        target_handle = self.gym.create_actor(env_ptr, self._sphere_asset, default_pose, "target", col_group,
                                              col_filter, segmentation_id)
        
        body_property = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle)
        body_property[0].mass = .0
        self.gym.set_actor_rigid_body_properties(env_ptr, target_handle, body_property)

        self._target_handles.append(target_handle)

        # wall
        if IS_WALL:
            default_pose = gymapi.Transform()
            default_pose.p.y = 1000

            wall_handle = self.gym.create_actor(env_ptr, self._box_asset, default_pose, "wall", col_group,
                                                col_filter, segmentation_id)
            self._wall_handles.append(wall_handle)

        return

    def _build_obj_state_tensors(self):
        num_actors = self.get_num_actors_per_env()

        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_ACTOR_ID, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self._humanoid_actor_ids + TAR_ACTOR_ID

        self._face_marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_FACING_ACTOR_ID, :]
        self._face_marker_pos = self._face_marker_states[..., :3]
        self._face_marker_rot = self._face_marker_states[..., 3:7]
        self._face_marker_actor_ids = self._humanoid_actor_ids + TAR_FACING_ACTOR_ID

        self._ball_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_BALL_ID, :]
        self._ball_actor_ids = self._humanoid_actor_ids + TAR_BALL_ID
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)

        # self._contact_forces - 이미 있음
        self._ball_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., bodies_per_env-4+2, :]
        #

        self._ball_pos = self._ball_states[..., :3]
        self._ball_vel = self._ball_states[..., 7:10]
        self._ball_ang_vel = self._ball_states[..., 10:]

        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_ID, :]
        self._target_actor_ids = self._humanoid_actor_ids + TAR_ID
        
        self._target_pos = self._target_states[..., :3]
        self._target_vel = self._target_states[..., 7:10]

        if self._ball_profile is not None:
            self._init_ball_friction(self._ball_handles, self._ball_actor_ids)
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
        return

    #region reset target
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
            # tar_speed = self._tar_speed_max * magnitude_left
        
        # during trainig
        elif self._enable_rand_heading:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            # tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        else:
            rand_theta = torch.zeros(n, device=self.device)
            rand_face_theta = torch.zeros(n, device=self.device)
            # tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)
        
        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

        # self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

        self._ball_contact_flag = torch.zeros([n], device=self.device, dtype=torch.bool)
        self._touched[env_ids] = torch.zeros([n], device=self.device, dtype=torch.float)
        self._nearest[env_ids] = torch.ones([n], device=self.device) * 100
        self._ori_dist[env_ids] = torch.pow(self._humanoid_root_states[env_ids, :2] - self._fall_pos[env_ids, :2], 2).sum(dim=-1).sqrt()
        self._counter[env_ids] = torch.zeros([n], device=self.device, dtype=torch.float)
        self._prev_ball_vel[env_ids] = self._ball_vel[env_ids]

        # target fall position - heading direction 기준 +- 90
        # self._target_fall_pos[env_ids] *= 0
        # self._target_fall_pos[env_ids, :3] += self._humanoid_root_states[env_ids, :3]
        # heading_rot = torch_utils.calc_heading_quat(self._humanoid_root_states[env_ids, 3:7])
        # facing_dir = torch.zeros_like(self._humanoid_root_states[env_ids, :3])
        # facing_dir[..., 0] = 1.0
        # facing_dir = quat_rotate(heading_rot, facing_dir) # 3d
        # root_facing_angle = torch.atan2(facing_dir[:, 1], facing_dir[:, 0])
        # rand_theta = (2 * np.pi * torch.rand(n, device=self.device) - np.pi) / 1
        # rand_fall_dist = (1 - 0) * torch.rand(n, device=self.device) + 0
        # target_fall_angle = root_facing_angle + rand_theta
        # self._target_fall_pos[env_ids] *= 0
        # self._target_fall_pos[env_ids, :2] = torch.stack([torch.cos(target_fall_angle), torch.sin(target_fall_angle)], dim=-1)
        # self._target_fall_pos[env_ids, :2] *= rand_fall_dist.unsqueeze(1)
        # self._target_fall_pos[env_ids, :2] += self._ball_pos[env_ids, :2]
        
        # num_classes = self._onehot_body_part.shape[-1]
        # if IS_GROUND_BALL:
        #     # only foots - for ground ball
        #     left = n//2
        #     random_indices = torch.zeros((n,), dtype=torch.long, device=self.device)
        #     random_indices[:left] = 3
        #     random_indices[left:] = 5
        #     self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(random_indices, num_classes=num_classes)
        # else:
        #     self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(torch.randint(0, num_classes, (n,), device=self.device), num_classes=num_classes)
        
        self._onehot_contact_state[env_ids] = torch.tensor([0, 1], device=self.device)

        # test - specific fixed body part
        # self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(torch.randint(0, 1, (n,), device=self.device), num_classes=num_classes)

        #################################
        # 반영이 안되서 비활성화. 대신 update_marker_target() 에다가 적용
        # delta = torch.stack([self._tar_dir[..., 0] * self._tar_speed, self._tar_dir[..., 1] * self._tar_speed], dim=-1)
        # self._target_states[..., :2] = self._ball_states[..., :2]# + delta
        return
    
    #endregion

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_facing_dir
            ball_pos = self._ball_pos
            ball_vel = self._ball_vel
            ball_ang_vel = self._ball_ang_vel
            fall_pos = self._fall_pos
            target_z = self._target_z
            target_fall_pos = self._target_fall_pos
            onehot_body_part = self._onehot_body_part
            onehot_contact_state = self._onehot_contact_state

        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_facing_dir[env_ids]
            ball_pos = self._ball_pos[env_ids]
            ball_vel = self._ball_vel[env_ids]
            ball_ang_vel = self._ball_ang_vel[env_ids]
            fall_pos = self._fall_pos[env_ids]
            target_z = self._target_z[env_ids]
            target_fall_pos = self._target_fall_pos[env_ids]
            onehot_body_part = self._onehot_body_part[env_ids]
            onehot_contact_state = self._onehot_contact_state[env_ids]

        if IS_SIMPLE:
            if IS_TARGET_FALL:
                if IS_TARGET_PART:
                    if IS_CONTACT_OBS:
                        obs = compute_heading_observations_simple_target_fall_onehot_contact(root_states, ball_pos, ball_vel, ball_ang_vel, target_fall_pos, onehot_body_part, onehot_contact_state)
                    else:
                        obs = compute_heading_observations_simple_target_fall_onehot(root_states, ball_pos, ball_vel, ball_ang_vel, target_fall_pos, onehot_body_part)
                else:
                    obs = compute_heading_observations_simple_target_fall(root_states, ball_pos, ball_vel, ball_ang_vel, target_fall_pos)
            else:
                if IS_TARGET_PART:
                    obs = compute_heading_observations_simple_onehot(root_states, ball_pos, ball_vel, ball_ang_vel, onehot_body_part)
                else:
                    obs = compute_heading_observations_simple(root_states, tar_dir, tar_speed, tar_face_dir, ball_pos, ball_vel, ball_ang_vel, fall_pos, target_z)
        else:
            obs = compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir, ball_pos, ball_vel, ball_ang_vel, fall_pos, target_z)
        
        # speed test
        # root_pos = self._humanoid_root_states[..., 0:3]
        # delta_root_pos = root_pos - self._prev_root_pos
        # root_vel = delta_root_pos / self.dt
        # print(torch.pow(root_states[0, 7:9], 2).sum().sqrt(), torch.pow(root_vel[0, :2], 2).sum().sqrt(), tar_speed)
        
        return obs

    # region compute reward
    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        ball_pos = self._ball_states[..., :3]

        # compute contact conditions
        contact_forces = self._contact_forces
        ball_contact_forces = self._ball_contact_forces
        # print(contact_forces[0, self._contact_body_ids, :3], ball_contact_forces[0])
        contact_force_magnitude = torch.pow(contact_forces, 2).sum(dim=-1).sqrt()
        contact_force_magnitude = contact_force_magnitude[..., self._contact_body_ids[2:]]
        ball_contact_force_magnitude = torch.pow(ball_contact_forces, 2).sum(dim=-1).sqrt()

        flag = torch.logical_and(torch.any(contact_force_magnitude, dim=-1), ball_contact_force_magnitude)
        self._ball_contact_flag = torch.logical_and(flag, ball_pos[..., 2] > .2)
        self._moved_dist = torch.norm(ball_pos - self._initial_ball_pos, dim=-1)

        # # self._touched[torch.logical_and(self._touched==0, flag)] = 1
        # self._touched[flag] = 1
        # self._counter[self._touched==1] += 1
        # self._touched[self._counter > 5] += 1

        # # flag = ball_contact_force_magnitude > 1
        # # self._ball_contact_flag = torch.logical_and(flag, ball_pos[..., 2] > .2)

        # delta_v = torch.norm(self._ball_vel[..., :2] - self._prev_ball_vel[..., :2], 2, dim=1)
        # # self._touched[delta_v > .5] = 1
        # # self._counter[self._touched==1] += 1
        # # self._touched[self._counter > 5] += 1

        ####################################################################

        # 발과 거리가 가깝고 & 공 속력변화가 크고 & 공의 높이가 어느정도 높이 이상
        # ball_foot_dist_r = torch.norm(self._rigid_body_pos[..., 11, :] - self._ball_pos[..., :3], dim=1)
        # ball_foot_dist_l = torch.norm(self._rigid_body_pos[..., 14, :] - self._ball_pos[..., :3], dim=1)
        # ball_leg_dist_r = torch.norm(self._rigid_body_pos[..., 10, :] - self._ball_pos[..., :3], dim=1)
        # ball_leg_dist_l = torch.norm(self._rigid_body_pos[..., 13, :] - self._ball_pos[..., :3], dim=1)
        # ball_head_dist = torch.norm(self._rigid_body_pos[..., 2, :] - self._ball_pos[..., :3], dim=1)
        # ball_chest_dist = torch.norm(self._rigid_body_pos[..., 1, :] - self._ball_pos[..., :3], dim=1)
        # delta_v = torch.norm(self._ball_vel[..., :2] - self._prev_ball_vel[..., :2], 2, dim=1)
        ball_height = self._ball_pos[..., 2]

        # ball_no_below = torch.logical_and(self._rigid_body_pos[..., 11, 2] < self._ball_pos[..., 2], self._rigid_body_pos[..., 14, 2] < self._ball_pos[..., 2])
        # is_contact = torch.logical_or(ball_foot_dist_r < 0.3, ball_foot_dist_l < 0.3)
        # is_contact = torch.logical_and(is_contact, delta_v > 1) #is_contact
        
        # is_contact = torch.logical_and(is_contact, ball_no_below)
        # is_contact = torch.logical_or(ball_foot_dist_l < 0.3, ball_foot_dist_r < 0.3)
        # is_contact = torch.logical_and(is_contact , delta_v > 0.5)
        # is_contact = torch.logical_and(is_contact, ball_height > 0.2)
        
        is_contact = torch.logical_and(ball_contact_force_magnitude > .1, ball_height > 0.2)

        self._touched[torch.logical_and(self._touched==0, is_contact)] = 1
        self._counter[self._touched>0] += 1
        self._touched[self._counter >= 10] = 2
        
        # before contact, set target fall pos - ball pos
        self._target_fall_pos[self._touched==0] = ball_pos[self._touched==0]
        self._target_fall_pos[self._touched==0, 2] = 0

        # after contact, set target fall pos
        # target fall position - heading direction 기준 +- 90
        env_set_target = self._counter == 1
        heading_rot = torch_utils.calc_heading_quat(self._humanoid_root_states[env_set_target, 3:7])
        facing_dir = torch.zeros_like(self._humanoid_root_states[env_set_target, :3])
        facing_dir[..., 0] = 1.0
        facing_dir = quat_rotate(heading_rot, facing_dir) # 3d
        root_facing_angle = torch.atan2(facing_dir[:, 1], facing_dir[:, 0])
        rand_theta = (2 * np.pi * torch.rand(sum(env_set_target), device=self.device) - np.pi) / 2
        rand_fall_dist = (1 - 0) * torch.rand(sum(env_set_target), device=self.device) + 0
        target_fall_angle = root_facing_angle + rand_theta
        # target_fall_angle = rand_theta * 0 - np.pi / 2
        # rand_fall_dist = rand_fall_dist * 0 + 1
        self._target_fall_pos[env_set_target] *= 0
        self._target_fall_pos[env_set_target, :2] = torch.stack([torch.cos(target_fall_angle), torch.sin(target_fall_angle)], dim=-1)
        self._target_fall_pos[env_set_target, :2] *= (1 * rand_fall_dist.unsqueeze(1))
        self._target_fall_pos[env_set_target, :2] += self._ball_pos[env_set_target, :2]
        
        self._onehot_contact_state[env_set_target] = torch.tensor([1, 0], device=self.device)
        
        # print('> target fall pos:', self._target_fall_pos[0])

        # for flag
        cur_dist = torch.pow(root_pos[..., :2] - self._fall_pos, 2).sum(dim=-1).sqrt()
        self._nearest = torch.min(cur_dist, self._nearest)

        # if ball_contact_force_magnitude[0] > 1:
        #     print(ball_contact_force_magnitude[0])

        self.rew_buf[:] = compute_heading_reward(root_pos, self._prev_root_pos, root_rot,
                                                 self._tar_dir, self._tar_speed,
                                                 self._tar_facing_dir, ball_pos, self._prev_ball_pos, 
                                                 self._ball_target, self.dt, self._ball_contact_flag,
                                                 self.max_episode_length, self.progress_buf, self._touched,
                                                 self._fall_pos, self._ori_dist, self._nearest,
                                                 self._rigid_body_pos,
                                                 self._launch_direc,
                                                 self._target_fall_pos,
                                                 self._onehot_body_part
                                                 )
        print('reward:', sum(self.rew_buf).item())
        # self._touched[self._touched==1] = 2

        self._prev_ball_vel[...] = self._ball_vel[...]

        # print('', torch.sum(torch.pow(self._humanoid_root_states[0, 7:9], 2)).sqrt())
        return
    
    # endregion
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces,
                                                                           self._contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights,
                                                                           self._humanoid_root_states[..., :3],
                                                                           self._ball_states[..., :3],
                                                                           self._ball_contact_forces,
                                                                           self._touched,
                                                                           self._is_ground,
                                                                           self._initial_dist,
                                                                           self._moved_dist,
                                                                           )
        return
    
    # region reset ball
    def _reset_actors(self, env_ids):
        if IS_INIT:
            self.sample = np.random.choice(init_dof_pos.shape[0], size=len(env_ids))

        super()._reset_actors(env_ids)
        
        air_ratio = 0.5  # env 마다 이 확률로 공중볼, 나머지 땅볼
        is_air = torch.rand(len(env_ids), device=self.device) < air_ratio

        # if IS_GROUND_BALL:
        #     self._reset_ball_new_ground(env_ids)
        # else:
        #     self._reset_ball_new_2(env_ids)

        self._reset_ball_new_2(env_ids[is_air])
        self._reset_ball_new_ground(env_ids[~is_air])
        return
    
    def _reset_default(self, env_ids):
        if IS_INIT:
            self._humanoid_root_states[env_ids] = torch.tensor(init_root_states[self.sample], device=self.device)
            self._humanoid_root_states[env_ids, :2] = 0
            self._dof_pos[env_ids] = torch.tensor(init_dof_pos[self.sample], device=self.device)
            self._dof_vel[env_ids] = torch.tensor(init_dof_vel[self.sample], device=self.device)
            self._rigid_body_pos[env_ids] = torch.tensor(init_rb_pos[self.sample], device=self.device)
            self._rigid_body_rot[env_ids] = torch.tensor(init_rb_rot[self.sample], device=self.device)
            self._rigid_body_vel[env_ids] = torch.tensor(init_rb_vel[self.sample], device=self.device)
            self._rigid_body_ang_vel[env_ids] = torch.tensor(init_rb_ang_vel[self.sample], device=self.device)
            self._reset_default_env_ids = env_ids
        else:
            super()._reset_default(env_ids)
        return

    
    def _reset_ball(self, env_ids):
        # 공이 캐릭터와 일정 거리 이상 멀어지면 어느정도 랜덤으로 캐릭터 근처로 다시 셋팅
        # 속도도 적당히 셋팅_target_states
        # marker 와 다르게 실제로 물리엔진 내에서의 공의 상태를 업데이트 해줘야함
        
        n = len(env_ids)

        # init pos
        rand_dist = (self._init_ball_dist_max - self._init_ball_dist_min) * torch.rand(n, device=self.device) + self._init_ball_dist_min
        rand_theta =  (2 * np.pi * torch.rand(n, device=self.device) - np.pi) / 4 # / 4
        self._ball_states[env_ids, 0] = rand_dist * torch.cos(rand_theta)
        self._ball_states[env_ids, 1] = rand_dist * torch.sin(rand_theta)
        
        # init height
        rand_h = (self._init_ball_h_max - self._init_ball_h_min) * torch.rand(n, device=self.device) + self._init_ball_h_min
        self._ball_states[env_ids, 2] = rand_h

        # init vel - root 쪽으로 날아와야함
        init_dir = torch_norm(self._humanoid_root_states[env_ids, :2] - self._ball_states[env_ids, :2], dim=-1)
        rand_theta = torch.atan2(init_dir[..., 1], init_dir[..., 0])
        rand_theta += (np.pi *.25) * torch.rand(n, device=self.device) - (np.pi * .125)
        rand_speed = (self._init_ball_speed_max - self._init_ball_speed_min) * torch.rand(n, device=self.device) + self._init_ball_speed_min
        rand_speed_z = (self._init_ball_speed_z_max - self._init_ball_speed_z_min) * torch.rand(n, device=self.device) + self._init_ball_speed_z_min
        # self._ball_states[env_ids, 7] = rand_speed * torch.cos(rand_theta)
        # self._ball_states[env_ids, 8] = rand_speed * torch.sin(rand_theta)
        # self._ball_states[env_ids, 9] = rand_speed_z

        # t = torch.rand(n, device=self.device) * .1 + .5
        t = torch.clip(rand_dist * .1 + torch.rand(n, device=self.device) * .4 - .2, .5)
        self._fall_pos[env_ids] = torch.zeros([n, 2], device=self.device) + torch.rand([n, 2], device=self.device) * 2. -1. #* 4 - 2
        self._ball_states[env_ids, 7:9] = (self._fall_pos[env_ids] -self._ball_states[env_ids, :2]) / t.unsqueeze(1)
        self._ball_states[env_ids, 9] = 0.5 * 9.81 * t - self._ball_states[env_ids, 2] / t

        self._target_pos[env_ids, :2] = self._fall_pos[env_ids]
        self._target_z[env_ids] = torch.rand([n, 1], device=self.device) * 0. + .0 #
        self._is_ground[env_ids] = False

        # 각속도는 일단 놔둠
        # self._ball_states[env_ids, 10:13] = 0.0
        # self._target_states[env_ids, :3] = self._ball_states[env_ids, :3]

    def _reset_ball_new_2(self, env_ids):
        num_envs = len(env_ids)
        
        ratios = [1., .0]
        sizes = [int(r * len(env_ids)) for r in ratios] # normal, far
        sizes[-1] += len(env_ids) - sum(sizes)

        min_init_theta = -180 #-180
        max_init_theta = 180 #180
        min_speed = 10
        max_speed = 15
        min_fall_dist = 0.0
        max_fall_dist = 2
        min_fall_theta = -180
        max_fall_theta = 180
        min_phi = 15 #15
        max_phi = 60 #60
        g = 9.81

        # far ball
        min_fall_dist_far = 2.5
        max_fall_dist_far = 3.5
        min_fall_theta_far = 135
        max_fall_theta_far = 225
        min_phi_far = 45
        max_phi_far = 60

        rot = self._humanoid_root_states[env_ids, 3:7]
        # root_dir = torch_utils.calc_heading(rot)
        root_speed = torch.norm(self._humanoid_root_states[env_ids, 7:9], dim=1)
        root_dir = torch.atan2(self._humanoid_root_states[env_ids, 8], self._humanoid_root_states[env_ids, 7])
        root_dir = torch.tensor(root_dir, device=self.device)

        speed = torch.rand(num_envs, device=self.device) * (max_speed - min_speed) + min_speed
        # speed = 10 #((torch.rand(num_envs, device=self.device)-.5) * .4 + 1) * dist * 1
        ratio = (speed - min_speed) / (max_speed - min_speed)
        max_fall_dist = ratio * max_fall_dist + 2

        # fall_theta = torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_fall_theta - min_fall_theta) + min_fall_theta)
        fall_theta = root_dir + torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_fall_theta - min_fall_theta) + min_fall_theta)
        fall_theta[sizes[0]:] = root_dir[sizes[0]:] + torch.deg2rad(torch.rand(sizes[1], device=self.device) * (max_fall_theta_far - min_fall_theta_far) + min_fall_theta_far)
        fall_dist = torch.rand(num_envs, device=self.device) * (max_fall_dist - min_fall_dist) + min_fall_dist
        fall_dist[sizes[0]:] = torch.rand(sizes[1], device=self.device) * (max_fall_dist_far - min_fall_dist_far) + min_fall_dist_far
        
        fall_x = torch.cos(fall_theta) * fall_dist + self._humanoid_root_states[env_ids, 7] * .5
        fall_y = torch.sin(fall_theta) * fall_dist + self._humanoid_root_states[env_ids, 8] * .5

        # fast_envs = root_speed > .5
        # fall_x[fast_envs] = torch.cos(fall_theta[fast_envs]) * root_speed[fast_envs] * 1
        # fall_y[fast_envs] = torch.sin(fall_theta[fast_envs]) * root_speed[fast_envs] * 1

        theta = torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_init_theta - min_init_theta) + min_init_theta)
        phi = torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_phi - min_phi) + min_phi)
        phi[sizes[0]:] = torch.deg2rad(torch.rand(sizes[1], device=self.device) * (max_phi_far - min_phi_far) + min_phi_far)
        t = 2*speed*torch.sin(phi) / g
        dist = speed*torch.cos(phi) * t

        x = fall_x + dist * torch.cos(theta)
        y = fall_y + dist * torch.sin(theta)
        z = 0.115

        dx = fall_x - x
        dy = fall_y - y
        launch_direc = torch.atan2(dy, dx)
        self._launch_direc[env_ids, 0] = dx
        self._launch_direc[env_ids, 1] = dy
        self._launch_direc[env_ids] = torch.nn.functional.normalize(self._launch_direc[env_ids], dim=1)

        # speed calibration
        speed = (1 + t*.025) * speed
        vx = torch.cos(phi) * torch.cos(launch_direc) * speed
        vy = torch.cos(phi) * torch.sin(launch_direc) * speed
        vz = torch.sin(phi) * speed

        self._ball_states[env_ids, 0] = torch.tensor(x, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 1] = torch.tensor(y, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 2] = torch.tensor(z, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 7] = torch.tensor(vx, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 8] = torch.tensor(vy, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 9] = torch.tensor(vz, device='cuda', dtype=torch.float32)
        self._fall_pos[env_ids, 0] = torch.tensor(fall_x, device='cuda', dtype=torch.float32)
        self._fall_pos[env_ids, 1] = torch.tensor(fall_y, device='cuda', dtype=torch.float32)
        self._target_z[env_ids] = 0
        self._target_pos[env_ids, :2] = self._target_fall_pos[env_ids, :2] # self._fall_pos[env_ids]
        self._is_ground[env_ids] = False

        num_classes = self._onehot_body_part.shape[-1]
        self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(torch.randint(0, num_classes, (num_envs,), device=self.device), num_classes=num_classes)
        # self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(torch.randint(3, 4, (num_envs,), device=self.device), num_classes=num_classes)

        # self._ball_states[env_ids, 10] = torch.rand(num_envs, device=self.device) * 30
        # self._fall_pos[env_ids, 11] = 0#torch.tensor(fall_y, device='cuda', dtype=torch.float32)
        # self._fall_pos[env_ids, 12] = 0#torch.tensor(fall_y, device='cuda', dtype=torch.float32)

    def _reset_ball_ground(self, env_ids):
        # 공이 캐릭터와 일정 거리 이상 멀어지면 어느정도 랜덤으로 캐릭터 근처로 다시 셋팅
        # 속도도 적당히 셋팅_target_states
        # marker 와 다르게 실제로 물리엔진 내에서의 공의 상태를 업데이트 해줘야함
        
        n = len(env_ids)

        # init pos
        rand_dist = (self._init_ball_dist_max - self._init_ball_dist_min) * torch.rand(n, device=self.device) + self._init_ball_dist_min
        rand_theta =  (2 * np.pi * torch.rand(n, device=self.device) - np.pi) / 4 # / 4
        self._ball_states[env_ids, 0] = rand_dist * torch.cos(rand_theta)
        self._ball_states[env_ids, 1] = rand_dist * torch.sin(rand_theta)
        
        # init height
        self._ball_states[env_ids, 2] = 0.2

        # init vel - root 쪽으로 날아와야함
        init_dir = torch_norm(self._humanoid_root_states[env_ids, :2] - self._ball_states[env_ids, :2], dim=-1)
        rand_theta = torch.atan2(init_dir[..., 1], init_dir[..., 0])
        rand_theta += (np.pi *.25) * torch.rand(n, device=self.device) - (np.pi * .125)
        rand_speed = (self._init_ball_speed_max - self._init_ball_speed_min) * torch.rand(n, device=self.device) + self._init_ball_speed_min
        rand_speed_z = (self._init_ball_speed_z_max - self._init_ball_speed_z_min) * torch.rand(n, device=self.device) + self._init_ball_speed_z_min
        # self._ball_states[env_ids, 7] = rand_speed * torch.cos(rand_theta)
        # self._ball_states[env_ids, 8] = rand_speed * torch.sin(rand_theta)
        # self._ball_states[env_ids, 9] = rand_speed_z

        # t = torch.rand(n, device=self.device) * .1 + .5
        t = torch.clip(rand_dist * .05 + torch.rand(n, device=self.device) * .4 - .2, .5)
        self._fall_pos[env_ids] = torch.zeros([n, 2], device=self.device) + torch.rand([n, 2], device=self.device) * 2. -1. #* 4 - 2
        self._ball_states[env_ids, 7:9] = (self._fall_pos[env_ids] -self._ball_states[env_ids, :2]) / t.unsqueeze(1)
        self._ball_states[env_ids, 9] = 0.1

        self._target_pos[env_ids, :2] = self._fall_pos[env_ids]
        self._target_z[env_ids] = torch.rand([n, 1], device=self.device) * 0. + .0 #
        self._is_ground[env_ids] = True

    def _reset_ball_new_ground(self, env_ids):
        num_envs = len(env_ids)

        min_init_theta = -180 #-180
        max_init_theta = 180 #180
        min_speed = 5
        max_speed = 13
        min_fall_dist = 0.0
        max_fall_dist = 2
        min_fall_theta = -180
        max_fall_theta = 180
        min_phi = 0
        max_phi = 1
        g = 9.81

        rot = self._humanoid_root_states[env_ids, 3:7]
        # root_dir = torch_utils.calc_heading(rot)
        root_speed = torch.norm(self._humanoid_root_states[env_ids, 7:9], dim=1)
        root_dir = torch.atan2(self._humanoid_root_states[env_ids, 8], self._humanoid_root_states[env_ids, 7])
        root_dir = torch.tensor(root_dir, device=self.device)

        speed = torch.rand(num_envs, device=self.device) * (max_speed - min_speed) + min_speed
        # speed = 10 #((torch.rand(num_envs, device=self.device)-.5) * .4 + 1) * dist * 1
        ratio = (speed - min_speed) / (max_speed - min_speed)
        max_fall_dist = ratio * max_fall_dist + 2

        # fall_theta = torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_fall_theta - min_fall_theta) + min_fall_theta)
        fall_theta = root_dir + torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_fall_theta - min_fall_theta) + min_fall_theta)
        fall_dist = torch.rand(num_envs, device=self.device) * (max_fall_dist - min_fall_dist) + min_fall_dist
        fall_x = torch.cos(fall_theta) * fall_dist + self._humanoid_root_states[env_ids, 7] * .5
        fall_y = torch.sin(fall_theta) * fall_dist + self._humanoid_root_states[env_ids, 8] * .5

        # fast_envs = root_speed > .5
        # fall_x[fast_envs] = torch.cos(fall_theta[fast_envs]) * root_speed[fast_envs] * 1
        # fall_y[fast_envs] = torch.sin(fall_theta[fast_envs]) * root_speed[fast_envs] * 1

        theta = torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_init_theta - min_init_theta) + min_init_theta)
        phi = torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_phi - min_phi) + min_phi)
        t = 2*speed*torch.sin(phi) / g
        # dist = speed*torch.cos(phi) * t
        dist = torch.clip(speed * 1.0 + torch.rand(num_envs, device=self.device) * 1.0 - .5, 2)

        x = fall_x + dist * torch.cos(theta)
        y = fall_y + dist * torch.sin(theta)
        z = 0.115

        dx = fall_x - x
        dy = fall_y - y
        launch_direc = torch.atan2(dy, dx)
        self._launch_direc[env_ids, 0] = dx
        self._launch_direc[env_ids, 1] = dy

        # speed calibration
        speed = (1 + t*.025) * speed
        vx = torch.cos(phi) * torch.cos(launch_direc) * speed
        vy = torch.cos(phi) * torch.sin(launch_direc) * speed
        vz = torch.sin(phi) * speed

        self._ball_states[env_ids, 0] = torch.tensor(x, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 1] = torch.tensor(y, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 2] = torch.tensor(z, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 7] = torch.tensor(vx, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 8] = torch.tensor(vy, device='cuda', dtype=torch.float32)
        self._ball_states[env_ids, 9] = torch.tensor(vz, device='cuda', dtype=torch.float32)
        self._fall_pos[env_ids, 0] = torch.tensor(fall_x, device='cuda', dtype=torch.float32)
        self._fall_pos[env_ids, 1] = torch.tensor(fall_y, device='cuda', dtype=torch.float32)
        self._target_z[env_ids] = 0
        self._target_pos[env_ids, :2] = self._fall_pos[env_ids]
        self._is_ground[env_ids] = True

        self._initial_dist[env_ids] = dist
        self._initial_ball_pos[env_ids, :2] = torch.stack([x, y], dim=-1)
        self._initial_ball_pos[env_ids, 2] = z
        self._moved_dist[env_ids] = 0

        num_classes = self._onehot_body_part.shape[-1]
        left = num_envs//2
        random_indices = torch.zeros((num_envs,), dtype=torch.long, device=self.device)
        random_indices[:left] = 3
        random_indices[left:] = 5
        self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(random_indices, num_classes=num_classes)

    # endregion

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = torch.cat([self._target_actor_ids[env_ids], self._ball_actor_ids[env_ids]], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        return

    # for ball prev states
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_ball_pos[:] = self._ball_root_states[..., 0:3]


    ########################################
    # Set Command
    ########################################

    def set_command(self, buttons, axes, hat):
        self._buttons = buttons
        self._axes = axes
        self._hat = hat
    
    def _draw_task(self):
        # 얘 활성화하면 공 초기화, 타겟 초기화 작동 안 함. 그래서 다른 거 확인하려면 얘는 비활성화 해야함
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

        # for i, env_ptr in enumerate(self.envs):
        #     curr_verts = verts[i:i+1]
        #     curr_verts = curr_verts.reshape([2, 6])
        #     self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, heading_cols)

        return
    
    # def _update_camera(self):
    #     self.gym.refresh_actor_root_state_tensor(self.sim)
    #     char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
    #     cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
    #     cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
    #     cam_delta = cam_pos - self._cam_prev_char_pos

    #     new_cam_target = gymapi.Vec3(4, 10, 1)
    #     new_cam_pos = gymapi.Vec3(4, -6, 1)

    #     self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

    #     self._cam_prev_char_pos[:] = char_root_pos
    #     return



#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir, ball_pos, ball_vel, ball_ang_vel, fall_pos, target_z):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)
    
    tar_face_dir3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)
    local_tar_face_dir = quat_rotate(heading_rot, tar_face_dir3d)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]

    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    # fall pos
    fall_pos_tmp = torch.zeros_like(root_pos, device='cuda')
    fall_pos_tmp[..., :2] = fall_pos
    local_fall_pos = fall_pos_tmp - root_pos
    local_fall_pos = quat_rotate(heading_rot, local_fall_pos)

    obs = torch.cat([local_tar_dir, tar_speed, local_tar_face_dir, local_ball_pos, local_ball_vel, local_ball_ang_vel, local_fall_pos[..., :2]], dim=-1)
    return obs

def compute_heading_observations_simple(root_states, tar_dir, tar_speed, tar_face_dir, ball_pos, ball_vel, ball_ang_vel, fall_pos, target_z):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)
    
    tar_face_dir3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)
    local_tar_face_dir = quat_rotate(heading_rot, tar_face_dir3d)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]

    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    # fall pos
    fall_pos_tmp = torch.zeros_like(root_pos, device='cuda')
    fall_pos_tmp[..., :2] = fall_pos
    local_fall_pos = fall_pos_tmp - root_pos
    local_fall_pos = quat_rotate(heading_rot, local_fall_pos)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel], dim=-1)
    return obs

def compute_heading_observations_simple_target_fall(root_states, ball_pos, ball_vel, ball_ang_vel, target_fall_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    # fall pos
    target_fall_pos_tmp = torch.zeros_like(root_pos, device='cuda')
    target_fall_pos_tmp[..., :3] = target_fall_pos
    local_fall_pos = target_fall_pos_tmp - root_pos
    local_fall_pos = quat_rotate(heading_rot, local_fall_pos)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, local_fall_pos], dim=-1)
    return obs

def compute_heading_observations_simple_onehot(root_states, ball_pos, ball_vel, ball_ang_vel, onehot_body_part):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    # ball - character pos
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, onehot_body_part], dim=-1)
    return obs

def compute_heading_observations_simple_target_fall_onehot(root_states, ball_pos, ball_vel, ball_ang_vel, target_fall_pos, onehot_body_part):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    # fall pos
    target_fall_pos_tmp = torch.zeros_like(root_pos, device='cuda')
    target_fall_pos_tmp[..., :3] = target_fall_pos
    local_fall_pos = target_fall_pos_tmp - root_pos
    local_fall_pos = quat_rotate(heading_rot, local_fall_pos)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, local_fall_pos, onehot_body_part], dim=-1)
    return obs

def compute_heading_observations_simple_target_fall_onehot_contact(root_states, ball_pos, ball_vel, ball_ang_vel, target_fall_pos, onehot_body_part, onehot_contact_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    # fall pos
    target_fall_pos_tmp = torch.zeros_like(root_pos, device='cuda')
    target_fall_pos_tmp[..., :3] = target_fall_pos
    local_fall_pos = target_fall_pos_tmp - root_pos
    local_fall_pos = quat_rotate(heading_rot, local_fall_pos)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, local_fall_pos, onehot_body_part, onehot_contact_obs], dim=-1)
    return obs

def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, tar_face_dir, ball_pos, prev_ball_pos, 
                           ball_target, dt, ball_contact_flag, max_episode_length, progress_buf, touched, fall_pos,
                           ori_dist, nearest, rb_pos, launch_direc, target_fall_pos,
                           onehot_body_part):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    vel_reward_w = 1.0
    fall_pos_w = 1.0 #0.5
    fall_vel_w = 0.0 #0.5
    pos_err_scale = 5.
    vel_err_scale = 5.

    # # ball speed err - plane vel
    delta_ball_pos = ball_pos - prev_ball_pos
    ball_vel = delta_ball_pos / dt
    root_vel = root_pos - prev_root_pos
    # vel_err = torch.pow(ball_vel[..., :3] - root_vel[..., :3], 2).sum(dim=-1)
    # vel_reward = torch.exp(-vel_err_scale * vel_err)

    # for only contact reward - nearest part
    dist_ball_nearest = torch.norm(rb_pos[..., [1, 2, 10, 11, 13, 14], :] - ball_pos[...].unsqueeze(1), dim=2)
    # values, nearest_indices = torch.min(dist_ball_nearest, dim=1)
    # dist_picked = dist_ball_nearest[torch.arange(dist_ball_nearest.size(0)), nearest_indices]
    
    # dist_ball_foot_r = torch.norm(rb_pos[..., 11, :] - ball_pos[...], dim=1)
    # dist_ball_foot_l = torch.norm(rb_pos[..., 14, :] - ball_pos[...], dim=1)
    # dist_ball_leg_r = torch.norm(rb_pos[..., 10, :] - ball_pos[...], dim=1)
    # dist_ball_leg_l = torch.norm(rb_pos[..., 13, :] - ball_pos[...], dim=1)
    # dist_ball_chest = torch.norm(rb_pos[..., 1, :] - ball_pos[...], dim=1)
    # dist_ball_head = torch.norm(rb_pos[..., 2, :] - ball_pos[...], dim=1)

    # for selective target body part
    dist_picked = dist_ball_nearest[onehot_body_part==1]
    dist_reward = torch.exp(-pos_err_scale*dist_picked)

    # target fall pos reward
    root_pos_plane = torch.zeros_like(root_pos)
    root_pos_plane[..., :2] = root_pos[..., :2]
    root_pos_plane[..., 2] = 0.11
    tar_ball_vel = 0.1 * (target_fall_pos[..., :2] - root_pos_plane[..., :2]) + root_vel[..., :2]

    fall_pos_err = torch.norm(ball_pos - target_fall_pos, dim=1) # root_pos_plane
    fall_vel_err = torch.norm(ball_vel[..., :2] - tar_ball_vel, dim=1)
    fall_pos_reward = fall_pos_w * torch.exp(-pos_err_scale*fall_pos_err)
    fall_vel_reward = fall_vel_w * torch.exp(-vel_err_scale*fall_vel_err)

    # before contact - 100%
    reward = dist_reward
    
    # after contact
    # reward[touched==1] = fall_pos_w * fall_pos_reward[touched==1] + fall_vel_w * fall_vel_reward[touched==1]

    # filter - not facing to ball
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_sim = (facing_dir[:, :2] * -launch_direc[:]).sum(dim=1)

    reward[facing_sim < 0.5] = 0

    return reward

# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, max_episode_length,
                           enable_early_termination, termination_heights, root_pos, ball_pos, ball_contact_forces, touched, 
                           is_ground, initial_dist, moved_dist):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0

    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        # masked_contact_buf[:, contact_body_ids[:2], :] = 0
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        tar_has_contact = torch.any(torch.abs(ball_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)

        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, contact_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)

        has_failed = torch.logical_or(has_fallen, tar_fail)
        # has_failed = has_fallen

        #######################
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        # dists = torch.pow(root_pos[..., :2] - ball_pos[..., :2], 2).sum(dim=-1).sqrt()
        # relocate_mask = dists > relocate_dist
        # has_failed = relocate_mask
        # has_failed *= (progress_buf > 1)

        contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        contact = torch.any(contact, dim=-1)
        ball_fall = ball_pos[..., 2] < .13
        ball_fall = torch.logical_and(ball_fall, ~is_ground)
        ball_contact_force_magnitude = torch.pow(ball_contact_forces, 2).sum(dim=-1).sqrt()
        ball_contact_flag = ball_contact_force_magnitude > 1
        
        has_failed = torch.logical_or(has_failed, ball_fall)
        has_failed = torch.logical_or(has_failed, contact)
        has_failed = torch.logical_or(has_failed, touched>=2)
        has_failed[is_ground] = torch.logical_or(has_failed[is_ground], moved_dist[is_ground] > initial_dist[is_ground] * 1.2)

        # 충돌만을 위한 phase 1 에서만 켜기
        # has_failed = torch.logical_or(has_failed, ball_contact_flag)
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), terminated)

    return reset, terminated
