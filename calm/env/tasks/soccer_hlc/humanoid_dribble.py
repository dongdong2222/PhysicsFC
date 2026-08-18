import torch
from torch.nn.functional import normalize as torch_norm

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import json
import pickle
import os

# hyper parameters
INIT_BALL_DIST = None
INIT_BALL_Z = None
BALL_REWARD_W = None
HUMANOID_REWARD_W = None
DIST_REWARD_W = None
IS_INIT_BY_FILE = None

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2
TAR_BALL_ID = 3
TAR_ID = 4


################3

if IS_INIT_BY_FILE:
    with open('/workspace/CALM/blackhole/init_states/trapping_ball_state.pkl', 'rb') as f:
        init_states = pickle.load(f)
        init_ball_states = init_states['ball_state']
        init_dof_pos = init_states['dof_pos']
        init_dof_vel = init_states['dof_vel']
        init_root_states = init_states['root_state']
        init_rb_pos = init_states['rb_pos']
        init_rb_rot = init_states['rb_rot']
        init_rb_vel = init_states['rb_vel']
        init_rb_ang_vel = init_states['rb_ang_vel']

################3

r1_buf, r2_buf, r3_buf = list(), list(), list()
file_prefix = 'rewards_'
file_path = f'{file_prefix}{os.getpid()}.json'

def _cleanup_reward_logs(max_files=8):
    reward_files = []
    for name in os.listdir('.'):
        if name.startswith(file_prefix) and name.endswith('.json'):
            reward_files.append(name)

    reward_files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    for old_path in reward_files[max_files:]:
        try:
            os.remove(old_path)
        except OSError:
            pass

_cleanup_reward_logs()
if os.path.exists(file_path):
    os.remove(file_path)

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
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {'r1':list(), 'r2':list(), 'r3':list()}
        with open(file_path, 'w') as f:
            data['r1'].extend(r1_buf)
            data['r2'].extend(r2_buf)
            data['r3'].extend(r3_buf)
            json.dump(data, f)

        r1_buf, r2_buf, r3_buf = list(), list(), list()
#################3

class HumanoidDribble(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        global INIT_BALL_DIST
        global INIT_BALL_Z
        global BALL_REWARD_W
        global HUMANOID_REWARD_W
        global DIST_REWARD_W
        global IS_INIT_BY_FILE
        
        INIT_BALL_DIST = cfg["env"]["INIT_BALL_DIST"]
        INIT_BALL_Z = cfg["env"]["INIT_BALL_Z"]
        BALL_REWARD_W = cfg["env"]["BALL_REWARD_W"]
        HUMANOID_REWARD_W = cfg["env"]["HUMANOID_REWARD_W"]
        DIST_REWARD_W = cfg["env"]["DIST_REWARD_W"]
        IS_INIT_BY_FILE = cfg["env"]["IS_INIT_BY_FILE"]

        self._tar_speed_min = cfg["env"]["tarSpeedMin"]
        self._tar_speed_max = cfg["env"]["tarSpeedMax"]
        self._heading_change_steps_min = cfg["env"]["headingChangeStepsMin"]
        self._heading_change_steps_max = cfg["env"]["headingChangeStepsMax"]
        self._enable_rand_heading = cfg["env"]["enableRandHeading"]
        self._relocate_dist = cfg["env"]["relocate_dist"]
        
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

        self._ball_target = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)

        # if not self.headless:
        self._build_obj_state_tensors()

        # ball prev states
        self._prev_ball_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        num_actors = self.get_num_actors_per_env()
        self._ball_root_states = self._root_states.view(self.num_envs, num_actors, 13)[..., TAR_BALL_ID, :]

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 5 + 3 + 3 + 3 # move + 공 위치 + 공 속력 + (ang vel)
        return obs_size

    def _update_obj_actors(self):
        # 이 함수는 physics_step 즉 중간 시점에 일어남
        # reward 계산은 post_physics_step 에 일어나므로, 여기에 target 을 업데이트해도 문제없음

        humanoid_root_pos = self._humanoid_root_states[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_dir
        self._marker_pos[..., 2] = 0.0

        heading_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        heading_q = quat_from_angle_axis(heading_theta, heading_axis)
        self._marker_rot[:] = heading_q

        # face_theta = torch.atan2(self._tar_facing_dir[..., 1], self._tar_facing_dir[..., 0])
        # face_axis = torch.zeros_like(self._marker_pos)
        # face_axis[..., -1] = 1.0
        # face_q = quat_from_angle_axis(face_theta, heading_axis)
        # self._face_marker_rot[:] = face_q

        # update target by user control
        delta = torch.stack([self._tar_dir[..., 0] * self._tar_speed, self._tar_dir[..., 1] * self._tar_speed], dim=-1) * self.dt * 5
        self._target_pos[..., :2] = self._ball_states[..., :2] + delta
        self._target_pos[..., 2] = 0
        self._ball_target = self._target_pos[..., :2]
        
        # marker, ball, target 전부 업데이트. face marker 는 필요없음!
        marker_ids = torch.cat([self._marker_actor_ids, self._target_actor_ids, self._ball_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(marker_ids), len(marker_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        # if not self.headless:
        self._marker_handles = []
        self._face_marker_handles = []
        self._ball_handles = []
        self._target_handles = []
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
        asset_root = "calm/data/assets/mjcf/"
        asset_file = "soccer_ball.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = .5
        asset_options.linear_damping = .2
        asset_options.max_angular_velocity = 100.0
        asset_options.override_inertia = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        shape_property = self.gym.get_asset_rigid_shape_properties(self._ball_asset)
        shape_property[0].restitution = 1.0
        shape_property[0].friction = .5
        shape_property[0].rolling_friction = .5
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

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        # if not self.headless:
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

        # contact forces 잘못된 사용되고 있지만, 사실 안 쓰고 있으므로 넘어감
        self._ball_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., TAR_BALL_ID, :]

        self._ball_pos = self._ball_states[..., :3]
        self._ball_vel = self._ball_states[..., 7:10]
        self._ball_ang_vel = self._ball_states[..., 10:]

        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., TAR_ID, :]
        self._target_actor_ids = self._humanoid_actor_ids + TAR_ID
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._target_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., TAR_ID, :]

        self._target_pos = self._target_states[..., :3]
        self._target_vel = self._target_states[..., 7:10]

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

    # ################EVALUATION Code Start##############
    def _eval_tgt_update(self):
        testcase_index = torch.div(self.progress_buf[0], self._eval_change_steps, rounding_mode='floor').item()

        self._tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.from_numpy(self._eval_testcase_dict["target_speeds"]).to(device=self.device, dtype=torch.float)[:,testcase_index] + self._tar_speed_min
        self._tar_dir = torch.from_numpy(self._eval_testcase_dict["target_move_directions"]).to(device=self.device, dtype=torch.float)[:,testcase_index]
    

    def _eval_process(self):
        super()._eval_process()
                
        current_frame = self.progress_buf[0].item() % self._eval_change_steps
        testcase_index = torch.div(self.progress_buf[0], self._eval_change_steps, rounding_mode='floor').item()
        
        root_state = self._humanoid_root_states.clone()
        root_pos_xy = root_state[..., 0:2]
        ball_state = self._ball_states.clone()
        ball_pos_xy = ball_state[..., 0:2]
        
        char_ball_pos_diff = torch.norm(root_pos_xy - ball_pos_xy, dim=-1).cpu().numpy()
        
        self._eval_current_result.get("char_ball_pos_diff")[
            :, 
            testcase_index, 
            current_frame
        ] = char_ball_pos_diff  
              
        self._eval_current_result["char_ball_pos_lost_rate"][:, testcase_index, current_frame] = self._eval_check_frame_conditions_or(
            testcase_index=testcase_index,
            current_frame=current_frame,
            criteria_metrics=["char_tgt_move_diff","char_ball_pos_diff"],
            criteria_types=["angle", "distance"],
            criteria_funcs=[np.greater_equal, np.greater_equal]
        )
        
    # ################EVALUATION Code End##############
    
    def _reset_task(self, env_ids):
        n = len(env_ids)
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
        
        # during trainig
        elif self._enable_rand_heading:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            rand_face_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        else:
            rand_theta = torch.zeros(n, device=self.device)
            rand_face_theta = torch.zeros(n, device=self.device)
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        # tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)
        
        face_tar_dir = torch.stack([torch.cos(rand_face_theta), torch.sin(rand_face_theta)], dim=-1)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

        #################################
        # 반영이 안되서 비활성화. 대신 update_marker_target() 에다가 적용
        # delta = torch.stack([self._tar_dir[..., 0] * self._tar_speed, self._tar_dir[..., 1] * self._tar_speed], dim=-1)
        # self._target_states[..., :2] = self._ball_states[..., :2]# + delta
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_face_dir = self._tar_facing_dir
            ball_pos = self._ball_pos
            ball_vel = self._ball_vel
            ball_ang_vel = self._ball_ang_vel
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_face_dir = self._tar_facing_dir[env_ids]
            ball_pos = self._ball_pos[env_ids]
            ball_vel = self._ball_vel[env_ids]
            ball_ang_vel = self._ball_ang_vel[env_ids]

        obs = compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir, ball_pos, ball_vel, ball_ang_vel)
        
        # speed test
        # root_pos = self._humanoid_root_states[..., 0:3]
        # delta_root_pos = root_pos - self._prev_root_pos
        # root_vel = delta_root_pos / self.dt
        # print(torch.pow(root_states[0, 7:9], 2).sum().sqrt(), torch.pow(root_vel[0, :2], 2).sum().sqrt(), tar_speed)
        
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        ball_pos = self._ball_states[..., :3]

        # self.rew_buf[:] = compute_heading_reward(root_pos, self._prev_root_pos,  root_rot,
        #                                          self._tar_dir, self._tar_speed,
        #                                          self._tar_facing_dir, ball_pos, self._prev_ball_pos, 
        #                                          self._ball_target, self.dt)
        # print('reward:', sum(self.rew_buf).item())
        # print('', torch.sum(torch.pow(self._humanoid_root_states[0, 7:9], 2)).sqrt())

        r1, r2, r3 = compute_heading_reward(root_pos, self._prev_root_pos,  root_rot,
                                                 self._tar_dir, self._tar_speed,
                                                 self._tar_facing_dir, ball_pos, self._prev_ball_pos, 
                                                 self._ball_target, self.dt)
        self.rew_buf[:] = r1 + r2 + r3
        add_reward(r1.cpu().numpy().sum(), r2.cpu().numpy().sum(), r3.cpu().numpy().sum())

        return
    
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
                                                                           self._relocate_dist
                                                                           )
        return
    
    # reset for ball
    def _reset_actors(self, env_ids):
        if IS_INIT_BY_FILE:
            self.sample = np.random.choice(init_dof_pos.shape[0], size=len(env_ids))
        super()._reset_actors(env_ids)
        self._reset_ball(env_ids)
        print("reset actors!", env_ids)
        return
    
    def _reset_default(self, env_ids):
        if IS_INIT_BY_FILE:
            self._humanoid_root_states[env_ids] = torch.tensor(init_root_states[self.sample], device=self.device)
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
        if IS_INIT_BY_FILE:
            # init from trapping
            self._ball_states[env_ids] = torch.tensor(init_ball_states[self.sample], device=self.device)
            self._target_states[env_ids, :3] = self._ball_states[env_ids, :3]
        else:
            # 공이 캐릭터와 일정 거리 이상 멀어지면 어느정도 랜덤으로 캐릭터 근처로 다시 셋팅
            # 속도도 적당히 셋팅_target_states
            # marker 와 다르게 실제로 물리엔진 내에서의 공의 상태를 업데이트 해줘야함
            
            n = len(env_ids)
            root_pos = self._humanoid_root_states[env_ids, :3]
            self._ball_states[env_ids, :3] = root_pos + torch.rand(size=root_pos.shape, device=self.device) - INIT_BALL_DIST
            self._ball_states[env_ids, 2] = INIT_BALL_Z
            self._ball_states[env_ids, 7:10] = 0.0
            self._ball_states[env_ids, 10:13] = 0.0

            self._target_states[env_ids, :3] = self._ball_states[env_ids, :3]


    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._ball_actor_ids[env_ids]
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

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i:i+1]
            curr_verts = curr_verts.reshape([2, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, heading_cols)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_heading_observations(root_states, tar_dir, tar_speed, tar_face_dir, ball_pos, ball_vel, ball_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
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

    obs = torch.cat([local_tar_dir, tar_speed, local_tar_face_dir, local_ball_pos, local_ball_vel, local_ball_ang_vel], dim=-1)
    return obs


#@torch.jit.script
def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, tar_face_dir, ball_pos, prev_ball_pos, ball_target, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    vel_err_scale = 10.0
    pos_err_scale = 10.0
    tangent_err_w = 0.1

    # 비율
    ball_reward_w = BALL_REWARD_W
    humanoid_reward_w = HUMANOID_REWARD_W
    dist_reward_w = DIST_REWARD_W

    # for ball reward
    delta_ball_pos = ball_pos - prev_ball_pos 
    ball_vel = delta_ball_pos / dt
    tar_dir_speed = torch.sum(tar_dir * ball_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = ball_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1) # dongg

    tar_vel_err = (tar_speed - tar_dir_speed) / tar_speed # for norm
    tangent_vel_err = tangent_speed / tar_speed # for norm
    ball_reward = torch.exp(-vel_err_scale * (tangent_err_w * tar_vel_err * tar_vel_err + 
                        tangent_vel_err * tangent_vel_err))

    speed_mask = tar_dir_speed <= 0
    ball_reward[speed_mask] = 0


    # for character reward
    ball_humanoid_dir = torch_norm(ball_pos[..., :2] - root_pos[..., :2], dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(ball_humanoid_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * ball_humanoid_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = (tar_speed - tar_dir_speed) / tar_speed # for norm
    tangent_vel_err = tangent_speed / tar_speed # for norm
    humanoid_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err + 
                        tangent_err_w * tangent_vel_err * tangent_vel_err))

    # ball pos err reward
    cur_ball_pos = ball_pos[..., :2]
    dist_err = torch.pow(cur_ball_pos - root_pos[..., :2], 2).sum(dim=-1).sqrt()
    dist_reward = torch.exp(-pos_err_scale*dist_err)

    # reward = ball_reward_w * ball_reward + humanoid_reward_w * humanoid_reward + dist_reward_w * dist_reward

    # return reward

    return ball_reward_w * ball_reward, humanoid_reward_w * humanoid_reward, dist_reward_w * dist_reward


#@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, max_episode_length,
                           enable_early_termination, termination_heights, root_pos, ball_pos, relocate_dist):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0

    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)

        # nonstrike_body_force = masked_contact_buf
        # nonstrike_body_force[:, strike_body_ids, :] = 0
        # nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        # nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        # tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)

        # has_failed = torch.logical_or(has_fallen, False)
        has_failed = has_fallen

        #######################
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        # dists = torch.pow(root_pos[..., :2] - ball_pos[..., :2], 2).sum(dim=-1).sqrt()
        # relocate_mask = dists > relocate_dist
        # has_failed = relocate_mask
        # has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), terminated)

    return reset, terminated
