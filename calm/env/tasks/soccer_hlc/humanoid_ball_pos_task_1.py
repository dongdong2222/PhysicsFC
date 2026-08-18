import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
from torch.nn.functional import normalize as torch_norm
from torch.nn.functional import cosine_similarity

import pickle


BALL_INDEX = 0
TARGET_INDEX = 1
GOAL_TEST = False

IS_INIT = False # for initialization from trapping contact state

################3

if IS_INIT:
    with open('/workspace/CALM/blackhole/init_states/dribble_ball_state.pkl', 'rb') as f:
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


class HumanoidBallPos_1(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._ball_dist_min = 1.0
        self._ball_dist_max = 3.0
        # self._tar_dist_min = 40.0
        # self._tar_dist_max = 80.0
        # self._tar_height_min = 0.0
        # self._tar_height_max = 2.0
        # self._tar_speed_min = 30.0
        # self._tar_speed_max = 50.0
        self._tar_dist_min = cfg['env']['tarDistMin']
        self._tar_dist_max = cfg['env']['tarDistMax']
        self._tar_speed_min = cfg['env']['tarSpeedMin']
        self._tar_speed_max = cfg['env']['tarSpeedMax']
        self._tar_height_min = cfg['env']['tarHeightMin']
        self._tar_height_max = cfg['env']['tarHeightMax']

        self._near_dist = .5
        self._near_prob = 0.5

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        strike_body_names = cfg["env"]["strikeBodyNames"]
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0],
                                                                   strike_body_names)

        self._build_target_tensors()
        self._ori_dist = torch.zeros_like(self._target_states[..., 0, 0])
        self._ori_z = torch.zeros_like(self._target_states[..., 0, 0])
        self._nearest = torch.zeros_like(self._target_states[..., 0, 0])
        self._fastest = torch.zeros_like(self._target_states[..., 0, 0])
        self._tar_speed = torch.zeros_like(self._target_states[..., 0, 0])
        self._ori_direc = torch.zeros_like(self._target_states[..., 0, :3])

        # for angle, speed diff measure
        self._measures_angle = list()
        self._measures_speed = list()
        self._min_angle = torch.ones_like(self._target_states[..., 0, 0]) * 180
        self._min_speed_diff = torch.ones_like(self._target_states[..., 0, 0]) * 100

        # style assign test
        target = 70
        # latent_path = '/workspace/CALM/latents_90_new_unnorm.npy'
        # self._latent_target = torch.tensor(np.load(latent_path), device=self.device)[target].unsqueeze(0) # freekick motion
        # self._latent_target = self._latent_target.repeat([self.num_envs, 1])
        self._latent_action = torch.zeros([self.num_envs, 64], device=self.device)
        self._is_wrong_foot = torch.zeros(size=(self.num_envs,), dtype=torch.float32, device=self.device)
        self._touched = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._counter = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._kill_flag = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._far_flag = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)

        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 10 #+ 64 * 0
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        return

    def _load_target_asset(self):
        asset_root = "calm/data/assets/mjcf/"
        asset_file = "soccer_ball.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.05
        asset_options.linear_damping = 0.1
        asset_options.max_angular_velocity = 100.0
        asset_options.override_inertia = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        shape_property = self.gym.get_asset_rigid_shape_properties(self._target_asset)
        shape_property[0].restitution = 0.8
        shape_property[0].friction = .2
        shape_property[0].rolling_friction = .2
        self.gym.set_asset_rigid_shape_properties(self._target_asset, shape_property)

        # target object
        asset_root = "calm/data/assets/mjcf/"
        asset_file = "soccer_target.urdf"

        asset_options_2 = gymapi.AssetOptions()
        asset_options_2.angular_damping = 0.0
        asset_options_2.linear_damping = 0.0
        asset_options_2.max_angular_velocity = 0.0
        asset_options_2.fix_base_link = True
        asset_options_2.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset_2 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options_2)

        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0b10
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group,
                                              col_filter, segmentation_id)
        
        body_property = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle)
        self.gym.set_actor_rigid_body_properties(env_ptr, target_handle, body_property)

        self._target_handles.append(target_handle)

        # target object
        col_filter_2 = 2
        default_pose = gymapi.Transform()

        target_handle_2 = self.gym.create_actor(env_ptr, self._target_asset_2, default_pose, "target_obj", col_group,
                                              col_filter_2, segmentation_id)
        
        body_property_2 = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle_2)
        body_property_2[0].mass = .0
        self.gym.set_actor_rigid_body_properties(env_ptr, target_handle_2, body_property_2)

        self._target_handles.append(target_handle_2)

        return

    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., (num_actors - 2):, :]

        self._tar_actor_ids = num_actors * np.arange(self.num_envs)
        self._tar_actor_ids = np.expand_dims(self._tar_actor_ids, axis=-1)
        self._tar_actor_ids = self._tar_actor_ids + np.reshape(np.array(self._target_handles), [self.num_envs, 2])
        self._tar_actor_ids = self._tar_actor_ids.flatten()
        self._tar_actor_ids = to_torch(self._tar_actor_ids, device=self.device, dtype=torch.int32)
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - 2), :]

        return

    def _reset_actors(self, env_ids):
        if IS_INIT:
            self.sample = np.random.choice(init_dof_pos.shape[0], size=len(env_ids))

        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_default(self, env_ids):
        if IS_INIT:
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


    # def _reset_default(self, env_ids):
    #     self._humanoid_root_states[env_ids, 7] = torch.ones([len(env_ids),], device=self.device) * 1.
    #     self._reset_default_env_ids = env_ids
    #     return

    def _reset_target(self, env_ids):
        n = len(env_ids)
        if GOAL_TEST:
            self._measures_angle.extend([ang for ang in self._min_angle.cpu().numpy() if not np.isnan(ang)])
            self._measures_speed.extend(self._min_speed_diff.cpu().numpy())
            print('angle err:', np.mean(self._measures_angle[len(self._measures_angle)-100:]))
            print('speed err:', np.mean(self._measures_speed[2:]))

        if IS_INIT:
            # init from dribble
            self._target_states[env_ids, BALL_INDEX, :] = torch.tensor(init_ball_states[self.sample], device=self.device)
            
            rand_theta = self._target_states[env_ids, BALL_INDEX, :2] - self._humanoid_root_states[env_ids, :2]
            rand_theta = torch.atan2(rand_theta[..., 1], rand_theta[..., 0])
            rand_theta = rand_theta + (2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) - np.pi) / 8
            height_max = self._tar_height_max * torch.ones([n], dtype=self._target_states.dtype,
                                                   device=self._target_states.device)
            rand_height = (height_max - self._tar_height_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_height_min
            dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                    device=self._target_states.device)

            rand_dist = (dist_max - self._ball_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                    device=self._target_states.device) + self._ball_dist_min

            self._target_states[env_ids, TARGET_INDEX, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
            self._target_states[env_ids, TARGET_INDEX, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
            self._target_states[env_ids, TARGET_INDEX, 2] = rand_height
            
        else:
            init_near = torch.rand([n], dtype=self._target_states.dtype,
                                device=self._target_states.device) < self._near_prob
            dist_max = self._ball_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                    device=self._target_states.device)
            dist_max[init_near] = self._near_dist
            rand_dist = (dist_max - self._ball_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                    device=self._target_states.device) + self._ball_dist_min
            
            # ball
            rand_theta = 1 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) - .5 * np.pi
            self._target_states[env_ids, BALL_INDEX, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
            self._target_states[env_ids, BALL_INDEX, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
            self._target_states[env_ids, BALL_INDEX, 2] = 0.11

            rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
            axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
            rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

            # self._target_states[env_ids, 3:7] = rand_rot
            self._target_states[env_ids, BALL_INDEX, 7:10] = 0.0
            self._target_states[env_ids, BALL_INDEX, 10:13] = 0.0

            # target object
            rand_theta = rand_theta + 0.25 * (2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device) - np.pi)
            dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                    device=self._target_states.device)

            rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                    device=self._target_states.device) + self._tar_dist_min
            
            height_max = self._tar_height_max * torch.ones([n], dtype=self._target_states.dtype,
                                                    device=self._target_states.device)

            rand_height = (height_max - self._tar_height_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                    device=self._target_states.device) + self._tar_height_min
            
            self._target_states[env_ids, TARGET_INDEX, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
            self._target_states[env_ids, TARGET_INDEX, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
            self._target_states[env_ids, TARGET_INDEX, 2] = rand_height

        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)

        rand_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_speed_min

        self._target_states[env_ids, TARGET_INDEX, 3:7] = rand_rot
        self._target_states[env_ids, TARGET_INDEX, 7:10] = 0.0
        self._target_states[env_ids, TARGET_INDEX, 10:13] = 0.0
        self._ori_dist = torch.pow(self._target_states[..., BALL_INDEX, :3] - self._target_states[..., TARGET_INDEX, :3], 2).sum(dim=-1).sqrt()
        self._ori_z = torch.abs(self._target_states[..., BALL_INDEX, 2] - self._target_states[..., TARGET_INDEX, 2])
        self._nearest = torch.ones_like(self._target_states[..., 0, 0]) * 100
        self._tar_speed[env_ids] = rand_speed
        self._fastest = torch.zeros_like(self._fastest)
        self._ori_direc = torch_norm(self._target_states[..., TARGET_INDEX, :3] - self._target_states[..., BALL_INDEX, :3], dim=1)
        self._is_wrong_foot[env_ids] = 0
        self._touched[env_ids] = 0
        self._counter[env_ids] = 0
        self._kill_flag[env_ids] = 0
        self._far_flag[env_ids] = 0

        if GOAL_TEST:
            self._min_angle[env_ids] = torch.ones_like(self._target_states[..., 0, 0]) * 180
            self._min_speed_diff[env_ids] = torch.ones_like(self._target_states[..., 0, 0]) * 100

        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._tar_actor_ids), len(self._tar_actor_ids))
        
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_states = self._target_states
            tar_speed = self._tar_speed
            ori_direc = self._ori_direc
            # latent_target = self._latent_target
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
            tar_speed = self._tar_speed[env_ids]
            ori_direc = self._ori_direc[env_ids]
            # latent_target = self._latent_target[env_ids]
        
        # obs = compute_shoot_observations(root_states, tar_states, tar_speed.unsqueeze(1), ori_direc, latent_target)
        obs = compute_shoot_observations(root_states, tar_states, tar_speed.unsqueeze(1), ori_direc)

        # print(f'speed:, {torch.pow(root_states[..., 7:9], 2).sum(dim=-1).sqrt()[0].item():.2f}, \
        #         ball:, {torch.pow(tar_states[..., BALL_INDEX, 7:10], 2).sum(dim=-1).sqrt()[0].item():.2f}')

        return obs

    def _compute_reward(self, actions):
        tar_pos = self._target_states[..., BALL_INDEX, 0:3]
        # tar_rot = self._target_states[..., BALL_INDEX, 3:7]
        tar_vel = self._target_states[..., BALL_INDEX, 7:10]
        char_root_state = self._humanoid_root_states
        # strike_body_vel = self._rigid_body_vel[..., self._strike_body_ids[0], :]
        tar_obj_pos = self._target_states[..., TARGET_INDEX, :3]
        
        axis = torch.tensor([1, 0, 0], device='cuda', dtype=torch.float32)
        axis = axis.repeat([self.num_envs, 1])
        
        # for flag
        cur_dist = torch.pow(tar_pos[..., :3] - tar_obj_pos[..., :3], 2).sum(dim=-1).sqrt()
        self._nearest = torch.min(cur_dist, self._nearest)

        # for speed save
        cur_speed = torch.pow(tar_vel[..., :3], 2).sum(dim=-1).sqrt()
        tar_speed = self._tar_speed
        self._fastest = torch.max(cur_speed, self._fastest)

        # for wrong foot filtering
        ball_speed = torch.norm(self._target_states[..., BALL_INDEX, 7:10], dim=1)
        foot_speed = torch.norm(self._rigid_body_vel[..., [11, 14], :], dim=2)
        wrong = torch.logical_and(ball_speed > 1, foot_speed[..., 1] > 5)
        right = torch.logical_and(ball_speed > 1, foot_speed[..., 0] > 5)

        self._is_wrong_foot[right] = 2
        self._is_wrong_foot[torch.logical_and(wrong, self._is_wrong_foot==0)] = 1

        has_vel = torch.norm(self._target_states[..., BALL_INDEX, 7:10], dim=1) > 0.5 # 0.1
        self._touched[torch.logical_and(self._touched==0, has_vel)] = 1
        self._counter[self._touched>0] += 1
        self._touched[self._counter >= 10] = 2
        # self._kill_flag[self._counter > 5] = 1


        self.rew_buf = compute_shoot_reward(tar_pos, tar_obj_pos, self._ori_dist, self._ori_z,
                                               char_root_state[..., :3], char_root_state[..., 7:10],
                                               char_root_state[..., 3:7], axis, self._nearest, tar_vel,
                                               self._fastest, tar_speed, self._ori_direc,
                                               self._min_angle, self._min_speed_diff,
                                               self._touched)
        
        # self.rew_buf[:] = rew_pos + rew_dir + rew_speed
        
        print('reward:', sum(self.rew_buf))
        # print(rew_pos[0].cpu().numpy() / 0., rew_dir[0].cpu().numpy() / .5, rew_speed[0].cpu().numpy() / .5)

        if GOAL_TEST:
            self._min_angle = min_angle
            self._min_speed_diff = min_speed_diff

        return

    def _compute_reset(self):

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self._tar_contact_forces,
                                                                           self._strike_body_ids,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights,
                                                                           self._is_wrong_foot,
                                                                           self._touched)
        return
    
    def set_latent_action(self, actions):
        self._latent_action = actions
        return
    
    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        cols_2 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._target_states[..., BALL_INDEX, 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        starts = self._target_states[..., BALL_INDEX, 0:3]
        ends = self._target_states[..., TARGET_INDEX, 0:3]
        verts_2 = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            curr_verts_2 = verts_2[i]
            curr_verts_2 = curr_verts_2.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)
            self.gym.add_lines(self.viewer, env_ptr, curr_verts_2.shape[0], curr_verts_2, cols_2)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_shoot_observations(root_states, tar_states, tar_speed, tar_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    # target state
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0, 0:3]
    tar_rot = tar_states[:, 0, 3:7]
    tar_vel = tar_states[:, 0, 7:10]
    tar_ang_vel = tar_states[:, 0, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    # target obj state
    # heading_rot = torch_utils.calc_heading_quat_inv(tar_pos)
    tar_obj_pos = tar_states[:, 1, 0:3]
    local_tar_obj_pos = tar_obj_pos - root_pos
    local_tar_obj_pos[..., -1] = tar_obj_pos[..., -1]
    local_tar_obj_pos = quat_rotate(heading_rot, local_tar_obj_pos)

    # target height
    tar_obj_z = tar_states[:, 1, 2:3]

    # ball - target pos
    ball_tar_pos = tar_states[..., 1, :3] - tar_states[..., 0, :3]

    # target ball direc
    # tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    tar_dir3d = tar_dir
    # print(tar_dir.shape, tar_dir3d.shape)
    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:3]

    obs = torch.cat([local_tar_pos, local_tar_vel, tar_speed, local_tar_dir], dim=-1)
    return obs


# @torch.jit.script
def compute_shoot_reward(tar_pos, tar_obj_pos, ori_dist, ori_z, root_pos, root_vel, root_rot, axis, nearest, ball_vel, fastest, tar_speed, ori_direc, min_angle, min_speed_diff, touched):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    distance_err_w = 0.0 # 0.6
    ball_vel_reward_w = 1. # .5
    tar_speed_reward_w = .0 # .5
    distance_err_scaler = 1.0
    direction_err_w = .2
    heading_err_w = .2
    z_err_w = .2
    direction_err_scaler = .1
    speed_err_scaler = 10.

    # ball - target distance
    # distance_err = torch.pow(tar_pos - tar_obj_pos, 2).sum(dim=-1).sqrt()
    # distance_err = nearest
    distance_err = nearest / (ori_dist - 1e-10)
    # distance_reward = torch.clip(1 - distance_err, 0)
    distance_reward = 1 - distance_err
    # distance_reward[distance_mask] = 0

    # z error
    # z_err = torch.abs(tar_pos[..., -1] - tar_obj_pos[..., -1])
    # z_err /= (ori_z + 1e-10)
    # z_reward = torch.clip(1 - z_err, 0)

    # ball speed to target error
    # target_direction = torch_norm(tar_obj_pos - tar_pos, dim=-1)
    ball_direction = torch_norm(ball_vel, dim=-1)[..., :2]
    # tar_vel = torch.clamp(torch.mul(ball_vel[..., :3], ori_direc).sum(dim=-1) / tar_speed, -1)
    tar_vel_err = torch.exp(-10*torch.pow(ball_vel[..., :3] / tar_speed.unsqueeze(1) - ori_direc, 2).sum(dim=-1))
    ball_vel_reward = tar_vel_err

    # target speed
    speed = torch.pow(ball_vel, 2).sum(dim=-1).sqrt()
    speed_err = (tar_speed - speed) / tar_speed
    # tar_speed_reward = torch.exp(-speed_err_scaler*speed_err)
    tar_speed_reward = 1 - torch.abs(tar_speed - speed) / tar_speed

    # vel direction err
    # root_vel[..., -1] = 0
    # speed = torch.pow(root_vel, 2).sum(dim=-1).sqrt()
    # target_vec = torch_norm(tar_pos - root_pos, dim=-1)
    # target_vec[..., -1] = 0
    # # vel_dir_err = cosine_similarity(target_vec, root_vel, dim=-1)
    # vel_dir_err = torch.mul(target_vec, root_vel).sum(dim=-1)

    # # vel_dir_reward = torch.clip(vel_dir_err, 0) * speed
    # vel_dir_reward = vel_dir_err


    ######################################################
    # for merged reward
    ######################################################
    # vel_err_scale = 5 # 0.25
    # tangent_err_w = 5

    # tar_dir_speed = torch.sum(ori_direc * ball_vel[..., :2], dim=-1)

    # tar_dir_vel = tar_dir_speed.unsqueeze(-1) * ori_direc
    # tangent_vel = ball_vel[..., :2] - tar_dir_vel

    # tangent_speed = torch.sum(tangent_vel, dim=-1)

    # tar_vel_err = (tar_speed - tar_dir_speed) / tar_speed # for norm
    # tangent_vel_err = tangent_speed / tar_speed # for norm
    # dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err + 
    #                     tangent_err_w * tangent_vel_err * tangent_vel_err))

    # reward = dir_reward

    reward = distance_err_w * distance_reward + \
            ball_vel_reward_w * ball_vel_reward + \
            tar_speed_reward_w * tar_speed_reward
            
        # direction_err_scaler * direction_err_w * vel_dir_reward + \
        # direction_err_scaler * heading_err_w * direction_reward

    # angle = torch.arccos(tar_vel) * 180 / 3.14
    # min_angle = torch.min(min_angle, angle)
    # min_speed_diff = torch.min(min_speed_diff, tar_speed-speed)
    # return distance_err_w * distance_reward, ball_vel_reward_w * ball_vel_reward, tar_speed_reward_w * tar_speed_reward, min_angle, min_speed_diff
    
    # reward = torch.exp(-1*torch.norm(latent_target - actions, dim=1))
    # style_reward = cosine_similarity(latent_target, actions, dim=-1)
    # reward = reward * .5 + style_reward * .5
    return reward


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           tar_contact_forces, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, is_wrong_foot, touched):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
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

        tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)

        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)

        has_failed = torch.logical_or(has_fallen, tar_fail)
        has_failed = torch.logical_or(has_failed, touched==2)
        # has_failed = torch.logical_or(has_failed, is_wrong_foot==1)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), terminated)

    return reset, terminated
