import os
import sys
import time

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

import torch
import torch.nn.functional


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
# 난수 생성 시드 고정
torch.manual_seed(42)

# import calm local packages
from utils.config import parse_sim_params
from utils import torch_utils
from utils import gamepad

import yaml
import numpy as np

from demo_utils import *

MOVE = "MOVE"
DRIBBLE = "DRIBBLE"
KICK = "KICK"
TRAP = "TRAP"
DYNAMIC = "DYNAMIC"

class DynamicSwitchingEnv(FootballSkillMapper):
    def action_to_pd_targets(self, action):
        return self._pd_action_offset + self._pd_action_scale * action

    def _add_character_actors(self, actor_asset, actor_name, col_group, col_filter, pos_list, color=None):
        self.dict_character_actor_idx_offset.update({
            actor_name: self.gym.get_actor_count(self.env)
        })
        self.dict_actor_num_bodies.update({
            actor_name: self.gym.get_asset_rigid_body_count(actor_asset)
        })
        self.dict_actor_dof_idx_offset.update({
            actor_name: self.gym.get_sim_dof_count(self.sim)
        })
        self.dict_actor_num_dofs.update({
            actor_name: self.gym.get_asset_dof_count(actor_asset)
        })
        self.dict_actor_body_idx_offset.update({
            actor_name: self.gym.get_sim_rigid_body_count(self.sim)
        })
        
        actor_handles = []
        for i in range(self.num_character):
            if actor_name == "humanoid":
                col_filter = col_filter << 1

            transform = gymapi.Transform()
            transform.p = pos_list[i]
            actor_handle = self.gym.create_actor(
                self.env,
                actor_asset,
                transform,
                f"{actor_name}_{i}",
                col_group,  #collision group
                col_filter,
                0   #segmentation id
            )
            actor_handles.append(actor_handle)
            if actor_name == "humanoid":
                self.gym.enable_actor_dof_force_sensors(self.env, actor_handle)
            if color is not None:
                if actor_name == "humanoid" and self.viz_marker:
                        skin_color = [232./255., 188./255., 133./255.]
                        # black vs sky blue + cobalt blue
                        # team_upper_color = [35./255., 35./255., 35./255.] if i%2 == 0 else [155./255,246./255,245./255]
                        # team_lower_color = [35./255., 35./255., 35./255.] if i%2 == 0 else [47./255,92./255,182./255.]
                        # black vs white
                        # team_upper_color = [35./255., 35./255., 35./255.] if i%2 == 0 else [220./255,220./255,220./255]
                        # team_lower_color = team_upper_color
                        # team_lower_color = [35./255., 35./255., 35./255.] if i%2 == 0 else [11./255,63./255,99./255.]
                        # team_socks = [200./255, 200./255, 200./255] if i%2 == 0 else [220./255,220./255,220./255.]
                        
                        # korea red vs korea black
                        if ((i+2)//2) == (self.num_character//2): #goal keeper
                            team_upper_color = [0.431376, 0.905886, 0.698036] if i%2 == 0 else [0.431376, 0.905886, 0.698036]
                            team_lower_color = [0.035294, 0.552941, 0.356861] if i%2 == 0 else [0.035294, 0.552941, 0.356861]
                        else:
                            team_upper_color = [1., 0.45, 0.50] if i%2 == 0 else [0.15, 0.21, 0.23]
                            team_lower_color = [0.996081, 0.388239, 0.419613] if i%2 == 0 else [0.109805, 0.145097, 0.164706]
                        team_socks = skin_color
                        # team_socks = [255./255, 100./255, 100./255.] if i%2 == 0 else [150./255, 100./255, 130./255]
                        
                        
                        shoes_color = [*np.random.rand(3)]
                        color_dict = {
                            'pelvis': team_upper_color,
                            'torso': team_upper_color,
                            'head': skin_color,
                            'right_upper_arm': team_upper_color,
                            'right_lower_arm': skin_color,
                            'right_hand': skin_color,
                            'left_upper_arm': team_upper_color,
                            'left_lower_arm': skin_color,
                            'left_hand': skin_color,
                            'right_thigh': team_lower_color,
                            'right_shin': team_socks,
                            'right_foot': shoes_color,
                            'left_thigh': team_lower_color,
                            'left_shin': team_socks,
                            'left_foot': shoes_color,
                        }
                        for j, name in enumerate(self.gym.get_actor_rigid_body_names(self.env, actor_handles[i])):
                            self.gym.set_rigid_body_color(self.env, actor_handles[i], j, gymapi.MESH_VISUAL, gymapi.Vec3(*color_dict[name]))
                else:
                    for j, name in enumerate(self.gym.get_actor_rigid_body_names(self.env, actor_handles[i])):
                        self.gym.set_rigid_body_color(self.env, actor_handles[i], j, gymapi.MESH_VISUAL, color)
        
        self.dict_character_actor_handles.update({
            actor_name: actor_handles
        })

        if actor_name.startswith("marker"):
            self.marker_handles.extend(actor_handles)
            self.marker_handles.sort()

    
    def _add_global_actors(self, actor_asset, actor_name, col_group, col_filter, pos, color=None):
        self.dict_global_actor_idx.update({
            actor_name: self.gym.get_actor_count(self.env)
        })
        self.dict_actor_num_bodies.update({
            actor_name: self.gym.get_asset_rigid_body_count(actor_asset)
        })
        self.dict_actor_num_dofs.update({
            actor_name: self.gym.get_asset_dof_count(actor_asset)
        })
        self.dict_actor_dof_idx_offset.update({
            actor_name: self.gym.get_sim_dof_count(self.sim)
        })
        self.dict_actor_body_idx_offset.update({
            actor_name: self.gym.get_sim_rigid_body_count(self.sim)
        })
        print(actor_name, 0)
        transform = gymapi.Transform()
        transform.p = pos

        actor_handle = self.gym.create_actor(
            self.env,
            actor_asset,
            transform,
            actor_name,
            col_group,  #collision group
            col_filter,
            0   #segmentation id
        )
        print(actor_name, 1)
        self.dict_global_actor_handle.update({
            actor_name: actor_handle
        })

        print(actor_name, 2)
        if color is not None:
            print(actor_name, "setup color")
            for j, name in enumerate(self.gym.get_actor_rigid_body_names(self.env, actor_handle)):
                self.gym.set_rigid_body_color(self.env, actor_handle, j, gymapi.MESH_VISUAL, color)

        if actor_name.startswith("marker"):
            self.marker_handles.append(actor_handle)
            self.marker_handles.sort()


    def __init__(self):
        self.gym = None
        self.sim = None
        self.env = None
        self.viewer = None
        self.device = 'cuda:0'
        self.cfg = None
        self.viz_marker = True
        self.cam_follow = True
        self.game_mode = TRAP
        self.need_reset = True

        self.dict_character_actor_handles = {}
        self.dict_character_actor_idx_offset = {}
        self.dict_global_actor_handle = {}
        self.dict_global_actor_idx = {}
        self.dict_actor_body_idx_offset = {}
        self.dict_actor_num_bodies = {}
        self.dict_actor_num_dofs = {}
        self.dict_actor_dof_idx_offset = {}
        self.marker_handles = []
        self.marker_ids = None
        self.actor_ids_to_update_root = []
        self.actor_ids_to_update_dof = []

        self._pd_action_offset = None
        self._pd_action_scale = None
        
        self.num_character = 22
        self.my_character_idx = 0 #num_character // 2 - (num_character % 2) + 1
        self.humanoid_init_distance_scale = 8.0
        self.joystick = gamepad.Joystick(joystick_idx=0)
        self.joystick_coroutine = self.joystick.command() if self.joystick.is_available() else None
        self.max_target_dribble_speed = 7.0
        self.min_target_dribble_speed = 0.0
        self.max_target_kick_speed = 35.0
        self.min_target_kick_speed = 5.0
        self.max_target_kick_angle = np.deg2rad(45.0)
        self.min_target_kick_angle = np.deg2rad(0.0)
        self.max_target_move_speed = 7.0
        self.min_target_move_speed = 0.0
        self.threshold_override_r_stick = 0.2
        self.threshold_to_be_move = 2.0
        self.control_frequency_inv = 2
        self.marker_disable_time = 0.3
        self.render_frequency_inv = 2

        self.camera_interpolate_step = 20
        self.switch_distance_threshold = 25 #float('inf') 
        self.camera_interpolate_buffer = torch.zeros(self.camera_interpolate_step, 3, dtype=torch.float, device=self.device)
        self.checker_xpad_hat_idle = True

        self.dribble_state_buffer = None
        self.is_ground_passing = False
        self.is_lob_passing = False
        self.is_kickking = False
        self.is_passing_to = -1
        

        # Build Gym & Sim
        self.gym = gymapi.acquire_gym()
        # parse arguments
        args = gymutil.parse_arguments(description="Body Physics Properties Example")
        with open(os.path.join(os.getcwd(), "calm/calm_exp/cfg/env.yaml"), 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        sim_params = parse_sim_params(args, self.cfg, None)
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        self.sim_params = self.gym.get_sim_params(self.sim)

        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-100, -100, 0), gymapi.Vec3(100, 100, 20), 1)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # if not self.viz_marker:
            # Add Ground
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["plane"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["plane"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

        # Load Humanoid Asset & Configure
        humanoid_asset_root = "calm/data/assets"
        humanoid_asset_file_name = "mjcf/amp_humanoid_shoe.xml"
        humanoid_asset_path = os.path.join(humanoid_asset_root, humanoid_asset_file_name)
        humanoid_asset_root = os.path.dirname(humanoid_asset_path)
        humanoid_asset_file_name = os.path.basename(humanoid_asset_path)
        humanoid_asset_options = gymapi.AssetOptions()
        humanoid_asset_options.angular_damping = 0.5
        humanoid_asset_options.linear_damping = 0.0
        humanoid_asset_options.max_angular_velocity = 100.0
        humanoid_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        humanoid_asset = self.gym.load_asset(
            self.sim, 
            humanoid_asset_root, 
            humanoid_asset_file_name, 
            humanoid_asset_options
        )
        humanoid_shape_properties = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        for humanoid_shape_property in humanoid_shape_properties:
            humanoid_shape_property.restitution = 0.0
            humanoid_shape_property.friction = 1.0
            humanoid_shape_property.rolling_friction = 0.5
        self.gym.set_asset_rigid_shape_properties(humanoid_asset, humanoid_shape_properties)

        # Load Ball Asset & Configure
        ball_asset_root = "calm/data/assets/mjcf/"
        ball_asset_file_name = "soccer_ball.urdf"
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.angular_damping = .05
        ball_asset_options.linear_damping = .1
        ball_asset_options.max_angular_velocity = 100.0
        ball_asset_options.override_inertia = True
        ball_asset_options.density = 80.71
        ball_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        ball_asset = self.gym.load_asset(
            self.sim,
            ball_asset_root,
            ball_asset_file_name,
            ball_asset_options
        )
        ball_shape_property = self.gym.get_asset_rigid_shape_properties(ball_asset)
        ball_shape_property[0].restitution = 0.8
        ball_shape_property[0].friction = .2
        ball_shape_property[0].rolling_friction = .2
        self.gym.set_asset_rigid_shape_properties(ball_asset, ball_shape_property)

        heading_marker_col_filter = ~( 0b100 << self.num_character )
        ball_col_filter = 0b01
        print("######################### ball-marker collision:", heading_marker_col_filter & ball_col_filter)
        print(f"Start creating actors")

        # Create Humanoid Actor
        print(f"creating humanoid...")
        humanoid_col_filter = 0b1
        print("######################### humanoid-marker collision:", heading_marker_col_filter & humanoid_col_filter)
        print("######################### humanoid-ball collision:", ball_col_filter & humanoid_col_filter)
        self.humanoid_init_pos_xy = self.humanoid_init_distance_scale * torch.tensor([
            [-3.,   -3.],
            [3.,    -3.],
            [-3,    -1],
            [3.,    -1.],
            [-3,    1.],
            [3.,    1.],
            [-3.,   3.],
            [3.,    3.],
            [-1.,   -2.],
            [1.,    -2.],
            [-1.1,   0.],
            [0.9,    0.],
            [-1.,   2.],
            [1.,    2.],
            [1.1,    0.],
            [-0.9,   0.],
            [2.,    -1.],
            [-2.,   -1.],
            [2.,    1.],
            [-2.,   1.],
            [-4.,    0.],
            [4.,   0.],
        ], device=self.device)
        self.humanoid_formation_x_ratio = torch.tensor(
            [
                2,2,2,2,2,2,2,2,    # 4B
                3,3,3,3,3,3,        # 3D
                4,4,                # 1OM
                5,5,5,5,            # 2ST
                1,1                 # 1GK
            ], 
            device=self.device
        ) / 7.5
        humanoid_pos_list = [
            gymapi.Vec3(*self.humanoid_init_pos_xy[i], 0.89) for i in range(self.num_character)
        ]
        self._add_character_actors(humanoid_asset, "humanoid", 0, humanoid_col_filter, humanoid_pos_list, gymapi.Vec3(0., 0.5, 0.))
        
        if self.viz_marker:
            # Load Field Asset & Configure
            field_asset_root = "calm/data/assets/mjcf/"
            field_asset_file_name = "soccer_field.urdf"
            field_asset_options = gymapi.AssetOptions()
            field_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            field_asset_options.fix_base_link = True
            field_asset_options.flip_visual_attachments = True
            field_asset_options.vhacd_enabled = True
            field_asset_options.vhacd_params = gymapi.VhacdParams()
            field_asset_options.vhacd_params.resolution = 500000
            field_asset_options.vhacd_params.max_num_vertices_per_ch = 1024
            # field_asset_options.vhacd_params.min_volume_per_ch = 0.00001

            field_asset = self.gym.load_asset(
                self.sim,
                field_asset_root,
                field_asset_file_name,
                field_asset_options
            )
            field_shape_property = self.gym.get_asset_rigid_shape_properties(field_asset)
            field_shape_property[0].restitution = 0.2
            field_shape_property[0].friction = 1.0
            field_shape_property[0].rolling_friction = 1.0
            self.gym.set_asset_rigid_shape_properties(field_asset, field_shape_property)


            # # Load ground Asset & Configure
            # ground_asset_root = "calm/data/assets/mjcf/"
            # ground_asset_file_name = "soccer_field_ground.urdf"
            # ground_asset_options = gymapi.AssetOptions()
            # ground_asset_options.fix_base_link = True
            # ground_asset_options.override_inertia = True
            # ground_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            # ground_asset = self.gym.load_asset(
            #     self.sim,
            #     ground_asset_root,
            #     ground_asset_file_name,
            #     ground_asset_options
            # )
            # ground_shape_property = self.gym.get_asset_rigid_shape_properties(ground_asset)
            # ground_shape_property[0].restitution = self.cfg["env"]["plane"]["restitution"]
            # ground_shape_property[0].friction = self.cfg["env"]["plane"]["staticFriction"]
            # ground_shape_property[0].rolling_friction = self.cfg["env"]["plane"]["dynamicFriction"]
            # self.gym.set_asset_rigid_shape_properties(ground_asset, ground_shape_property)


            # Load Heading Marker Asset
            heading_marker_asset_root = "calm/data/assets/mjcf/"
            heading_marker_asset_file = "heading_marker.urdf"
            heading_marker_asset_options = gymapi.AssetOptions()
            heading_marker_asset_options.angular_damping = 0.01
            heading_marker_asset_options.linear_damping = 0.01
            heading_marker_asset_options.max_angular_velocity = 100.0
            heading_marker_asset_options.density = 1.0
            heading_marker_asset_options.fix_base_link = True
            heading_marker_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            heading_marker_asset = self.gym.load_asset(
                self.sim,
                heading_marker_asset_root,
                heading_marker_asset_file,
                heading_marker_asset_options
            )

            # Load Heading Marker Asset
            xy_traj_marker_asset_root = "calm/data/assets/mjcf/xy_trajectory_marker/urdf/"
            xy_traj_marker_asset_file = "xy_trajectory_marker.urdf"
            xy_traj_marker_asset_options = gymapi.AssetOptions()
            xy_traj_marker_asset_options.fix_base_link = True
            xy_traj_marker_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            xy_traj_marker_asset = self.gym.load_asset(
                self.sim,
                xy_traj_marker_asset_root,
                xy_traj_marker_asset_file,
                xy_traj_marker_asset_options
            )
            xy_traj_marker_shape_properties = self.gym.get_asset_rigid_shape_properties(xy_traj_marker_asset)
            for xy_traj_marker_shape_property in xy_traj_marker_shape_properties:
                xy_traj_marker_shape_property.filter = heading_marker_col_filter
            self.gym.set_asset_rigid_shape_properties(xy_traj_marker_asset, xy_traj_marker_shape_properties)


            skill_indicator_asset_root = "calm/data/assets/mjcf"
            skill_indicator_asset_file = "location_marker.urdf"
            skill_indicator_asset_options = gymapi.AssetOptions()
            skill_indicator_asset_options.fix_base_link = True
            skill_indicator_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            skill_indicator_asset = self.gym.load_asset(
                self.sim,
                skill_indicator_asset_root,
                skill_indicator_asset_file,
                skill_indicator_asset_options
            )

            selection_marker_asset_root = "calm/data/assets/mjcf"
            selection_marker_asset_file = "selection_marker.urdf"
            selection_marker_asset_options = gymapi.AssetOptions()
            selection_marker_asset_options.fix_base_link = True
            selection_marker_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            selection_marker_asset = self.gym.load_asset(
                self.sim,
                selection_marker_asset_root,
                selection_marker_asset_file,
                selection_marker_asset_options
            )

            # Load Kick Traj Asset & Configure
            kick_traj_asset_root = "calm/data/assets/mjcf/trajectory_marker/urdf/"
            kick_traj_asset_file_name = "trajectory_marker.urdf"
            kick_traj_asset_options = gymapi.AssetOptions()
            kick_traj_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            kick_traj_asset_options.fix_base_link = True

            kick_traj_asset = self.gym.load_asset(
                self.sim,
                kick_traj_asset_root,
                kick_traj_asset_file_name,
                kick_traj_asset_options
            )
            kick_traj_shape_properties = self.gym.get_asset_rigid_shape_properties(kick_traj_asset)
            for kick_traj_shape_property in kick_traj_shape_properties:
                kick_traj_shape_property.filter = heading_marker_col_filter
            self.gym.set_asset_rigid_shape_properties(kick_traj_asset, kick_traj_shape_properties)

            # Create Move Direction Marker
            print(f"creating move_dir marker...")
            marker_pos_list = [
                gymapi.Vec3(*self.humanoid_init_pos_xy[i], 0.0) for i in range(self.num_character)
            ]
            self._add_character_actors(xy_traj_marker_asset, "marker_move_dir", 1, heading_marker_col_filter, marker_pos_list, gymapi.Vec3(1.0, 0.0, 0.0))

            # Create Face Direction Marker
            print(f"creating face_dir marker...")
            self._add_character_actors(heading_marker_asset, "marker_facing_dir", 1, heading_marker_col_filter, marker_pos_list, gymapi.Vec3(0.0, 0.0, 1.0))

            # Create Skill Indicator Sphere
            print("creating skill indicator...")
            indicator_pos_list = [
                gymapi.Vec3(*self.humanoid_init_pos_xy[i], 2.0) for i in range(self.num_character)
            ]
            self._add_character_actors(skill_indicator_asset, "marker_skill_state", 1, heading_marker_col_filter, indicator_pos_list)

            # Create Selection(Current Controlling Character) Marker
            print("creating Selection indicator...")
            self._add_global_actors(selection_marker_asset, "marker_selection", 1, heading_marker_col_filter, gymapi.Vec3(*self.humanoid_init_pos_xy[self.my_character_idx], 2.3), gymapi.Vec3(1.0, 0.6, 0.))

            # Create Kick Trajectory Marker
            print("creating Kick Trajectory Marker...")
            self._add_global_actors(kick_traj_asset, "marker_kick_traj", 1, heading_marker_col_filter, gymapi.Vec3(0., 0., 0.11))
            
            # Create Field Actor
            self._add_global_actors(field_asset, "field", 0, 0b0, gymapi.Vec3(0., 0., 0.0001))

            # # Create Ground Actor
            # self._add_global_actors(ground_asset, "ground", 0, 0b0, gymapi.Vec3(0., 0., 0.))

            print(f"Done creating actors")
            ##################################

        # Create Ball Actor
        self._add_global_actors(ball_asset, "ball", 0, ball_col_filter, gymapi.Vec3(*(self.humanoid_init_pos_xy[0] + torch.tensor([0.4, 0.], device=self.device)), 0.12))
        ball_body_prop = self.gym.get_actor_rigid_body_properties(self.env, self.dict_global_actor_handle["ball"])[0]
        print(f"\n\n\n\nball inertia: {ball_body_prop.inertia}, mass: {ball_body_prop.mass}")

        # Acquire State Tensors
        print("acquire observations")
        tensor_rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        tensor_dof_state = self.gym.acquire_dof_state_tensor(self.sim)
        tensor_actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        tensor_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # Build PD Control Constraints
        dof_prop = self.gym.get_actor_dof_properties(self.env, self.dict_character_actor_handles["humanoid"][0])
        dof_limits_lower = []
        dof_limits_upper = []

        for j in range(self.dict_actor_num_dofs["humanoid"]):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                dof_limits_lower.append(dof_prop['upper'][j])
                dof_limits_upper.append(dof_prop['lower'][j])
            else:
                dof_limits_lower.append(dof_prop['lower'][j])
                dof_limits_upper.append(dof_prop['upper'][j])

        dof_limits_lower = to_torch(dof_limits_lower, device=self.device)
        dof_limits_upper = to_torch(dof_limits_upper, device=self.device)
        dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        num_joints = len(dof_offsets) - 1
                
        lim_low = dof_limits_lower.cpu().numpy()
        lim_high = dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = dof_offsets[j]
            dof_size = dof_offsets[j + 1] - dof_offsets[j]

            if dof_size == 3:
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = to_torch(0.5 * (lim_high + lim_low), device=self.device)
        self._pd_action_scale = to_torch(0.5 * (lim_high - lim_low), device=self.device)

        self.gym.prepare_sim(self.sim) # NOTE: after prepare_sim, must not acquire tensor from sim
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # Slice State Tensors
        self.state_contact_forces = gymtorch.wrap_tensor(tensor_contact_forces)
        self.state_rigid_body = gymtorch.wrap_tensor(tensor_rigid_body_state).view(-1, 13)
        self.state_dof = gymtorch.wrap_tensor(tensor_dof_state).view(-1, 2)
        self.state_root = gymtorch.wrap_tensor(tensor_actor_root_state).view(-1, 13)
        self.state_root_humanoid = self.state_root[self.dict_character_actor_idx_offset["humanoid"]:self.dict_character_actor_idx_offset["humanoid"]+self.num_character]
        self.state_root_ball = self.state_root[self.dict_global_actor_idx["ball"]]
        self.state_my_character_pos = self.state_root_humanoid[self.dict_character_actor_idx_offset["humanoid"]+self.my_character_idx, 0:3]

        self.state_humanoid_body = self.state_rigid_body[self.dict_actor_body_idx_offset["humanoid"]:self.dict_actor_body_idx_offset["humanoid"]+self.num_character * self.dict_actor_num_bodies["humanoid"]].view(self.num_character, -1, 13)
        self.humanoid_indices = torch.tensor(self.dict_character_actor_handles["humanoid"], dtype=torch.int32, device=self.device)
        
        if self.viz_marker:
            self.state_move_dir_marker = self.state_root[self.dict_character_actor_idx_offset["marker_move_dir"]:self.dict_character_actor_idx_offset["marker_move_dir"]+self.num_character]
            self.state_facing_dir_marker = self.state_root[self.dict_character_actor_idx_offset["marker_facing_dir"]:self.dict_character_actor_idx_offset["marker_facing_dir"]+self.num_character]
            self.state_skill_indicator = self.state_root[self.dict_character_actor_idx_offset["marker_skill_state"]:self.dict_character_actor_idx_offset["marker_skill_state"]+self.num_character]
            self.state_selection_marker = self.state_root[self.dict_global_actor_idx["marker_selection"]]

            self.state_move_dir_marker_pos = self.state_move_dir_marker[..., 0:3]
            self.state_move_dir_marker_rot = self.state_move_dir_marker[..., 3:7]
            self.state_dof_move_dir_marker = self.state_dof[self.dict_actor_dof_idx_offset["marker_move_dir"]:self.dict_actor_dof_idx_offset["marker_move_dir"] + self.num_character * self.dict_actor_num_dofs["marker_move_dir"]].view(self.num_character, -1, 2)
            
            self.state_facing_dir_marker_pos = self.state_facing_dir_marker[..., 0:3]
            self.state_facing_dir_marker_rot = self.state_facing_dir_marker[..., 3:7]
            self.state_skill_indicator_pos = self.state_skill_indicator[..., 0:3]
            self.state_selection_marker_pos = self.state_selection_marker[..., 0:3]

            self.state_dof_kick_marker = self.state_dof[self.dict_actor_dof_idx_offset["marker_kick_traj"]:self.dict_actor_dof_idx_offset["marker_kick_traj"]+self.dict_actor_num_dofs["marker_kick_traj"]].view(-1, 2, 2) # dof, x|z, pos|vel
            self.state_kick_marker = self.state_root[self.dict_global_actor_idx["marker_kick_traj"]]
            self.state_kick_marker_pos = self.state_kick_marker[..., 0:3]
            self.state_kick_marker_rot = self.state_kick_marker[..., 3:7]
            self.memory_dof_kick_marker = self.state_dof_kick_marker.clone()

            self.marker_ids = torch.tensor(self.marker_handles, dtype=torch.int32, device=self.device)
            self.marker_kick_timer = 0.0
            self.kick_failed = torch.zeros(self.num_character, dtype=torch.bool, device=self.device)
            self.kick_success = torch.zeros(self.num_character, dtype=torch.bool, device=self.device)
            

        self.controller_network = ControllerNetwork(batch_size=self.num_character, device=self.device)

        # Target tensors
        self.target_move_speed = None
        self.target_move_dir = None
        self.target_facing_dir = None
        
        self.target_kick_speed = None
        self.target_kick_dir = None

        self.target_dribble_speed = None
        self.target_dribble_dir = None
        
        self.target_ball_fall_pos = None

        self.fsm = FootballFSM(num_character=self.num_character, device=self.device)

        self.init_state_dof = self.state_dof.clone()
        self.init_state_root = self.state_root.clone()

        self.setup_target_tensors()

        self.task_obs = torch.zeros_like(compute_move_observations(
            root_states=self.state_root_humanoid,
            target_dir=self.target_move_dir,
            target_speed=self.target_move_speed,
            target_face_dir=self.target_facing_dir
        ))

    def setup_target_tensors(self) -> None:
        self.target_move_speed = 0.0 * self.max_target_move_speed * torch.ones([self.num_character], device=self.device, dtype=torch.float32)
        self.target_move_dir = torch.zeros([self.num_character, 2], device=self.device, dtype=torch.float32)
        self.target_move_dir[0::2, 0] = 1.0
        self.target_move_dir[1::2, 0] = -1.0

        self.target_facing_dir = torch.zeros([self.num_character, 2], device=self.device, dtype=torch.float32)
        self.target_facing_dir[..., 0] = 1.0

        self.target_dribble_speed = 0.3 * self.max_target_dribble_speed * torch.ones([self.num_character], device=self.device, dtype=torch.float32)
        self.target_dribble_dir = torch.zeros([self.num_character, 2], device=self.device, dtype=torch.float32)      
        self.target_dribble_dir[0::2, 0] = 1.0
        self.target_dribble_dir[1::2, 0] = -1.0

        self.target_kick_speed = 0.5 * self.max_target_kick_speed * torch.ones([self.num_character], device=self.device, dtype=torch.float32)
        self.target_kick_dir = torch.zeros([self.num_character, 3], device=self.device, dtype=torch.float32)
        self.target_kick_dir[..., 0] = 1.0

        
        self.target_onehot = torch.zeros(self.num_character, 6, device=self.device)
        self.target_onehot[..., -3::2] = torch.nn.functional.one_hot(torch.randint(0, 2, (self.num_character,), device=self.device), num_classes=2) 

        self.target_ball_fall_pos = self.state_root_humanoid[..., 0:3].clone()

    def register_keyboard_event(self):
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_SPACE, "RESET")
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_1, TRAP)

    def handle_keyboard_event(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value > 0:
                if evt.action == "QUIT":
                    sys.exit()
                elif evt.action == "RESET":
                    self.need_reset = True
                else:
                    self.game_mode = evt.action
                    self.need_reset = True

    def reset(self):
        self.my_character_idx = 0
        self.camera_interpolate_buffer[:] = self.state_root_humanoid[self.my_character_idx, 0:3] #.view(1,3).expand(self.camera_interpolate_step, 3).clone()
        
        self.camera_interpolate_buffer[:, 0] = torch.clamp(self.camera_interpolate_buffer[:, 0], min=-40, max=40)
        self.camera_interpolate_buffer[:, 1] = torch.clamp(self.camera_interpolate_buffer[:, 1], min=-15, max=15)
        
        self.state_dof[:] = self.init_state_dof.clone()
        # DoF가 0인 actor index를 주면 에러남.
        for i in range(self.gym.get_actor_count(self.env)):
            if self.gym.get_actor_dof_count(self.env, i) > 0:
                self.actor_ids_to_update_dof.append(i)
        
        self.state_root[:] = self.init_state_root.clone()
        self.actor_ids_to_update_root.extend(range(self.gym.get_actor_count(self.env)))
        
        if self.game_mode != DYNAMIC:
            self.state_root[self.dict_actor_body_idx_offset["humanoid"]:self.dict_actor_body_idx_offset["humanoid"]+self.num_character, 0] = 100
            self.state_root[self.my_character_idx] = self.init_state_root[self.my_character_idx]
            self.state_root_humanoid[self.my_character_idx, 1] = 0
            self.state_root_ball[..., 1] = 0
        if self.game_mode == KICK:
            self.state_root[8] = self.init_state_root[8]
            other_idx = 8 if self.my_character_idx == 0 else 0
            self.state_root_humanoid[other_idx, 1] = 0
        if self.game_mode == MOVE:
            self.state_root_ball[..., 0] = 100

        self.setup_target_tensors()
        # Reset FSM
        self.fsm.reset_state()

        if self.game_mode == TRAP:
            self._reset_ball_for_trap()

    def compute_observations(self):

        state_ball_expanded = self.state_root_ball.view(-1, 13).expand(self.num_character, 13).clone()
        state_ball_pos = state_ball_expanded[..., 0:3]
        state_ball_linvel = state_ball_expanded[..., 7:10]
        state_ball_angvel = state_ball_expanded[..., 10:13]
        state_humanoid_body_pos = self.state_humanoid_body[..., 0:3].clone()
        state_humanoid_body_rot = self.state_humanoid_body[..., 3:7].clone()
        state_humanoid_body_linvel = self.state_humanoid_body[..., 7:10].clone()
        state_humanoid_body_angvel = self.state_humanoid_body[..., 10:13].clone()
        target_ball_fall_pos = self.target_ball_fall_pos.clone()
        target_skill_id = self.fsm.state
        state_root_humanoid = self.state_root_humanoid.clone()
        target_kick_dir = self.target_kick_dir.clone()
        target_kick_speed = self.target_kick_speed.clone()
        target_dribble_dir = self.target_dribble_dir.clone()
        target_dribble_speed = self.target_dribble_speed.clone()
        target_facing_dir = self.target_facing_dir.clone()
        target_move_dir = self.target_move_dir.clone()
        target_move_speed = self.target_move_speed.clone()
        target_onehot = self.target_onehot.clone()
        # print(target_onehot[self.my_character_idx])

        humanoid_obs = compute_humanoid_observations_max(
            state_humanoid_body_pos,
            state_humanoid_body_rot,
            state_humanoid_body_linvel,
            state_humanoid_body_angvel,
            local_root_obs=True,
            root_height_obs=True
        )
        for skill_id, skill_key in self.enum_skill:
            if skill_key == "move":
                self.task_obs[target_skill_id==skill_id] = compute_move_observations(
                    root_states=state_root_humanoid,
                    target_dir=target_move_dir,
                    target_speed=target_move_speed,
                    target_face_dir=self.target_facing_dir
                )[target_skill_id==skill_id]
            elif skill_key == "dribble":
                self.task_obs[target_skill_id==skill_id] = compute_dribble_observations(
                    root_states=state_root_humanoid,
                    target_dir=target_dribble_dir,
                    target_speed=target_dribble_speed,
                    target_face_dir=target_facing_dir,
                    ball_pos=state_ball_pos,
                    ball_vel=state_ball_linvel,
                    ball_ang_vel=state_ball_angvel
                )[target_skill_id==skill_id]
            elif skill_key == "trap":
                self.task_obs[target_skill_id==skill_id] = compute_heading_observations_simple_onehot(
                    root_states=state_root_humanoid,
                    ball_pos=state_ball_pos,
                    ball_vel=state_ball_linvel,
                    ball_ang_vel=state_ball_angvel,
                    onehot_body_part=target_onehot
                )[target_skill_id==skill_id]
                # self.task_obs[target_skill_id==skill_id] = compute_heading_observations_simple_target_fall_onehot(
                #     root_states=state_root_humanoid,
                #     ball_pos=state_ball_pos,
                #     ball_vel=state_ball_linvel,
                #     ball_ang_vel=state_ball_angvel,
                #     target_fall_pos=target_ball_fall_pos,
                #     onehot_body_part=target_onehot
                # )[target_skill_id==skill_id]
            elif skill_key == "kick":
                self.task_obs[target_skill_id==skill_id] = compute_kick_observations(
                    root_states=state_root_humanoid,
                    ball_states=state_ball_expanded,
                    target_speed=target_kick_speed,
                    target_dir=target_kick_dir
                )[target_skill_id==skill_id]
            else:
                print(f"ERROR! Unknown Skill Name: {skill_key}")
                quit()
        return torch.cat([humanoid_obs, self.task_obs.view(self.num_character, -1)], dim=-1)

    def _reset_ball_for_trap(self):
        min_init_theta = -45 #-180
        max_init_theta = 45  #180
        min_speed = 5
        max_speed = 20
        min_fall_dist = 0.5
        max_fall_dist = 2
        min_fall_theta = -180
        max_fall_theta = 180
        min_phi = 15
        max_phi = 60
        g = 9.81

        root_dir = torch.atan2(self.state_root_humanoid[self.my_character_idx, 8], self.state_root_humanoid[self.my_character_idx, 7])
        root_dir = torch.tensor(root_dir, device=self.device)
        humanoid_xy = self.state_root_humanoid[self.my_character_idx, 0:2].clone()

        # fall_theta = torch.deg2rad(torch.rand(num_envs, device=self.device) * (max_fall_theta - min_fall_theta) + min_fall_theta)
        fall_theta = root_dir + torch.deg2rad(torch.rand(1, device=self.device) * (max_fall_theta - min_fall_theta) + min_fall_theta)
        fall_dist = torch.rand(1, device=self.device) * (max_fall_dist - min_fall_dist) + min_fall_dist
        fall_x = humanoid_xy[0] + torch.cos(fall_theta) * fall_dist
        fall_y = humanoid_xy[1] + torch.sin(fall_theta) * fall_dist


        speed = torch.rand(1, device=self.device) * (max_speed - min_speed) + min_speed

        theta = torch.deg2rad(torch.rand(1, device=self.device) * (max_init_theta - min_init_theta) + min_init_theta)
        phi = torch.deg2rad(torch.rand(1, device=self.device) * (max_phi - min_phi) + min_phi)
        t = 2*speed*torch.sin(phi) / g
        dist = speed*torch.cos(phi) * t

        x = fall_x + dist * torch.cos(theta)
        y = fall_y + dist * torch.sin(theta)
        z = 0.115

        dx = fall_x - x
        dy = fall_y - y
        launch_direc = torch.atan2(dy, dx)

        # speed calibration
        speed = (1 + t*.0) * speed
        vx = torch.cos(phi) * torch.cos(launch_direc) * speed
        vy = torch.cos(phi) * torch.sin(launch_direc) * speed
        vz = torch.sin(phi) * speed

        self.state_root_ball[..., 0] = torch.tensor(x, device=self.device, dtype=torch.float32)
        self.state_root_ball[..., 1] = torch.tensor(y, device=self.device, dtype=torch.float32)
        self.state_root_ball[..., 2] = torch.tensor(z, device=self.device, dtype=torch.float32)
        self.state_root_ball[..., 7] = torch.tensor(vx, device=self.device, dtype=torch.float32)
        self.state_root_ball[..., 8] = torch.tensor(vy, device=self.device, dtype=torch.float32)
        self.state_root_ball[..., 9] = torch.tensor(vz, device=self.device, dtype=torch.float32)

        humanoid_ball_distance, ball_touched, ball_approaching, _ = self.check_ball_touch()
        start_commands = [False for i in range(self.num_character)]
        end_commands = [False for i in range(self.num_character)]
        start_commands[self.my_character_idx] = True
        print(self.fsm.state)
        self.fsm.update(start_commands, end_commands, humanoid_ball_distance, ball_touched, ball_approaching)   #update sm
        print(self.fsm.state, ball_approaching)
        
    def switch_character(self, target_dir:torch.Tensor, target_xy_position_diffs:torch.Tensor):
        in_direction = torch.less_equal(
            torch_utils.get_angle_between(
                target_dir.view(1, 2).expand(self.num_character//2, 2),
                torch.nn.functional.normalize(target_xy_position_diffs, dim=-1)
            ),
            torch.deg2rad(torch.tensor(45., device=self.device))
        )
        in_direction[self.my_character_idx//2] = False
        filtered_dist = torch.where(
            in_direction,
            torch.norm(target_xy_position_diffs, dim=-1),
            torch.tensor(float('inf'), device=self.device)
        )
        if torch.any(in_direction):
            self.my_character_idx = torch.argmin(filtered_dist).item() * 2
            return True
        
        return False

    
    def check_state_and_switch_selected(self):
        target_skill_id = self.fsm.state[::2]
        if torch.any(target_skill_id == self.skill_id("dribble")) or torch.any(target_skill_id == self.skill_id("trap")):
            teammate_ball_distance = torch.norm(self.state_root_humanoid[::2, 0:2] - self.state_root_ball[0:2], dim=-1)
            filtered_distance = torch.where(
                torch.logical_or(target_skill_id == self.skill_id("dribble"),
                                 target_skill_id == self.skill_id("trap")
                ),
                teammate_ball_distance,
                torch.tensor(float('inf'), device=self.device)
            )
            closest_dribble_or_trap_teammate = torch.argmin(filtered_distance).item() * 2
            if self.my_character_idx != closest_dribble_or_trap_teammate:
                self.my_character_idx = closest_dribble_or_trap_teammate
                return True
        
        return False
    

    def compute_kick_targets(self, pass_target_idx:torch.Tensor, angle:torch.Tensor):
        target_position_xy = self.state_root_humanoid[pass_target_idx, 0:2]
        target_velocity_xy = self.state_root_humanoid[pass_target_idx, 7:9]

        target_diff = target_position_xy - self.state_root_ball[0:2]
        target_distance = torch.norm(target_diff)
        # t = torch.sqrt(2.0 * torch.norm(target_diff) * torch.sin(angle) / (9.8 * torch.cos(angle)))
        
        # future_target_pos_xy = target_position_xy + 0.5 * target_velocity_xy * t
        # future_target_diff = future_target_pos_xy - self.state_root_ball[0:2]
        # future_target_distance = torch.norm(future_target_diff)

        target_kick_dir_xy = torch.nn.functional.normalize(target_diff, dim=-1)
        target_kick_speed = 1. * torch.sqrt(9.8*target_distance/(2*torch.sin(angle)*torch.cos(angle)))
        target_kick_dir = torch.cat([
            target_kick_dir_xy * torch.cos(angle), 
            torch.sin(angle).view(-1)
        ])
        return target_kick_dir, target_kick_speed

    def find_proper_pass_target(self, target_dir:torch.Tensor):
        pos_diffs = self.state_root_humanoid[::2, 0:2] - self.state_root_humanoid[self.my_character_idx, 0:2]
        unit_pos_diffs = torch.nn.functional.normalize(pos_diffs, dim=-1)
        angle_diffs = torch_utils.get_angle_between(unit_pos_diffs, target_dir.view(1, 2).expand(self.num_character//2, 2))
        angle_diffs[self.my_character_idx//2] = float('inf')
        # angle_diffs = torch.where(
        #     torch.norm(pos_diffs, dim=-1) < 20.0,
        #     angle_diffs,
        #     torch.ones_like(angle_diffs) * float('inf')
        # )
        target_idx = torch.argmin(angle_diffs).item() * 2
        return target_idx

    def set_command(self, buttons, axes, hat):
        target_skill_id = self.fsm.state
        tensor_axes = torch.tensor(axes, device=self.device, dtype=torch.float32)
        tensor_axes[1::3] = -tensor_axes[1::3] # flip y
        tensor_axes[2::3] = (tensor_axes[2::3] + 1.0) / 2.0 # normalize trigger input

        tensor_l_stick = tensor_axes[0:2]
        tensor_r_stick = tensor_axes[3:5]
        tensor_l_trigger = tensor_axes[2]
        tensor_r_trigger = tensor_axes[5]

        xpad_btn_a, xpad_btn_b, xpad_btn_x, xpad_btn_y, xpad_btn_l, xpad_btn_r = buttons[:6]
        xpad_hat_x, xpad_hat_y = hat
        humanoid_pos = self.state_root_humanoid[:, 0:3].clone()
        ball_pos = self.state_root_ball[0:3].clone()
        ball_vel = self.state_root_ball[7:10].clone()
        
        # Reset
        if self.need_reset:
            self.reset()
            target_skill_id = self.fsm.state
            self.need_reset = False
            
        start_commands = [False for i in range(self.num_character)]
        end_commands = [False for i in range(self.num_character)]
        
        
        clamped_ball_vel = ball_vel[0:2].clone()
        clamped_ball_vel[0] = torch.clamp(clamped_ball_vel[0], min=0.5)
        clamped_ball_vel[1] = torch.clamp(clamped_ball_vel[1], min=-0.5, max=0.5)
        clamped_ball_vel = torch.nn.functional.normalize(clamped_ball_vel, dim=-1)
        clamped_ball_vel = torch.sum(torch.mul(clamped_ball_vel, ball_vel[0:2]), dim=-1) * clamped_ball_vel
        
        # compute formation considering ball position
        clamped_ball_pos_ratio = torch.clamp(ball_pos[0] / 40.0, min=-1, max=1).view(1).expand(self.num_character).clone()
        clamped_ball_pos_ratio[1::2] = -clamped_ball_pos_ratio[1::2]

        target_formation_ratio = 2. * torch.lerp(
            clamped_ball_pos_ratio * self.humanoid_formation_x_ratio, 
            torch.ones_like(self.humanoid_formation_x_ratio), 
            self.humanoid_formation_x_ratio
        ) - 1.0 # (-1:1) values
        target_formation_ratio[1::2] = -target_formation_ratio[1::2]
        
        target_formation_xy = self.humanoid_init_pos_xy.clone()
        target_formation_xy[..., 0] = target_formation_ratio * 45
        self.target_move_dir[:] = target_formation_xy - humanoid_pos[..., 0:2]
        self.target_move_speed[:] = torch.clamp(0.1 * torch.pow(torch.norm(target_formation_xy - humanoid_pos[..., 0:2], dim=-1), exponent=2), max=0.5*self.max_target_move_speed)

        closest_opposite_formation_distance = torch.min(torch.norm(target_formation_xy[1::2, 0:2] - ball_pos[0:2], dim=-1))
        closest_oppsoite_humanoid_distance = torch.min(torch.norm(humanoid_pos[1::2, 0:2] - ball_pos[0:2], dim=-1))
        opposite_following_condition = torch.logical_and(
            torch.less_equal(
                torch.norm(target_formation_xy[1::2, 0:2] - ball_pos[0:2], dim=-1),
                max(15.0, closest_opposite_formation_distance)
            ),
            torch.less_equal(
                torch.norm(humanoid_pos[1::2, 0:2] - ball_pos[0:2], dim=-1),
                max(10.0, closest_oppsoite_humanoid_distance)
            ),
        )
        self.target_move_dir[1::2] = torch.where(
            opposite_following_condition.view(-1, 1).expand(-1, 2),
            -humanoid_pos[1::2, 0:2] + ball_pos[0:2] + clamped_ball_vel,
            self.target_move_dir[1::2]
        )
        self.target_move_speed[1::2] = torch.where(
            opposite_following_condition,
            torch.clamp(0.8 * torch.norm(-humanoid_pos[1::2, 0:2] + ball_pos[0:2] + clamped_ball_vel, dim=-1), min=1.0, max=self.max_target_move_speed),
            self.target_move_speed[1::2]
        )
        self.target_facing_dir[0::2] = ball_pos[0:2] - humanoid_pos[0::2, 0:2]
        self.target_facing_dir[1::2] = torch.where(opposite_following_condition.view(-1, 1).expand(-1, 2), ball_pos[0:2] - humanoid_pos[1::2, 0:2], self.target_move_dir[1::2].clone())
        if self.game_mode != DYNAMIC:
            self.target_move_dir[:, 0] += 100
            
        if self.game_mode == KICK:
            other_player_idx = 8 if self.my_character_idx == 0 else 0
            self.target_move_dir[other_player_idx] = ball_pos[0:2] - humanoid_pos[other_player_idx, 0:2] + torch.tensor([15.0, .0], device=self.device)
            self.target_move_speed[other_player_idx] = 1*torch.norm(self.target_move_dir[other_player_idx], dim=-1)
            # self.target_facing_dir[other_player_idx] = 
        
        if self.game_mode == DYNAMIC:
            # Control Switch (when not in dribble or kick state)
            # if target_skill_id[self.my_character_idx] != self.skill_id("dribble") and target_skill_id[self.my_character_idx] != self.skill_id("kick"):
            if self.checker_xpad_hat_idle and not self.check_state_and_switch_selected():
                if xpad_hat_x != 0 or xpad_hat_y != 0:
                    humanoid_xy_diffs = humanoid_pos[::2, 0:2] - humanoid_pos[self.my_character_idx, 0:2] # ::2 for only teammate
                    temp_id = self.my_character_idx
                    if self.switch_character(torch.tensor(hat, dtype=torch.float, device=self.device), humanoid_xy_diffs):
                        if target_skill_id[temp_id] == self.skill_id("trap"):
                            end_commands[temp_id] = True
                    self.checker_xpad_hat_idle = False
                
            if xpad_hat_x == 0 and xpad_hat_y == 0:
                self.checker_xpad_hat_idle = True

            if xpad_btn_y:
                if target_skill_id[self.my_character_idx] == self.skill_id("trap"):
                    end_commands[self.my_character_idx] = True
                self.my_character_idx = torch.argmin(torch.norm(humanoid_pos[::2, 0:2] - ball_pos[0:2], dim=-1)).item() * 2
                    
        if self.game_mode == DYNAMIC or self.game_mode == KICK or self.game_mode == TRAP:
            ## Button input controls
            # Map B as Dribble->Kick
            if target_skill_id[self.my_character_idx]==self.skill_id("dribble") or target_skill_id[self.my_character_idx]==self.skill_id("kick"):
                if xpad_btn_a or xpad_btn_x or tensor_r_trigger > 0.01:
                    start_commands[self.my_character_idx] = True
            # Map RB as Move->Trap
            if target_skill_id[self.my_character_idx]==self.skill_id("move") or target_skill_id[self.my_character_idx]==self.skill_id("trap"):
                if xpad_btn_r:
                    start_commands[self.my_character_idx] = True
            # LB as Trap->Move or Kick->Dribble
            if xpad_btn_l:
                end_commands[self.my_character_idx] = True
                self.is_ground_passing = False
                self.is_lob_passing = False
                self.is_kickking = False
            
            if self.is_kickking and tensor_r_trigger <= 0.01:
                end_commands[self.my_character_idx] = True
                self.is_ground_passing = False
                self.is_lob_passing = False
                self.is_kickking = False
            
            

            if self.kick_success[self.my_character_idx] and self.is_passing_to >= 0:
                if self.target_kick_dir[self.my_character_idx, -1] <= 0.1:
                    self.target_onehot[..., -3::2] = torch.nn.functional.one_hot(torch.randint(0, 2, (self.num_character,), device=self.device), num_classes=2)
                else:
                    self.target_onehot[:] = torch.nn.functional.one_hot(torch.randint(0, 6, (self.num_character,), device=self.device), num_classes=6)
                self.my_character_idx = self.is_passing_to
                
                start_commands[self.my_character_idx] = True
                self.is_passing_to = -1

            # pass & kick control
            # if tensor_l_trigger > 0.01:     # PASS
            
            if xpad_btn_x:
                self.is_lob_passing = True
                self.is_ground_passing = False
                self.is_kickking = False
            elif xpad_btn_a:
                self.is_lob_passing = False
                self.is_ground_passing = True
                self.is_kickking = False
            elif tensor_r_trigger > 0.01:
                self.is_lob_passing = False
                self.is_ground_passing = False
                self.is_kickking = True
            
            if self.is_lob_passing or self.is_ground_passing:
                pass_target_idx = self.find_proper_pass_target(torch.nn.functional.normalize(tensor_l_stick, dim=-1))
                if self.game_mode == KICK:
                    pass_target_idx = 8 if self.my_character_idx == 0 else 0
                self.target_kick_dir[self.my_character_idx], self.target_kick_speed[self.my_character_idx] = self.compute_kick_targets(
                    pass_target_idx=pass_target_idx, 
                    angle=torch.deg2rad(torch.clamp(45. * torch.norm(tensor_l_stick), min=20, max=45))
                )
                if self.is_ground_passing:
                    self.target_kick_dir[self.my_character_idx, 2] = 0.05
                    self.target_kick_speed[self.my_character_idx] = torch.clamp(self.target_kick_speed[self.my_character_idx], max=20.0)
                self.is_passing_to = pass_target_idx
            
            if self.is_kickking:                           # KICK
                target_kick_dir_xy = torch.nn.functional.normalize(tensor_l_stick, dim=-1)
                target_kick_dir_angle = (self.max_target_kick_angle - self.min_target_kick_angle) * torch.clamp(torch.norm(tensor_l_stick), min=0.0, max=1.0) + self.min_target_kick_angle
                
                self.target_kick_dir[self.my_character_idx] = torch.cat([
                    target_kick_dir_xy * torch.cos(target_kick_dir_angle), 
                    torch.sin(target_kick_dir_angle).view(-1)
                ])
                self.target_kick_speed[self.my_character_idx] = (self.max_target_kick_speed - self.min_target_kick_speed) * torch.clamp(tensor_r_trigger, min=0.0, max=1.0) + self.min_target_kick_speed
                self.is_passing_to = -1
        # =====================if self.game_mode == DYNAMIC: ends here===========================


        ## Joystick input controls
        if torch.norm(tensor_r_stick) < self.threshold_override_r_stick:
            # if r stick is idle, set facing control same as move control
            tensor_r_stick[:] = tensor_l_stick.clone()
        # move control
        # self.target_facing_dir[:] = ball_pos[0:2] - humanoid_pos[..., 0:2]
        self.target_facing_dir[self.my_character_idx] = tensor_r_stick.clone()
        self.target_move_dir[self.my_character_idx] = tensor_l_stick.clone()
        self.target_move_speed[self.my_character_idx] = (self.max_target_move_speed - self.min_target_move_speed) * torch.clamp(torch.norm(tensor_l_stick), 0.0, 1.0) + self.min_target_move_speed

        # dribble control
        self.target_dribble_dir[self.my_character_idx] = tensor_l_stick.clone()
        self.target_dribble_speed[self.my_character_idx] = (self.max_target_dribble_speed - self.min_target_dribble_speed) * torch.clamp(torch.norm(tensor_l_stick), 0.0, 1.0) + self.min_target_dribble_speed

        # normalization
        self.target_ball_fall_pos[:] = self.state_humanoid_body[:, 11, 0:3]
        self.target_ball_fall_pos[..., 0:2] += torch.nn.functional.normalize(tensor_l_stick, dim=-1) * torch.clamp(torch.norm(tensor_l_stick), 0.0, 1.0) * 0.5
        self.target_move_dir = torch.nn.functional.normalize(self.target_move_dir, dim=-1)
        self.target_facing_dir = torch.nn.functional.normalize(self.target_facing_dir, dim=-1)
        self.target_kick_dir = torch.nn.functional.normalize(self.target_kick_dir, dim=-1)
        self.target_move_speed = torch.clamp(self.target_move_speed, min=self.min_target_move_speed, max=self.max_target_move_speed)
        self.target_dribble_speed = torch.clamp(self.target_dribble_speed, min=self.min_target_dribble_speed, max=self.max_target_dribble_speed)
        self.target_kick_speed = torch.clamp(self.target_kick_speed, min=self.min_target_kick_speed, max=self.max_target_kick_speed)
                
        return start_commands, end_commands

    def compute_cam(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        cam_target = torch.mean(self.camera_interpolate_buffer, dim=0)
        cam_target[-1] = 0
        cam_pos = cam_target + torch.tensor([0., -130, 50], device=self.device)
        return gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target)

    def update_markers(self):
        if not self.viz_marker:
            return
        
        forward = torch.zeros_like(self.state_facing_dir_marker_pos)
        forward[..., 0] = 1.0

        #UPDATE MARKER STATES
        self.state_move_dir_marker_pos[..., 0:2] = self.state_root_humanoid[..., 0:2] + 0.8*self.target_move_dir
        self.state_move_dir_marker_pos[..., 2] = 0.0
        target_move_3d = torch.zeros_like(self.state_move_dir_marker_pos)
        target_move_3d[..., 0:2] = self.target_move_dir.clone()
        self.state_move_dir_marker_rot[:] = torch_utils.quat_from_two_vectors(forward, target_move_3d)
        range_tensor = torch.arange(1, self.dict_actor_num_dofs["marker_move_dir"]+1, dtype=torch.float, device=self.device)
        self.state_dof_move_dir_marker[:, :, 0] = (0.3 + 0.3 * self.target_move_speed.view(-1, 1)) * range_tensor.view(1, self.dict_actor_num_dofs["marker_move_dir"]).expand(self.num_character, -1) / (self.dict_actor_num_dofs["marker_move_dir"] + 1)
        self.state_dof_move_dir_marker[:, :, 1] = 0
        self.actor_ids_to_update_dof.extend(self.dict_character_actor_handles["marker_move_dir"])

        self.state_facing_dir_marker_pos[..., 0:2] = self.state_root_humanoid[..., 0:2] + self.target_facing_dir
        self.state_facing_dir_marker_pos[..., 2] = -0.01
        target_facing_3d = torch.zeros_like(self.state_facing_dir_marker_pos)
        target_facing_3d[..., 0:2] = self.target_facing_dir.clone()
        self.state_facing_dir_marker_rot[:] = torch_utils.quat_from_two_vectors(forward, target_facing_3d)

        self.state_skill_indicator_pos[..., 0:2] = self.state_root_humanoid[..., 0:2].clone()
        for i, skill_indicator_handle in enumerate(self.dict_character_actor_handles["marker_skill_state"]):
            self.gym.set_rigid_body_color(
                self.env, 
                skill_indicator_handle, 
                0,
                gymapi.MESH_VISUAL, 
                gymapi.Vec3(*(get_axis_params(1.0, self.fsm.state[i]) if self.fsm.state[i] < 3 else [1.0, 1.0, 0.0]))
            )

        self.state_selection_marker_pos[0:2] = self.state_root_humanoid[self.my_character_idx, 0:2].clone()

        # Update kick traj marker
        # if self.fsm.state[self.my_character_idx] == self.skill_id("dribble") or self.fsm.state[self.my_character_idx] == self.skill_id("kick"):
        if self.fsm.state[self.my_character_idx] == self.skill_id("kick"):
            self.state_kick_marker_pos[:2] = self.state_root_ball[..., 0:2].clone()
            self.state_kick_marker_pos[-1] = 0.11
            
            target_kick_xz_proj = self.target_kick_dir[self.my_character_idx, 0:3].clone()
            target_kick_xz_proj[-1] = 0.0
            normalized_target_kick_xz_proj = torch.nn.functional.normalize(target_kick_xz_proj, dim=-1)
            x_vec = torch.tensor([1., 0., 0.], device=self.device)
            self.state_kick_marker_rot[:] = torch_utils.quat_from_two_vectors(x_vec, normalized_target_kick_xz_proj)

            num_traj_shperes = self.dict_actor_num_dofs["marker_kick_traj"]//2
            for i in range(num_traj_shperes):
                t = float(i+1)/(num_traj_shperes+1) * (self.marker_disable_time/2.0)
                x = self.target_kick_speed[self.my_character_idx] * torch.norm(self.target_kick_dir[self.my_character_idx, 0:2], dim=-1)*t
                z = torch.clamp(self.target_kick_speed[self.my_character_idx] * self.target_kick_dir[self.my_character_idx, -1]*t - 0.5*9.8*t*t, min=0.0)
                self.state_dof_kick_marker[i, 0, 0] = x
                self.state_dof_kick_marker[i, 1, 0] = z
                self.state_dof_kick_marker[i, :, 1] *= 0.0
            
            self.memory_dof_kick_marker[:] = self.state_dof_kick_marker.clone()
            self.marker_kick_timer = 0.0
        else:
            self.marker_kick_timer += self.sim_params.dt
            self.state_dof_kick_marker[:] = self.memory_dof_kick_marker.clone()
            if self.marker_kick_timer > self.marker_disable_time or self.kick_failed[self.my_character_idx]:
                self.state_kick_marker_pos[-1] = -100.0

        self.actor_ids_to_update_dof.append(self.dict_global_actor_handle["marker_kick_traj"])

        self.state_facing_dir_marker_pos[..., -1] = torch.where(
            self.fsm.state == self.skill_id("move"),
            self.state_facing_dir_marker_pos[..., -1],
            -1 * torch.ones_like(self.state_facing_dir_marker_pos[..., -1])
        )

        self.state_move_dir_marker_pos[..., -1] = torch.where(
            torch.logical_or(self.fsm.state == self.skill_id("move"), self.fsm.state == self.skill_id("dribble")),
            self.state_move_dir_marker_pos[..., -1],
            -1 * torch.ones_like(self.state_move_dir_marker_pos[..., -1])
        )
    
    def update_root_state(self):
        '''
        이거 안쓰고 step당 set_state 여러번 호출하면 오류 발생 가능
        '''
        self.actor_ids_to_update_root.extend(self.marker_handles)
        self.actor_ids_to_update_root = list(dict.fromkeys(self.actor_ids_to_update_root)) # marker 추가 및 중복 제거
        self.actor_ids_to_update_root.sort()
        if len(self.actor_ids_to_update_root) > 0:
            self.tensor_actor_ids_root = torch.tensor(self.actor_ids_to_update_root, dtype=torch.int32, device=self.device)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.state_root),
                                            gymtorch.unwrap_tensor(self.tensor_actor_ids_root), len(self.tensor_actor_ids_root))
            self.actor_ids_to_update_root.clear()

    def update_dof_state(self):
        '''
        이거 안쓰고 step당 set_dof 여러번 호출하면 오류 발생 가능
        NOTE: DoF가 0인 actor index를 주면 에러남.
        '''
        # return
        self.actor_ids_to_update_dof = list(dict.fromkeys(self.actor_ids_to_update_dof)) # 중복 제거
        self.actor_ids_to_update_dof.sort()
        if len(self.actor_ids_to_update_dof) > 0:
            self.tensor_actor_ids_dof = torch.tensor(self.actor_ids_to_update_dof, dtype=torch.int32, device=self.device)
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.state_dof),
                                            gymtorch.unwrap_tensor(self.tensor_actor_ids_dof), len(self.tensor_actor_ids_dof))
            self.actor_ids_to_update_dof.clear()

    def check_ball_touch(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        contact_force_xy = torch.norm(self.state_contact_forces[..., 0:2], dim=-1)
        # contact: 1, no contact: 0
        contact_between_body = torch.logical_not(torch.isclose(contact_force_xy, torch.zeros_like(contact_force_xy), atol=5e-5))

        humanoid_contact = contact_between_body[self.dict_actor_body_idx_offset["humanoid"]:self.dict_actor_body_idx_offset["humanoid"]+self.num_character*self.dict_actor_num_bodies["humanoid"]]
        humanoid_contact = humanoid_contact.view(self.num_character, self.dict_actor_num_bodies["humanoid"])
        humanoid_contact = torch.logical_not(torch.prod(torch.logical_not(humanoid_contact), dim=-1))

        ball_contact = contact_between_body[self.dict_actor_body_idx_offset["ball"]]
        # if ball_contact:
        #     print("ball contact!")

        humanoid_ball_contact = torch.logical_and(
            ball_contact,
            humanoid_contact
        )
        # Find out which humanoid is in contact with the ball
        humanoid_body_pos = self.state_humanoid_body[..., 0:3]
        humanoid_body_pos = humanoid_body_pos.view(self.num_character, self.dict_actor_num_bodies["humanoid"], 3)
        ball_pos = self.state_root_ball[0:3].clone()

        # Calculate closest(with ball) body distance value for each humanoid
        humanoid_ball_distance, _ = torch.min(torch.norm(humanoid_body_pos - ball_pos, dim=-1), dim=-1) # get closest body part's distance
        humanoid_is_closest = torch.zeros(self.num_character, dtype=torch.bool, device=self.device)
        humanoid_is_closest[torch.argmin(humanoid_ball_distance, dim=-1).item()] = True
        humanoid_ball_is_close_enough = torch.less(
            humanoid_ball_distance,
            0.3
        )

        if humanoid_ball_contact.any():
            # Mark only one nearest humanoid as contact occurred
            humanoid_ball_contact = torch.logical_and(
                humanoid_ball_contact,
                torch.logical_and(
                    humanoid_is_closest,
                    humanoid_ball_is_close_enough
                )
            )
        
        # Find out ball is aproaching to the character
        humanoid_ball_getting_close = torch.less_equal(
            torch.bmm( 
                torch.unsqueeze(self.state_root_humanoid[..., 7:10] - self.state_root_ball[..., 7:10].clone(), dim=-2),
                torch.unsqueeze(self.state_root_humanoid[..., 0:3] - self.state_root_ball[..., 0:3].clone(), dim=-1)
            ).squeeze(),
            0.0,
        )
        # Update FSM
        
        humanoid_ball_is_closest = torch.zeros_like(humanoid_ball_is_close_enough)
        humanoid_ball_is_closest[self.my_character_idx] = True
        closest_opposite_idx = 1+2*torch.argmin(torch.norm(self.state_root_humanoid[1::2, 0:3] - self.state_root_ball[..., 0:3].clone(), dim=-1))
        humanoid_ball_is_closest[closest_opposite_idx] = True
        
        return humanoid_ball_distance, humanoid_ball_contact, humanoid_ball_getting_close, humanoid_ball_is_closest
        
    

    def run(self):
        # Create viewer using the default camera properties
        cam_props = gymapi.CameraProperties()
        cam_props.height = 1080
        cam_props.width = 1920
        cam_props.horizontal_fov = 15
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
        
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')
        
        self.register_keyboard_event()
        
        cam_pos, cam_target = self.compute_cam()
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        timer = time.time()

        stack_ball_touched = torch.zeros(self.control_frequency_inv, self.num_character, dtype=torch.bool, device=self.device)
        stack_ball_approaching = torch.zeros(self.control_frequency_inv, self.num_character, dtype=torch.bool, device=self.device)
        ball_closest = torch.zeros(self.num_character, dtype=torch.bool, device=self.device)
        last_humanoid_ball_distance = torch.zeros(self.num_character, dtype=torch.float, device=self.device)
        
        progress_buf = 0
        render_freq_counter = 0
        while not self.gym.query_viewer_has_closed(self.viewer):

            actions = self.controller_network.get_action(self.compute_observations(), self.fsm.state)

            pd_target = self.state_dof[..., 0].clone()

            pd_target[self.dict_actor_dof_idx_offset["humanoid"]:self.dict_actor_dof_idx_offset["humanoid"] + self.num_character*self.dict_actor_num_dofs["humanoid"]] \
                = self.action_to_pd_targets(actions).flatten()
            # if self.game_mode != DYNAMIC:
            #     temp_pd = torch.zeros_like(pd_target)
            #     # my character only
            #     temp_pd[self.dict_actor_dof_idx_offset["humanoid"]:self.dict_actor_dof_idx_offset["humanoid"]+self.dict_actor_num_dofs["humanoid"]] \
            #         = pd_target[self.dict_actor_dof_idx_offset["humanoid"]:self.dict_actor_dof_idx_offset["humanoid"]+self.dict_actor_num_dofs["humanoid"]]
            #     if self.game_mode == KICK: # one teammate too
            #         teammate_dof_offset = self.dict_actor_dof_idx_offset["humanoid"]+8*self.dict_actor_dof_idx_offset["humanoid"]
            #         temp_pd[teammate_dof_offset:teammate_dof_offset+self.dict_actor_dof_idx_offset["humanoid"]] \
            #             = pd_target[teammate_dof_offset:teammate_dof_offset+self.dict_actor_dof_idx_offset["humanoid"]]
                
            #     pd_target[self.dict_actor_dof_idx_offset["humanoid"]:self.dict_actor_dof_idx_offset["humanoid"] + self.num_character*self.dict_actor_num_dofs["humanoid"]] \
            #         = temp_pd[self.dict_actor_dof_idx_offset["humanoid"]:self.dict_actor_dof_idx_offset["humanoid"] + self.num_character*self.dict_actor_num_dofs["humanoid"]]
                
                
            self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(pd_target), gymtorch.unwrap_tensor(self.humanoid_indices), len(self.humanoid_indices))
            
            
            start_commands, end_commands = self.set_command(*next(self.joystick_coroutine)) 

            # in controlFrequencyInv steps, any collision & approaching happened, assume it as True
            self.kick_failed[:], self.kick_success[:] = self.fsm.update(
                start_commands, 
                end_commands, 
                last_humanoid_ball_distance, 
                stack_ball_touched.any(dim=0), 
                stack_ball_approaching.any(dim=0),
                ball_closest
                )   #update sm

            # step the physics
            for i in range(self.control_frequency_inv):
                # update marker states
                self.update_markers()
                self.update_root_state()
                self.update_dof_state()

                render_freq_counter += 1
                
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                
                last_humanoid_ball_distance[:], stack_ball_touched[i, :], stack_ball_approaching[i, :], ball_closest[:] = self.check_ball_touch()
                if render_freq_counter == self.render_frequency_inv:

                    progress_buf = (progress_buf + 1)%self.camera_interpolate_step
                    clamped_ball_xy_pos = self.state_root_ball[0:2].clone() if self.game_mode != MOVE else self.state_root_humanoid[self.my_character_idx, 0:2]
                    clamped_ball_xy_pos[0] = torch.clamp(clamped_ball_xy_pos[0], min=-40, max=40)
                    clamped_ball_xy_pos[1] = torch.clamp(clamped_ball_xy_pos[1], min=-15, max=15)
                    self.camera_interpolate_buffer[progress_buf, 0:2] = clamped_ball_xy_pos
                    # Wait for dt to elapse in real time.
                    if self.cam_follow:
                        cam_pos, cam_target = self.compute_cam()
                        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

                    # update the viewer
                    self.gym.step_graphics(self.sim)
                    self.gym.draw_viewer(self.viewer, self.sim, True)

                    render_freq_counter = 0
                    
                self.handle_keyboard_event()

                new_timer = time.time()
                sleep_time = self.sim_params.dt - (new_timer - timer)
                if sleep_time > 0 :
                    time.sleep(sleep_time)
                timer = new_timer
            
        print('Done')

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == '__main__':
    DynamicSwitchingEnv().run()
    # DynamicSwitchingEnv().replay()