import math

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp_task as humanoid_amp_task
import env.tasks.soccer_hlc.humanoid_trapping_3 as humanoid_trapping_3
from env.tasks.soccer_hlc.ball_physics import GROUND_EPS, compute_non_ground_impulse
from env.tasks.soccer_hlc.fall_init import FallInitMixin
from utils import torch_utils

# HumanoidTrapping_3 를 고친 트래핑 태스크. 설계와 근거는 trap_fix_plan.md.
#   1. 공: cfg["env"]["ballProfile"] (BallPhysicsMixin, HumanoidTrapping_3 가 적용)
#   2. 리셋: env 마다 AIR_RATIO 확률로 공중볼 / 땅볼
#      공중볼: 착지 범위 min(FALL_DIST_PER_SEC * T, FALL_DIST_CAP), linear damping 해석해 + ½gh 로 발사
#      땅볼: 도착 속도 / 시간에서 순수 구름 닫힌 식으로 발사 (+ 발사각 GROUND_PHI_*), 각속도 = 구름 + 노이즈
#   3. 트래핑 판정: 공중볼 = 공 접촉력, 땅볼 = 공 속도 급변. 판정 후 TRAP_WINDOW_STEPS 뒤 종료
#   4. 실패: 넘어짐, 핸드볼 (HANDBALL_BODIES 접촉력), 공중볼 착지, 땅볼 통과
#   5. 보상: 트래핑 전 PRE_NEAR_W * face * exp(-k d(목표 부위 geom 중심, 공)), 트래핑 후 exp(-k |v_ball - v_root|_xy)
#   6. 관측: 원본 simple_onehot, 목표 부위 one-hot 6 -> 8 (허벅지 추가). OBS_ONEHOT: False 면 one-hot 을 관측에서 뺀다
#      (목표 부위를 모르므로 pre 보상의 거리는 목표 부위 대신 root 와 공의 수평 거리)
#      WINDOW_PRE_W: 트래핑 후 창에서도 pre 보상 (이 가중치로, post 보상에 더함)
#   7. POST_TARGET_VEL 옵션: 리셋 때 트래핑 후 목표 공 속도를 뽑아 관측 (방향 2 + 속력 1) 에 넣고,
#      트래핑 후 보상을 목표 속도와의 차이로 바꾼다: 목표 방향 성분과 수직 성분으로 나눠 수직은 POST_TAN_W 로 가중
#      방향 = (공이 날아온 쪽, -launch_dir) ± POST_TAR_AZ_RANGE, 속력 U(POST_TAR_SPEED_MIN, POST_TAR_SPEED_MAX)
#      POST_TAR_ZERO_PROB 확률로 속력 0: 이 env 는 원래 보상 exp(-k |v_ball - v_root|_xy) 를 쓴다
#      play mode 에서는 왼쪽 스틱으로 조작하고 (스틱을 놓으면 속력 0), 뷰어에서 빨간 marker 로 표시 (캐릭터 옆, 목표 방향, 속력이 클수록 멀리)
#   8. test 전용: FALL_INIT_PROB 확률로 넘어진 자세에서 시작, FALL_GRACE_STEPS 동안 넘어짐 / 핸드볼 종료 없음 (fall_init.py)

# 목표 부위 one-hot 순서 (앞 6 개는 원본과 같음)와 각 body 의 geom 중심 (body 좌표계, amp_humanoid_shoe.xml)
TARGET_BODIES = ["torso", "head", "right_shin", "right_foot", "left_shin", "left_foot", "right_thigh", "left_thigh"]
TARGET_OFFSETS = [[0.0, 0.0, 0.12], [0.0, 0.0, 0.175], [0.0, 0.0, -0.2], [0.045, 0.0, -0.02],
                  [0.0, 0.0, -0.2], [0.045, 0.0, -0.02], [0.0, 0.0, -0.21], [0.0, 0.0, -0.21]]
GROUND_TARGETS = [3, 5]  # 땅볼은 양발만
# 공중볼 착지 / 트래핑 구분용 다리 모양 (body 좌표계, amp_humanoid_shoe.xml)
LEG_CAPSULE_BODIES = ["right_shin", "left_shin", "right_thigh", "left_thigh"]
LEG_CAPSULE_ENDS = [[[0.0, 0.0, -0.045], [0.0, 0.0, -0.355]]] * 2 + [[[0.0, 0.0, -0.06], [0.0, 0.0, -0.36]]] * 2  # 축 양 끝
LEG_CAPSULE_RADIUS = [0.05, 0.05, 0.055, 0.055]
FOOT_CENTER = [0.045, 0.0, -0.02]
FOOT_RADIUS = 0.10  # 중심에서 신발 끝까지 (발끝 약 0.096 m)
NUM_JOINT_TARGETS = 6    # TARGET_POINT = joint 일 때 공중볼 목표는 앞 6 개 (허벅지 제외)
BODY_COLOR = (0.54, 0.85, 0.2)  # humanoid 기본색 (Humanoid._build_env)
TARGET_COLOR = (1.0, 0.1, 0.1)  # 뷰어에서 목표 부위 표시
# 뷰어: 트래핑 후 목표 공 속도 marker (heading_marker.urdf, 길이 0.5 m 막대)
POST_MARKER_DIST = 0.6         # 속력 0 일 때 캐릭터에서 marker 중심까지 [m]
POST_MARKER_SPEED_SCALE = 0.5  # 속력 1 m/s 당 더 멀어지는 거리 [m]
POST_MARKER_HEIGHT = 0.03      # marker 중심 높이 [m] (바닥 위)

# 종료 사유 (_end_reason, 로그용). 0 = 진행 중
END_REASONS = ["success", "window_fail", "fall", "handball", "landed", "passed", "timeout"]
SUCCESS, WINDOW_FAIL, FALL, HANDBALL, LANDED, PASSED, TIMEOUT = range(1, 8)

G = 9.81


class HumanoidTrap_Fix(FallInitMixin, humanoid_trapping_3.HumanoidTrapping_3):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        self._setup_ball_profile(cfg)
        self._setup_fall_init(cfg)

        self._air_ratio = env_cfg["AIR_RATIO"]
        self._air_speed_min = env_cfg["AIR_SPEED_MIN"]
        self._air_speed_max = env_cfg["AIR_SPEED_MAX"]
        self._air_phi_min = math.radians(env_cfg["AIR_PHI_MIN"])
        self._air_phi_max = math.radians(env_cfg["AIR_PHI_MAX"])
        self._air_spin_max = env_cfg["AIR_SPIN_MAX"]
        self._air_init = env_cfg["AIR_INIT"]
        assert self._air_init in ("fix", "trapping"), self._air_init

        self._ground_speed_min = env_cfg["GROUND_ARRIVE_SPEED_MIN"]
        self._ground_speed_max = env_cfg["GROUND_ARRIVE_SPEED_MAX"]
        self._ground_t_min = env_cfg["GROUND_T_MIN"]
        self._ground_t_max = env_cfg["GROUND_T_MAX"]
        self._ground_spin_noise = env_cfg["GROUND_SPIN_NOISE"]
        self._ground_spin_z_max = env_cfg["GROUND_SPIN_Z_MAX"]
        self._ground_phi_min = math.radians(env_cfg["GROUND_PHI_MIN"])
        self._ground_phi_max = math.radians(env_cfg["GROUND_PHI_MAX"])
        self._pass_dist = env_cfg["PASS_DIST"]

        self._fall_dist_per_sec = env_cfg["FALL_DIST_PER_SEC"]
        self._fall_dist_cap = env_cfg["FALL_DIST_CAP"]
        self._fall_lead_time_max = env_cfg["FALL_LEAD_TIME_MAX"]

        self._ball_force_thr = env_cfg["BALL_FORCE_THR"]
        self._trap_impulse_thr = env_cfg["TRAP_IMPULSE_THR"]
        self._trap_min_height = env_cfg["TRAP_MIN_HEIGHT"]  # None 이면 높이를 보지 않음
        self._ground_dv_thr = env_cfg["GROUND_DV_THR"]
        self._trap_near_margin = env_cfg["TRAP_NEAR_MARGIN"]
        self._handball_force_thr = env_cfg["HANDBALL_FORCE_THR"]
        self._trap_window_steps = env_cfg["TRAP_WINDOW_STEPS"]
        self._success_rel_speed = env_cfg["SUCCESS_REL_SPEED"]
        self._success_dist = env_cfg["SUCCESS_DIST"]

        self._target_point = env_cfg["TARGET_POINT"]
        assert self._target_point in ("geom", "joint"), self._target_point
        self._obs_onehot = env_cfg["OBS_ONEHOT"]
        self._pre_near_w = env_cfg["PRE_NEAR_W"]
        self._window_pre_w = env_cfg["WINDOW_PRE_W"]
        self._pre_near_scale = env_cfg["PRE_NEAR_SCALE"]
        self._face_floor = env_cfg["FACE_FLOOR"]
        self._face_err_scale = env_cfg["FACE_ERR_SCALE"]
        self._post_w = env_cfg["POST_W"]
        self._post_vel_scale = env_cfg["POST_VEL_SCALE"]
        self._post_vel_z = env_cfg["POST_VEL_Z"]
        self._post_target_vel = env_cfg["POST_TARGET_VEL"]
        self._post_tar_az_range = math.radians(env_cfg["POST_TAR_AZ_RANGE"])
        self._post_tar_speed_min = env_cfg["POST_TAR_SPEED_MIN"]
        self._post_tar_speed_max = env_cfg["POST_TAR_SPEED_MAX"]
        self._post_tar_zero_prob = env_cfg["POST_TAR_ZERO_PROB"]
        self._post_tan_w = env_cfg["POST_TAN_W"]

        # 공 모델 상수 (trap_fix_plan.md §2.2): 공중 감쇠 c, 땅볼 구름 감속 dv/dt = -(a_r + k v)
        ball = self._ball_profile["ball"]
        mj = self._ball_profile.get("mj_friction") or {}
        self._ball_radius = ball["radius"]
        inertia_ratio = ball["inertia_ratio"] if ball.get("inertia_ratio") is not None else 0.4
        roll_j = 1.0 + inertia_ratio
        self._lin_damp = ball["linear_damping"]
        self._roll_a = mj.get("rolling", 0.0) * G / (self._ball_radius * roll_j)
        self._roll_k = self._lin_damp / roll_j
        # PhysX 기본 마찰 결합 (평균). 바닥이 공에 줄 수 있는 충격량의 마찰 원뿔
        self._ground_mu = 0.5 * (ball["friction"] + self._ball_profile["ground"]["dynamic_friction"])

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        self._substep_dt = self.sim_params.dt / self.sim_params.substeps
        self._target_body_ids = self._build_key_body_ids_tensor(TARGET_BODIES)
        # joint: body 원점 (관절 위치) 까지의 거리, 허벅지는 목표에서 제외. one-hot 크기는 8 그대로
        if self._target_point == "geom":
            self._target_offsets = to_torch(TARGET_OFFSETS, device=self.device)
            self._num_air_targets = len(TARGET_BODIES)
        else:
            self._target_offsets = torch.zeros([len(TARGET_BODIES), 3], device=self.device)
            self._num_air_targets = NUM_JOINT_TARGETS
        self._ground_targets = to_torch(GROUND_TARGETS, device=self.device, dtype=torch.long)
        self._handball_body_ids = self._build_key_body_ids_tensor(env_cfg["HANDBALL_BODIES"])
        self._capsule_body_ids = self._build_key_body_ids_tensor(LEG_CAPSULE_BODIES)
        self._capsule_ends = to_torch(LEG_CAPSULE_ENDS, device=self.device)
        self._capsule_radius = to_torch(LEG_CAPSULE_RADIUS, device=self.device)
        self._foot_body_ids = self._build_key_body_ids_tensor(["right_foot", "left_foot"])
        self._foot_center = to_torch(FOOT_CENTER, device=self.device)

        self._onehot_body_part = torch.zeros([self.num_envs, len(TARGET_BODIES)], device=self.device, dtype=torch.long)
        self._post_tar_dir = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)  # 트래핑 후 목표 공 방향 (world xy)
        self._post_tar_speed = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._joy_post_tar_dir = None  # play mode: 마지막으로 조이스틱이 정한 목표 방향 (스틱을 놓아도 유지)
        self._phase = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)       # 0 트래핑 전, 1 창
        self._window_count = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._end_reason = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._ball_contact_max = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)  # control step 중 공 접촉력 최대
        self._hard_reset_requested = False  # 테스트: buttons[2] 를 누르면 다음 reset 판정에서 전체 env 리셋
        self._hard_reset_held = False
        self._env0_ended = False  # env0 종료 로그를 한 번만 찍기 위한 상태
        self._colored_body = [-1] * self.num_envs  # 뷰어: env 별로 목표 색을 칠해 둔 body (-1 = 없음)

        # 에피소드 길이 (§2.6): 가장 긴 도착 시간 + 트래핑 창 + 여유
        t_air_max = 2.0 * self._air_speed_max * math.sin(self._air_phi_max) / G
        t_ground_max = self._ground_t_max + self._pass_dist / self._ground_speed_min
        episode_time = max(t_air_max, t_ground_max) + self._trap_window_steps * self.dt + env_cfg["EPISODE_MARGIN"]
        self.max_episode_length = int(math.ceil(episode_time / self.dt))
        print(f"[HumanoidTrap_Fix] max_episode_length = {self.max_episode_length} steps "
              f"(air T max {t_air_max:.2f} s, ground {t_ground_max:.2f} s)")
        self._build_fall_states()
        return

    def get_task_obs_size(self):
        if self._enable_task_obs:
            # 공 위치 + 공 속도 + 공 각속도 (+ 목표 부위 one-hot) (+ 트래핑 후 목표 공 방향 + 속력)
            return 3 + 3 + 3 + (len(TARGET_BODIES) if self._obs_onehot else 0) + (3 if self._post_target_vel else 0)
        return 0

    def _compute_task_obs(self, env_ids=None):
        obs = super()._compute_task_obs(env_ids)
        if not self._obs_onehot:  # 원본 관측의 마지막 칸이 one-hot
            obs = obs[..., :-len(TARGET_BODIES)]
        if not self._post_target_vel:
            return obs
        if env_ids is None:
            root_rot, tar_dir, tar_speed = self._humanoid_root_states[:, 3:7], self._post_tar_dir, self._post_tar_speed
        else:
            root_rot = self._humanoid_root_states[env_ids, 3:7]
            tar_dir, tar_speed = self._post_tar_dir[env_ids], self._post_tar_speed[env_ids]
        return torch.cat([obs, compute_post_target_obs(root_rot, tar_dir, tar_speed)], dim=-1)

    def _load_obj_asset(self):
        asset_root = "calm/data/assets/mjcf/"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._marker_asset = self.gym.load_asset(self.sim, asset_root, "heading_marker.urdf", asset_options)

        ## ball (settings from cfg["env"]["ballProfile"])
        self._ball_asset = self._create_ball_asset()

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._sphere_asset = self.gym.load_asset(self.sim, asset_root, "soccer_target.urdf", asset_options)
        return

    def _update_task(self):
        # play mode + POST_TARGET_VEL 일 때만 조이스틱으로 트래핑 후 목표 공 속도를 조작 (모든 env):
        #   왼쪽 스틱 방향 = 목표 방향 (world, kick_fix 와 같은 축. 크기 0.1 미만이면 이전 방향 유지), 크기 = 속력 U(min, max) 사이 비례
        if not (self._is_play and self._post_target_vel and self._axes):
            return
        left_x, left_y = float(self._axes[0]), float(self._axes[1])
        magnitude = min(math.hypot(left_x, left_y), 1.0)
        if magnitude > 0.1:
            az = math.atan2(-left_y, left_x)
            self._joy_post_tar_dir = to_torch([math.cos(az), math.sin(az)], device=self.device)
        if self._joy_post_tar_dir is not None:  # 리셋된 env 의 랜덤 방향도 덮어쓴다
            self._post_tar_dir[:] = self._joy_post_tar_dir
        if magnitude > 0.1:
            self._post_tar_speed[:] =(self._post_tar_speed_max - self._post_tar_speed_min) * magnitude + self._post_tar_speed_min
        else:  # 스틱을 놓으면 속력 0 (캐릭터와 같이 움직이게 하는 보상)
            self._post_tar_speed[:] = 0
        return

    def _update_obj_actors(self):
        # HumanoidTrapping_3 과 같고, POST_TARGET_VEL 이면 heading marker (빨간 막대) 를 캐릭터 옆에 트래핑 후 목표 공 방향으로 세운다.
        # 캐릭터에서 marker 중심까지 거리 = POST_MARKER_DIST + POST_MARKER_SPEED_SCALE * 목표 속력.
        # 같은 step 에 set_actor_root_state_tensor_indexed 를 두 번 부르면 마지막 것만 반영되므로 super() 를 부르지 않고 한 번에 쓴다
        root_xy = self._humanoid_root_states[..., 0:2]
        if self._post_target_vel:
            dist = POST_MARKER_DIST + POST_MARKER_SPEED_SCALE * self._post_tar_speed
            self._marker_pos[..., 0:2] = root_xy + dist.unsqueeze(-1) * self._post_tar_dir
            self._marker_pos[..., 2] = POST_MARKER_HEIGHT
            marker_dir = self._post_tar_dir
        else:
            self._marker_pos[..., 0:2] = root_xy + self._tar_dir
            self._marker_pos[..., 2] = -10.0
            marker_dir = self._tar_dir
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        self._marker_rot[:] = quat_from_angle_axis(torch.atan2(marker_dir[..., 1], marker_dir[..., 0]), heading_axis)

        self._face_marker_pos[..., 2] = -1
        self._target_pos[..., :2] = self._target_fall_pos[..., :2]
        self._target_pos[..., 2] = 0

        marker_ids = torch.cat([self._marker_actor_ids, self._face_marker_actor_ids, self._target_actor_ids, self._ball_actor_ids], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(marker_ids), len(marker_ids))
        return

    def _reset_task(self, env_ids):
        self._phase[env_ids] = 0
        self._window_count[env_ids] = 0
        self._end_reason[env_ids] = 0
        if self._env0_ended and 0 in env_ids:  # env_ids 는 텐서 또는 (HRL 첫 step) 빈 리스트
            self._env0_ended = False
        return

    def set_command(self, buttons, axes, hat):
        super().set_command(buttons, axes, hat)
        # buttons[2] 를 누르는 순간 (0 -> 1) 한 번만 hard reset
        pressed = buttons is not None and len(buttons) > 2 and bool(buttons[2])
        if pressed and not self._hard_reset_held:
            self._hard_reset_requested = True
        self._hard_reset_held = pressed
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_ball_vel[:] = self._ball_vel
        return

    def _physics_step(self):
        # BallPhysicsMixin._physics_step 과 같고, simulate() 마다 공 접촉력을 읽어 최대값을 남긴다.
        # step 끝 값만 보면 control step 중간에 다리에 맞고 튀어 나간 공의 접촉을 놓친다 (substep 안쪽은 여전히 못 읽음).
        self._ball_contact_max.zero_()
        for i in range(self.control_freq_inv):
            if i > 0 and self._ball_friction is not None:
                # 첫 simulate 는 텐서를 그대로 쓴다: reset 이 indexed 로 쓴 root state 를 refresh 가 덮어쓸 수 있다
                self.gym.refresh_actor_root_state_tensor(self.sim)
            self.render()
            if self._ball_friction is not None:
                self._ball_friction.apply(self._root_states)
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self._ball_contact_max = torch.maximum(self._ball_contact_max, torch.norm(self._ball_contact_forces, dim=-1))
        return

    #region reset ball
    def _reset_actors(self, env_ids):
        if humanoid_trapping_3.IS_INIT:
            self.sample = np.random.choice(humanoid_trapping_3.init_dof_pos.shape[0], size=len(env_ids))
        humanoid_amp_task.HumanoidAMPTask._reset_actors(self, env_ids)

        is_air = torch.rand(len(env_ids), device=self.device) < self._air_ratio
        air_ids, ground_ids = env_ids[is_air], env_ids[~is_air]
        if len(air_ids) > 0:
            if self._air_init == "fix":
                self._reset_ball_air(air_ids)
            else:
                self._reset_ball_air_trapping(air_ids)
        if len(ground_ids) > 0:
            self._reset_ball_ground(ground_ids)
        if self._post_target_vel:
            self._reset_post_target(env_ids)

        if self.viewer is not None:
            self._update_target_color(env_ids)
        self._apply_fall_init(env_ids)
        return

    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        self._apply_fall_rigid_bodies()
        return

    def _reset_post_target(self, env_ids):
        # 트래핑 후 목표 공 속도: 방향 = 공이 날아온 쪽 (-launch_dir) ± POST_TAR_AZ_RANGE, 속력 균등. 에피소드 동안 고정
        # POST_TAR_ZERO_PROB 확률로 속력 0 (보상은 v_root 와의 차이)
        n = len(env_ids)
        launch_dir = self._launch_direc[env_ids]
        az = torch.atan2(-launch_dir[:, 1], -launch_dir[:, 0]) \
             + self._post_tar_az_range * (2 * torch.rand(n, device=self.device) - 1)
        self._post_tar_dir[env_ids] = torch.stack([torch.cos(az), torch.sin(az)], dim=-1)
        speed = (self._post_tar_speed_max - self._post_tar_speed_min) * torch.rand(n, device=self.device) + self._post_tar_speed_min
        speed[torch.rand(n, device=self.device) < self._post_tar_zero_prob] = 0
        self._post_tar_speed[env_ids] = speed
        return

    def _update_target_color(self, env_ids):
        # 뷰어 전용: 목표 부위 body 를 빨간색으로, 이전 목표 body 는 humanoid 기본색으로 되돌린다 (env 별 CPU 호출)
        body_ids = self._target_body_ids[torch.argmax(self._onehot_body_part[env_ids], dim=-1)].tolist()
        for env_id, body_id in zip(env_ids.tolist(), body_ids):
            old = self._colored_body[env_id]
            if old == body_id:
                continue
            env_ptr, handle = self.envs[env_id], self.humanoid_handles[env_id]
            if old >= 0:
                self.gym.set_rigid_body_color(env_ptr, handle, old, gymapi.MESH_VISUAL, gymapi.Vec3(*BODY_COLOR))
            self.gym.set_rigid_body_color(env_ptr, handle, body_id, gymapi.MESH_VISUAL, gymapi.Vec3(*TARGET_COLOR))
            self._colored_body[env_id] = body_id
        return

    def _sample_fall_pos(self, env_ids, flight_time):
        # 착지점 = root + root_vel * min(T, FALL_LEAD_TIME_MAX) + U(0, min(FALL_DIST_PER_SEC * T, FALL_DIST_CAP)) * 랜덤 방향 (§2.3)
        n = len(env_ids)
        max_dist = torch.clamp(self._fall_dist_per_sec * flight_time, max=self._fall_dist_cap)
        dist = torch.rand(n, device=self.device) * max_dist
        angle = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        lead_time = torch.clamp(flight_time, max=self._fall_lead_time_max)
        center = self._humanoid_root_states[env_ids, 0:2] + self._humanoid_root_states[env_ids, 7:9] * lead_time.unsqueeze(-1)
        return center + dist.unsqueeze(-1) * torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

    def _set_ball_launch(self, env_ids, fall_xy, launch_dist, launch_dir, vel, ang_vel, is_ground):
        self._ball_states[env_ids, 0:2] = fall_xy - launch_dist.unsqueeze(-1) * launch_dir
        self._ball_states[env_ids, 2] = self._ball_radius
        self._ball_states[env_ids, 7:10] = vel
        self._ball_states[env_ids, 10:13] = ang_vel
        self._launch_direc[env_ids] = launch_dir
        self._fall_pos[env_ids] = fall_xy
        self._target_fall_pos[env_ids, 0:2] = fall_xy  # 화면 표시용 (target marker)
        self._target_fall_pos[env_ids, 2] = 0
        self._target_pos[env_ids, 0:2] = fall_xy
        self._is_ground[env_ids] = is_ground
        return

    def _reset_ball_air(self, env_ids):
        n = len(env_ids)
        speed = (self._air_speed_max - self._air_speed_min) * torch.rand(n, device=self.device) + self._air_speed_min
        phi = (self._air_phi_max - self._air_phi_min) * torch.rand(n, device=self.device) + self._air_phi_min
        flight_time = 2 * speed * torch.sin(phi) / G
        launch_dist = speed * speed * torch.sin(2 * phi) / G

        fall_xy = self._sample_fall_pos(env_ids, flight_time)
        theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        launch_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

        # linear damping c 를 포함해 T 초 뒤 수평 R, 같은 높이에 오는 발사 속도 + semi-implicit Euler 보정 ½gh (§2.3)
        if self._lin_damp > 0:
            f = 1 - torch.exp(-self._lin_damp * flight_time)
            v_h = self._lin_damp * launch_dist / f
            v_z = G * flight_time / f - G / self._lin_damp
        else:
            v_h = launch_dist / flight_time
            v_z = 0.5 * G * flight_time
        v_z = v_z + 0.5 * G * self._substep_dt
        vel = torch.cat([v_h.unsqueeze(-1) * launch_dir, v_z.unsqueeze(-1)], dim=-1)
        ang_vel = self._air_spin_max * (2 * torch.rand([n, 3], device=self.device) - 1)

        self._set_ball_launch(env_ids, fall_xy, launch_dist, launch_dir, vel, ang_vel, False)
        self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(
            torch.randint(0, self._num_air_targets, (n,), device=self.device), num_classes=len(TARGET_BODIES))
        return

    def _reset_ball_air_trapping(self, env_ids):
        # HumanoidTrapping_3._reset_ball_new_2 (far ball 비율 0) 와 같은 방식, 속도 / 각 범위만 AIR_SPEED_*, AIR_PHI_* 를 쓴다.
        # 착지점 / 발사 계산이 진공 포물선 + 경험식 속도 보정 (1 + 0.025 T) 이라 linear damping 이 있으면 착지점보다 조금 앞에 떨어진다.
        # 원본처럼 착지점에 캐릭터 위치를 더하지 않고 (IS_INIT 리셋은 root xy = 0), 각속도는 이전 값을 그대로 둔다.
        # 원본과 다른 점: 목표 부위는 TARGET_POINT 에 따름
        n = len(env_ids)
        speed = (self._air_speed_max - self._air_speed_min) * torch.rand(n, device=self.device) + self._air_speed_min
        max_fall_dist = 2.0 + 2.0 * (speed - self._air_speed_min) / (self._air_speed_max - self._air_speed_min)
        fall_dist = torch.rand(n, device=self.device) * max_fall_dist
        fall_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        fall_xy = 0.5 * self._humanoid_root_states[env_ids, 7:9] \
                  + fall_dist.unsqueeze(-1) * torch.stack([torch.cos(fall_theta), torch.sin(fall_theta)], dim=-1)

        phi = (self._air_phi_max - self._air_phi_min) * torch.rand(n, device=self.device) + self._air_phi_min
        flight_time = 2 * speed * torch.sin(phi) / G
        launch_dist = speed * torch.cos(phi) * flight_time
        theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        launch_dir = -torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # 공은 착지점에서 theta 방향에 놓인다

        speed = (1 + flight_time * 0.025) * speed
        vel = torch.cat([(speed * torch.cos(phi)).unsqueeze(-1) * launch_dir, (speed * torch.sin(phi)).unsqueeze(-1)], dim=-1)

        self._set_ball_launch(env_ids, fall_xy, launch_dist, launch_dir, vel, self._ball_states[env_ids, 10:13].clone(), False)
        self._ball_states[env_ids, 2] = 0.115
        self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(
            torch.randint(0, self._num_air_targets, (n,), device=self.device), num_classes=len(TARGET_BODIES))
        return

    def _reset_ball_ground(self, env_ids):
        n = len(env_ids)
        arrive_speed = (self._ground_speed_max - self._ground_speed_min) * torch.rand(n, device=self.device) + self._ground_speed_min
        arrive_time = (self._ground_t_max - self._ground_t_min) * torch.rand(n, device=self.device) + self._ground_t_min

        # 순수 구름 dv/dt = -(a_r + k v) 의 닫힌 식으로 발사 속도와 거리 (§2.4)
        a, k = self._roll_a, self._roll_k
        if k > 0:
            speed = (arrive_speed + a / k) * torch.exp(k * arrive_time) - a / k
            launch_dist = (speed + a / k) * (1 - torch.exp(-k * arrive_time)) / k - a * arrive_time / k
        else:
            speed = arrive_speed + a * arrive_time
            launch_dist = 0.5 * (speed + arrive_speed) * arrive_time

        fall_xy = self._sample_fall_pos(env_ids, arrive_time)
        theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        launch_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        # 발사각: 수평 속도는 그대로 두고 v_z = speed * tan(phi) 를 더한다 (작은 각이라 도착 시간 / 거리 계산은 구름 식 그대로)
        phi = (self._ground_phi_max - self._ground_phi_min) * torch.rand(n, device=self.device) + self._ground_phi_min
        vel = torch.cat([speed.unsqueeze(-1) * launch_dir, (speed * torch.tan(phi)).unsqueeze(-1)], dim=-1)

        # 구름 각속도 (진행 방향에 수직인 수평축, z x dir) + 수평 노이즈 + 수직축 회전
        roll_axis = torch.stack([-launch_dir[:, 1], launch_dir[:, 0]], dim=-1)
        ang_vel = torch.zeros([n, 3], device=self.device)
        ang_vel[:, 0:2] = (speed / self._ball_radius).unsqueeze(-1) * roll_axis \
                          + self._ground_spin_noise * (2 * torch.rand([n, 2], device=self.device) - 1)
        ang_vel[:, 2] = self._ground_spin_z_max * (2 * torch.rand(n, device=self.device) - 1)

        self._set_ball_launch(env_ids, fall_xy, launch_dist, launch_dir, vel, ang_vel, True)
        part = self._ground_targets[torch.randint(0, len(GROUND_TARGETS), (n,), device=self.device)]
        self._onehot_body_part[env_ids] = torch.nn.functional.one_hot(part, num_classes=len(TARGET_BODIES))
        return
    #endregion

    def _update_trap_state(self):
        # 트래핑 판정과 종료 사유 (§2.5, §2.7, §2.8). 우선순위: 넘어짐 -> 핸드볼 -> 착지 -> 트래핑 -> 통과 -> 창 종료
        root_xy = self._humanoid_root_states[..., 0:2]
        root_vel_xy = self._humanoid_root_states[..., 7:9]
        ball_pos = self._ball_states[..., 0:3]
        ball_vel_xy = self._ball_states[..., 7:9]
        phase0 = self._phase == 0
        started = self.progress_buf >= 2  # 리셋 직후 step 의 접촉 정보는 쓰지 않음

        if self._enable_early_termination:
            has_fallen = compute_has_fallen(self._contact_forces, self._contact_body_ids, self._rigid_body_pos,
                                            self._termination_heights)
        else:
            has_fallen = torch.zeros_like(phase0)
        handball = torch.any(torch.norm(self._contact_forces[:, self._handball_body_ids], dim=-1) > self._handball_force_thr, dim=-1)
        handball = torch.logical_and(handball, started)
        fall_grace = self._in_fall_grace()  # 넘어진 채 시작한 env: 일어나는 동안 바닥 접촉으로 끝내지 않는다
        has_fallen &= ~fall_grace
        handball &= ~fall_grace

        # 착지: 지금 바닥 근처이거나, 직전 상태에서 자유낙하로 이번 step 안에 바닥에 닿았어야 하는 경우
        # (substep 이 2 개라 바닥에 튕겨 GROUND_EPS 위로 다시 올라간 채 step 이 끝날 수 있다).
        # net contact force 로는 바닥과 몸의 접촉을 구분할 수 없어서, 공이 다리 (허벅지, 정강이, 발) 표면 근처면 접촉을
        # 트래핑으로 본다. 이때는 다리가 공을 막았을 수 있으므로 자유낙하 외삽을 쓰지 않는다.
        near_body = compute_leg_gap(self._rigid_body_pos, self._rigid_body_rot, self._capsule_body_ids, self._capsule_ends,
                                    self._capsule_radius, self._foot_body_ids, self._foot_center, ball_pos) \
                    < self._ball_radius + self._trap_near_margin
        free_fall_z = self._prev_ball_pos[..., 2] + self._prev_ball_vel[..., 2] * self.dt - 0.5 * G * self.dt * self.dt
        near_ground = (ball_pos[..., 2] - self._ball_radius < GROUND_EPS) \
                      | (~near_body & (free_fall_z - self._ball_radius < GROUND_EPS))
        contact = self._ball_contact_max > self._ball_force_thr
        # 바닥으로 설명되지 않는 충격량: 접촉력을 못 읽은 substep 안의 짧은 접촉도 속도 변화로 잡는다 (Kick_Fix 와 같은 식)
        ground_possible = torch.min(torch.min(self._prev_ball_pos[..., 2], ball_pos[..., 2]), free_fall_z) - self._ball_radius < GROUND_EPS
        impulse = compute_non_ground_impulse(self._prev_ball_vel, self._ball_states[..., 7:10], ground_possible,
                                             self._ground_mu, self.dt) > self._trap_impulse_thr
        air_trap = phase0 & ~self._is_ground & started & ((contact & (near_body | ~near_ground)) | impulse)
        if self._trap_min_height is not None:  # Trapping_3 의 공 높이 조건: 바닥 바운스 / 발 높이에서 받는 공은 트래핑이 아님
            air_trap = air_trap & (ball_pos[..., 2] > self._trap_min_height)
        landed = phase0 & ~self._is_ground & started & near_ground & ~air_trap
        ground_trap = phase0 & self._is_ground & started \
                      & (torch.norm(ball_vel_xy - self._prev_ball_vel[..., 0:2], dim=-1) > self._ground_dv_thr)
        trapped = air_trap | ground_trap

        ball_speed = torch.norm(ball_vel_xy, dim=-1)
        ahead = torch.sum((ball_pos[..., 0:2] - root_xy) * ball_vel_xy, dim=-1) / ball_speed.clamp_min(1e-6)
        passed = phase0 & self._is_ground & ~trapped & (ball_speed > 0.5) & (ahead > self._pass_dist)

        # 창: 트래핑 step 다음부터 센다
        in_window = self._phase == 1
        self._window_count[in_window] += 1
        self._phase[trapped] = 1
        window_end = in_window & (self._window_count >= self._trap_window_steps)
        success = window_end \
                  & (torch.norm(ball_vel_xy - root_vel_xy, dim=-1) < self._success_rel_speed) \
                  & (torch.norm(ball_pos[..., 0:2] - root_xy, dim=-1) < self._success_dist)

        reason = torch.zeros_like(self._end_reason)
        for mask, code in ((window_end, WINDOW_FAIL), (success, SUCCESS), (passed, PASSED), (landed, LANDED),
                           (handball, HANDBALL), (has_fallen, FALL)):  # 뒤에 오는 것이 우선
            reason[mask] = code
        self._end_reason[:] = reason
        return

    def _compute_reward(self, actions):
        self._update_trap_state()

        r_pre = compute_pre_reward(self._rigid_body_pos, self._rigid_body_rot, self._target_body_ids, self._target_offsets,
                                   self._onehot_body_part, self._ball_states[..., 0:3],
                                   self._humanoid_root_states[..., 3:7], self._launch_direc,
                                   self._pre_near_scale, self._face_floor, self._face_err_scale, not self._obs_onehot)
        r_post = compute_post_reward(self._ball_states[..., 7:10], self._humanoid_root_states[..., 7:9], self._post_tar_dir,
                                     self._post_tar_speed, self._post_target_vel, self._post_vel_z, self._post_tan_w,
                                     self._post_w, self._post_vel_scale)
        pre_mask = (self._phase == 0).float()
        r_window_pre = self._window_pre_w * r_pre * (1 - pre_mask)  # 창에서도 pre 보상
        r_pre = self._pre_near_w * r_pre * pre_mask
        r_post = r_post * (1 - pre_mask)
        self.rew_buf[:] = r_pre + r_window_pre + r_post
        self.extras["reward_terms"] = {"pre": r_pre, "window_pre": r_window_pre, "post": r_post}  # logged by RLGPUAlgoObserver
        return

    def _compute_reset(self):
        ended = self._end_reason > 0
        timeout = ~ended & (self.progress_buf >= self.max_episode_length)
        self._end_reason[timeout] = TIMEOUT
        self._terminate_buf[:] = ended.long()
        self.reset_buf[:] = (ended | timeout).long()
        if self._hard_reset_requested:
            self.reset_buf[:] = 1
            self._hard_reset_requested = False
        # 끝난 env 의 종료 사유 (episode_stats 는 끝난 env 에서만 평균)
        self.extras["episode_stats"] = {f"end_{name}": (self._end_reason == i + 1).float() for i, name in enumerate(END_REASONS)}
        # env0 종료 사유 출력. 리셋 전에 step 이 더 돌아도 (llc_steps > 1 등) 에피소드당 한 번만 (플래그는 env0 리셋 때 해제)
        if not self._env0_ended and self.reset_buf[0]:
            print(f"[HumanoidTrap_Fix] env0 end: {END_REASONS[self._end_reason[0] - 1]} "
                  f"({'ground' if self._is_ground[0] else 'air'}, step {self.progress_buf[0].item()})")
            self._env0_ended = True
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_has_fallen(contact_buf, contact_body_ids, rigid_body_pos, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    # HumanoidTrapping_3 의 has_fallen 과 같은 식
    masked_contact_buf = contact_buf.clone()
    masked_contact_buf[:, contact_body_ids, :] = 0
    fall_contact = torch.any(torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1), dim=-1)

    fall_height = rigid_body_pos[..., 2] < termination_heights
    fall_height[:, contact_body_ids] = False
    fall_height = torch.any(fall_height, dim=-1)
    return torch.logical_and(fall_contact, fall_height)


#@torch.jit.script
def compute_leg_gap(rb_pos, rb_rot, capsule_ids, capsule_ends, capsule_radius, foot_ids, foot_center, ball_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # 공 중심에서 다리 capsule / 발 (구 근사) 표면까지 거리의 최솟값
    n, k = rb_pos.shape[0], capsule_ids.shape[0]
    cap_pos, cap_rot = rb_pos[:, capsule_ids].reshape(-1, 3), rb_rot[:, capsule_ids].reshape(-1, 4)
    a = cap_pos + quat_rotate(cap_rot, capsule_ends[:, 0].unsqueeze(0).expand(n, k, 3).reshape(-1, 3))
    b = cap_pos + quat_rotate(cap_rot, capsule_ends[:, 1].unsqueeze(0).expand(n, k, 3).reshape(-1, 3))
    p = ball_pos.unsqueeze(1).expand(n, k, 3).reshape(-1, 3)
    ab = b - a
    s = torch.clamp(torch.sum((p - a) * ab, dim=-1) / torch.sum(ab * ab, dim=-1), 0.0, 1.0)
    capsule_gap = torch.norm(p - (a + s.unsqueeze(-1) * ab), dim=-1).view(n, k) - capsule_radius

    k = foot_ids.shape[0]
    foot_pos, foot_rot = rb_pos[:, foot_ids].reshape(-1, 3), rb_rot[:, foot_ids].reshape(-1, 4)
    c = foot_pos + quat_rotate(foot_rot, foot_center.expand(n * k, 3))
    p = ball_pos.unsqueeze(1).expand(n, k, 3).reshape(-1, 3)
    foot_gap = torch.norm(p - c, dim=-1).view(n, k) - FOOT_RADIUS
    return torch.minimum(capsule_gap.min(dim=-1).values, foot_gap.min(dim=-1).values)


#@torch.jit.script
def compute_post_target_obs(root_rot, tar_dir, tar_speed):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # 트래핑 후 목표 공 방향 (heading 기준 xy) + 목표 속력
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)[..., 0:2]
    return torch.cat([local_tar_dir, tar_speed.unsqueeze(-1)], dim=-1)


def compute_post_reward(ball_vel, root_vel_xy, tar_dir, tar_speed, use_target_vel, use_vel_z, tan_w, post_w, vel_scale):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, float, float, float) -> Tensor
    # POST_W * exp(-k |err|). err = v_ball_xy - v_root_xy (목표 속도를 안 쓰거나 목표 속력 0 인 env),
    # 목표 속도를 쓰면 (dribble ball reward 처럼) 목표 방향 성분 오차와 수직 성분으로 나눠 |err|² = e_par² + tan_w * e_perp²
    root_err = ball_vel[..., 0:2] - root_vel_xy
    err_sq = torch.sum(root_err * root_err, dim=-1)
    if use_target_vel:
        par_speed = torch.sum(ball_vel[..., 0:2] * tar_dir, dim=-1)
        perp_vel = ball_vel[..., 0:2] - par_speed.unsqueeze(-1) * tar_dir
        tar_err_sq = (par_speed - tar_speed) ** 2 + tan_w * torch.sum(perp_vel * perp_vel, dim=-1)
        err_sq = torch.where(tar_speed > 0, tar_err_sq, err_sq)
    if use_vel_z:  # 공 z 속도 (절대값, root 기준 아님) 도 포함
        err_sq = err_sq + ball_vel[..., 2] ** 2
    return post_w * torch.exp(-vel_scale * torch.sqrt(err_sq))


def compute_pre_reward(rb_pos, rb_rot, target_body_ids, target_offsets, onehot_body_part, ball_pos, root_rot, launch_dir,
                       near_scale, face_floor, face_err_scale, use_root):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, bool) -> Tensor
    # face * exp(-k |공 - 목표 부위 geom 중심|), face = floor + (1 - floor) exp(-k_f |θ|/π), θ = heading 과 공이 오는 방향의 각도
    # use_root (one-hot 관측 없음): 목표 부위 대신 root 와 공의 수평 거리
    if use_root:
        near_reward = torch.exp(-near_scale * torch.norm(ball_pos[..., 0:2] - rb_pos[:, 0, 0:2], dim=-1))
    else:
        part = torch.argmax(onehot_body_part, dim=-1)
        env_ids = torch.arange(rb_pos.shape[0], device=rb_pos.device)
        body_ids = target_body_ids[part]
        center = rb_pos[env_ids, body_ids] + quat_rotate(rb_rot[env_ids, body_ids], target_offsets[part])
        near_reward = torch.exp(-near_scale * torch.norm(ball_pos - center, dim=-1))

    heading = torch_utils.calc_heading(root_rot)
    face_err = torch.atan2(-launch_dir[..., 1], -launch_dir[..., 0]) - heading
    face_err = torch.atan2(torch.sin(face_err), torch.cos(face_err))
    face = face_floor + (1.0 - face_floor) * torch.exp(-face_err_scale * torch.abs(face_err) / np.pi)
    return face * near_reward
