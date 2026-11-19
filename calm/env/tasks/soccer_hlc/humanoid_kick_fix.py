import math
import pickle

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp_task as humanoid_amp_task
import env.tasks.soccer_hlc.humanoid_ball_pos_task_1 as humanoid_ball_pos_task_1
from env.tasks.soccer_hlc.ball_physics import BallPhysicsMixin, GROUND_EPS, compute_non_ground_impulse
from env.tasks.soccer_hlc.fall_init import FallInitMixin
from env.tasks.soccer_hlc.humanoid_trap_fix import compute_has_fallen
from utils import torch_utils

# HumanoidBallPos_1 을 고친 킥 태스크. 설계와 근거는 kick_fix_plan.md.
#   1. 공: cfg["env"]["ballProfile"] (BallPhysicsMixin)
#   2. 리셋: env 마다 INIT_FILE_PROB 확률로 상태 파일 (INIT_FILES 중 균등) 에서, 나머지는 원본 기본 배치
#   3. 목표: 리셋 때 방향 / 속력을 뽑아 에피소드 동안 고정 (리셋하는 env 만)
#      방위각 = (root -> 공 방향) ± TAR_AZ_RANGE, 고각 U(TAR_ELEV_MIN, TAR_ELEV_MAX), 속력 U(TAR_SPEED_MIN, TAR_SPEED_MAX)
#      GROUND_PROB 확률로 땅볼 목표: 고각 0, 속력 U(GROUND_SPEED_MIN, GROUND_SPEED_MAX)
#   4. 킥 판정: 바닥으로 설명되지 않는 공 충격량 > KICK_DV_THR 이고 공 속력 >= KICK_SPEED_MIN. 판정 후 KICK_WINDOW_STEPS 뒤 종료
#   5. 실패: 넘어짐, 핸드볼 (팔 / 손 body 접촉력 > HANDBALL_THR)
#   6. 보상: 원본과 같은 식 exp(-10 |v_ball / tar_speed - tar_dir|^2), 단 킥 판정 step 부터만 (킥 전 0)
#      땅볼 목표는 수평 성분만 exp(-10 |v_xy / tar_speed - tar_dir_xy|^2) x 높이 계수
#      (공 아래쪽 높이 GROUND_EPS 이하 1, GROUND_HEIGHT_MAX 까지 선형으로 0)
#   7. 관측: 원본 + 공 각속도 (10 -> 13)
#   8. test 전용: FALL_INIT_PROB 확률로 넘어진 자세에서 시작, FALL_GRACE_STEPS 동안 넘어짐 / 핸드볼 종료 없음 (fall_init.py)

HANDBALL_BODIES = ["right_upper_arm", "right_lower_arm", "right_hand",
                   "left_upper_arm", "left_lower_arm", "left_hand"]  # 접촉하면 종료하는 body

# 종료 사유 (_end_reason, 로그용). 0 = 진행 중
END_REASONS = ["kick", "fall", "handball", "timeout"]
KICK, FALL, HANDBALL, TIMEOUT = range(1, 5)

G = 9.81
BALL_INDEX = humanoid_ball_pos_task_1.BALL_INDEX
TARGET_INDEX = humanoid_ball_pos_task_1.TARGET_INDEX


class HumanoidKick_Fix(FallInitMixin, BallPhysicsMixin, humanoid_ball_pos_task_1.HumanoidBallPos_1):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]
        self._setup_ball_profile(cfg)
        self._setup_fall_init(cfg)

        self._init_file_prob = env_cfg["INIT_FILE_PROB"]
        self._tar_az_range = math.radians(env_cfg["TAR_AZ_RANGE"])
        self._tar_elev_min = math.radians(env_cfg["TAR_ELEV_MIN"])
        self._tar_elev_max = math.radians(env_cfg["TAR_ELEV_MAX"])
        self._kick_tar_speed_min = env_cfg["TAR_SPEED_MIN"]
        self._kick_tar_speed_max = env_cfg["TAR_SPEED_MAX"]
        self._tar_vis_dist = env_cfg["TAR_VIS_DIST"]
        self._ground_prob = env_cfg["GROUND_PROB"]
        self._ground_speed_min = env_cfg["GROUND_SPEED_MIN"]
        self._ground_speed_max = env_cfg["GROUND_SPEED_MAX"]
        self._ground_height_max = env_cfg["GROUND_HEIGHT_MAX"]

        self._kick_dv_thr = env_cfg["KICK_DV_THR"]
        self._kick_speed_min = env_cfg["KICK_SPEED_MIN"]
        self._kick_window_steps = env_cfg["KICK_WINDOW_STEPS"]
        self._handball_thr = env_cfg["HANDBALL_THR"]

        ball = self._ball_profile["ball"]
        self._ball_radius = ball["radius"]
        # PhysX 기본 마찰 결합 (평균). 바닥이 공에 줄 수 있는 충격량의 마찰 원뿔
        self._ground_mu = 0.5 * (ball["friction"] + self._ball_profile["ground"]["dynamic_friction"])

        init_states = []
        self._rb_reset = []  # [(env_ids, 파일 index, 샘플 index)]: 리셋 직후 refresh 뒤에 덮어쓸 rigid body 상태
        if self._init_file_prob > 0:
            for path in env_cfg["INIT_FILES"]:
                with open(path, "rb") as f:
                    init_states.append(pickle.load(f))

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        # v3 파일 (root_canonical) 은 meta dict 등 배열이 아닌 키가 있고 rb_* 가 없다
        self._init_states = [{k: torch.tensor(np.asarray(v), device=self.device, dtype=torch.float)
                              for k, v in d.items() if isinstance(v, (np.ndarray, list))} for d in init_states]

        self._handball_body_ids = self._build_key_body_ids_tensor(HANDBALL_BODIES)

        self._prev_ball_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_ball_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._ball_impulse = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)  # 바닥 외 충격량 (로그 / 테스트용)
        self._ground_tar = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)  # 땅볼 목표
        self._phase = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)       # 0 킥 전, 1 창
        self._window_count = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._end_reason = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self._env0_ended = False  # env0 종료 로그를 한 번만 찍기 위한 상태
        self._build_fall_states()
        return

    def get_task_obs_size(self):
        if self._enable_task_obs:
            return 3 + 3 + 3 + 1 + 3  # 공 위치 + 공 속도 + 공 각속도 + 목표 속력 + 목표 방향
        return 0

    def _load_target_asset(self):
        ## ball (settings from cfg["env"]["ballProfile"])
        self._target_asset = self._create_ball_asset()

        # target object (표시용)
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._target_asset_2 = self.gym.load_asset(self.sim, "calm/data/assets/mjcf/", "soccer_target.urdf", asset_options)
        return

    def _build_target_tensors(self):
        super()._build_target_tensors()
        self._init_ball_friction(self._target_handles[BALL_INDEX::2], self._tar_actor_ids.view(self.num_envs, 2)[:, BALL_INDEX])
        return

    def _reset_env_tensors(self, env_ids):
        humanoid_amp_task.HumanoidAMPTask._reset_env_tensors(self, env_ids)
        tar_actor_ids = self._tar_actor_ids.view(self.num_envs, 2)[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(tar_actor_ids), len(tar_actor_ids))
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_ball_pos[:] = self._target_states[..., BALL_INDEX, 0:3]
        self._prev_ball_vel[:] = self._target_states[..., BALL_INDEX, 7:10]
        return

    #region reset
    def _reset_actors(self, env_ids):
        humanoid_amp_task.HumanoidAMPTask._reset_actors(self, env_ids)

        from_file = torch.rand(len(env_ids), device=self.device) < self._init_file_prob
        file_ids, default_ids = env_ids[from_file], env_ids[~from_file]
        if len(file_ids) > 0:
            self._reset_from_file(file_ids)
        if len(default_ids) > 0:
            self._reset_ball_default(default_ids)
        self._reset_target(env_ids)
        self._apply_fall_init(env_ids)
        return

    def _reset_from_file(self, env_ids):
        # 파일 상태 (캐릭터 + 공) 를 그대로 쓴다. env 마다 INIT_FILES 중 하나를 균등하게 고른다. rigid body 는 첫 관측용 (원본 IS_INIT 과 같음)
        file_idx = torch.randint(0, len(self._init_states), (len(env_ids),), device=self.device)
        for i, st in enumerate(self._init_states):
            ids = env_ids[file_idx == i]
            if len(ids) == 0:
                continue
            s = torch.randint(0, st["ball_state"].shape[0], (len(ids),), device=self.device)
            self._humanoid_root_states[ids] = st["root_state"][s]
            self._dof_pos[ids] = st["dof_pos"][s]
            self._dof_vel[ids] = st["dof_vel"][s]
            if "rb_pos" in st:  # 없으면 (v3) 기본 리셋처럼 다음 step 에 갱신된다
                self._rb_reset.append((ids, i, s))
            self._target_states[ids, BALL_INDEX] = st["ball_state"][s]
        return

    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        # 리셋 직후 refresh 는 시뮬레이션 전이라 이전 에피소드의 rigid body 상태를 다시 읽어 온다.
        # 그래서 _reset_actors 에서 넣으면 덮여 버리므로 refresh 뒤 (첫 관측 / AMP 관측 계산 전) 에 넣는다
        for ids, i, s in self._rb_reset:
            st = self._init_states[i]
            self._rigid_body_pos[ids] = st["rb_pos"][s]
            self._rigid_body_rot[ids] = st["rb_rot"][s]
            self._rigid_body_vel[ids] = st["rb_vel"][s]
            self._rigid_body_ang_vel[ids] = st["rb_ang_vel"][s]
        self._rb_reset = []
        self._apply_fall_rigid_bodies()
        return

    def _reset_ball_default(self, env_ids):
        # 원본 기본 배치: 캐릭터 앞쪽 반원 (world +x 쪽), 50 % 는 0.5-1 m, 나머지 1-3 m, 정지
        n = len(env_ids)
        init_near = torch.rand(n, device=self.device) < self._near_prob
        dist_max = torch.full([n], self._ball_dist_max, device=self.device)
        dist_max[init_near] = self._near_dist
        dist = (dist_max - self._ball_dist_min) * torch.rand(n, device=self.device) + self._ball_dist_min
        theta = np.pi * torch.rand(n, device=self.device) - 0.5 * np.pi

        root_xy = self._humanoid_root_states[env_ids, 0:2]
        self._target_states[env_ids, BALL_INDEX, 0] = root_xy[:, 0] + dist * torch.cos(theta)
        self._target_states[env_ids, BALL_INDEX, 1] = root_xy[:, 1] + dist * torch.sin(theta)
        self._target_states[env_ids, BALL_INDEX, 2] = self._ball_radius
        self._target_states[env_ids, BALL_INDEX, 7:13] = 0.0
        return

    def _reset_target(self, env_ids):
        # 방위각은 root -> 공 방향 기준 (공에서 잰다), 고각은 각도 자체를 균등하게
        n = len(env_ids)
        ball_pos = self._target_states[env_ids, BALL_INDEX, 0:3]
        to_ball = ball_pos[:, 0:2] - self._humanoid_root_states[env_ids, 0:2]
        az = torch.atan2(to_ball[:, 1], to_ball[:, 0]) + self._tar_az_range * (2 * torch.rand(n, device=self.device) - 1)
        elev = (self._tar_elev_max - self._tar_elev_min) * torch.rand(n, device=self.device) + self._tar_elev_min
        speed = (self._kick_tar_speed_max - self._kick_tar_speed_min) * torch.rand(n, device=self.device) + self._kick_tar_speed_min
        ground = torch.rand(n, device=self.device) < self._ground_prob
        elev[ground] = 0.0
        speed[ground] = (self._ground_speed_max - self._ground_speed_min) * torch.rand(int(ground.sum()), device=self.device) \
                        + self._ground_speed_min
        tar_dir = torch.stack([torch.cos(elev) * torch.cos(az), torch.cos(elev) * torch.sin(az), torch.sin(elev)], dim=-1)

        self._ori_direc[env_ids] = tar_dir
        self._tar_speed[env_ids] = speed
        self._ground_tar[env_ids] = ground
        self._target_states[env_ids, TARGET_INDEX, 0:3] = ball_pos + self._tar_vis_dist * tar_dir
        self._target_states[env_ids, TARGET_INDEX, 7:13] = 0.0

        self._phase[env_ids] = 0
        self._window_count[env_ids] = 0
        self._end_reason[env_ids] = 0
        if self._env0_ended and (env_ids == 0).any():
            self._env0_ended = False
        return

    def _update_task(self):
        # play mode 에서만 조이스틱으로 목표를 조작: 왼쪽 스틱 방향 = 방위각 (world), 크기 = 고각, 오른쪽 트리거 = 속력
        if not (self._is_play and self._axes):
            return
        left_x, left_y, left_trigger, right_x, right_y, right_trigger = self._axes
        magnitude_left = min(float(np.sqrt(left_x * left_x + left_y * left_y)), 1.0)
        trigger = (float(right_trigger) + 1.0) / 2.0

        az = float(np.arctan2(-left_y, left_x))
        elev = (self._tar_elev_max - self._tar_elev_min) * magnitude_left + self._tar_elev_min
        tar_dir = to_torch([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)], device=self.device)
        self._ori_direc[:] = tar_dir
        self._tar_speed[:] = (self._kick_tar_speed_max - self._kick_tar_speed_min) * trigger + self._kick_tar_speed_min
        self._ground_tar[:] = False  # 조이스틱은 뜬 공 목표만
        self._target_states[..., TARGET_INDEX, 0:3] = self._target_states[..., BALL_INDEX, 0:3] + self._tar_vis_dist * tar_dir

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._tar_actor_ids), len(self._tar_actor_ids))
        return
    #endregion

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            ball_states = self._target_states[:, BALL_INDEX]
            tar_speed = self._tar_speed
            tar_dir = self._ori_direc
        else:
            root_states = self._humanoid_root_states[env_ids]
            ball_states = self._target_states[env_ids, BALL_INDEX]
            tar_speed = self._tar_speed[env_ids]
            tar_dir = self._ori_direc[env_ids]
        return compute_kick_observations(root_states, ball_states, tar_speed, tar_dir)

    def _update_kick_state(self):
        # 킥 판정과 종료 사유. 우선순위: 넘어짐 -> 핸드볼 -> 창 종료
        ball_pos = self._target_states[..., BALL_INDEX, 0:3]
        ball_vel = self._target_states[..., BALL_INDEX, 7:10]
        started = self.progress_buf >= 2  # 리셋 직후 step 의 접촉 / 속도 변화는 쓰지 않음

        # 이번 step 안에 바닥에 닿았을 수 있으면 (직전, 지금, 자유낙하 외삽 중 하나라도 바닥 근처) 바닥 충격량을 빼고 본다
        free_fall_z = self._prev_ball_pos[..., 2] + self._prev_ball_vel[..., 2] * self.dt - 0.5 * G * self.dt * self.dt
        lowest_z = torch.minimum(torch.minimum(self._prev_ball_pos[..., 2], ball_pos[..., 2]), free_fall_z)
        near_ground = lowest_z - self._ball_radius < GROUND_EPS
        self._ball_impulse[:] = compute_non_ground_impulse(self._prev_ball_vel, ball_vel, near_ground, self._ground_mu, self.dt)

        touch = started & (self._ball_impulse > self._kick_dv_thr)
        kick = (self._phase == 0) & touch & (torch.norm(ball_vel, dim=-1) >= self._kick_speed_min)

        # 창: 킥 step 다음부터 센다
        in_window = self._phase == 1
        self._window_count[in_window] += 1
        self._phase[kick] = 1
        window_end = in_window & (self._window_count >= self._kick_window_steps)

        if self._enable_early_termination:
            has_fallen = compute_has_fallen(self._contact_forces, self._contact_body_ids, self._rigid_body_pos,
                                            self._termination_heights) & started
        else:
            has_fallen = torch.zeros_like(started)
        handball = started & torch.any(torch.norm(self._contact_forces[:, self._handball_body_ids], dim=-1)
                                       > self._handball_thr, dim=-1)
        fall_grace = self._in_fall_grace()  # 넘어진 채 시작한 env: 일어나는 동안 바닥 접촉으로 끝내지 않는다
        has_fallen &= ~fall_grace
        handball &= ~fall_grace

        reason = torch.zeros_like(self._end_reason)
        for mask, code in ((window_end, KICK), (handball, HANDBALL), (has_fallen, FALL)):  # 뒤에 오는 것이 우선
            reason[mask] = code
        self._end_reason[:] = reason
        return

    def _compute_reward(self, actions):
        self._update_kick_state()
        ball_vel = self._target_states[..., BALL_INDEX, 7:10]
        ball_height = self._target_states[..., BALL_INDEX, 2] - self._ball_radius  # 공 아래쪽 높이
        rew = compute_kick_reward(ball_vel, self._tar_speed, self._ori_direc, self._ground_tar, ball_height,
                                  GROUND_EPS, self._ground_height_max)
        self.rew_buf[:] = torch.where(self._phase == 1, rew, torch.zeros_like(rew))  # 킥 판정 step 부터만
        self.extras["reward_terms"] = {"vel": self.rew_buf.clone()}  # logged by RLGPUAlgoObserver
        return

    def _compute_reset(self):
        ended = self._end_reason > 0
        timeout = ~ended & (self.progress_buf >= self.max_episode_length)
        self._end_reason[timeout] = TIMEOUT
        self._terminate_buf[:] = ended.long()
        self.reset_buf[:] = (ended | timeout).long()
        # 끝난 env 의 종료 사유 (episode_stats 는 끝난 env 에서만 평균)
        self.extras["episode_stats"] = {f"end_{name}": (self._end_reason == i + 1).float() for i, name in enumerate(END_REASONS)}
        # play mode: env0 종료 때 목표와 공 속도 비교 출력. 에피소드당 한 번만 (플래그는 env0 리셋 때 해제)
        if self._is_play and not self._env0_ended and self.reset_buf[0]:
            self._print_env0_end()
            self._env0_ended = True
        return

    def _print_env0_end(self):
        # 종료 step 의 공 속도 (world) vs 목표. 방위각 / 고각 [deg], 사이각 = 공 속도와 목표 방향 사이 각
        v = self._target_states[0, BALL_INDEX, 7:10].tolist()
        d = self._ori_direc[0].tolist()
        speed = math.sqrt(sum(x * x for x in v))
        az = lambda u: math.degrees(math.atan2(u[1], u[0]))
        el = lambda u: math.degrees(math.atan2(u[2], math.hypot(u[0], u[1])))
        cos = sum(a * b for a, b in zip(v, d)) / max(speed, 1e-6)
        d_az = (az(v) - az(d) + 180.0) % 360.0 - 180.0
        print(f"[HumanoidKick_Fix] env0 end: {END_REASONS[self._end_reason[0] - 1]} (step {self.progress_buf[0].item()}) | "
              f"speed {speed:.1f} / tar {self._tar_speed[0].item():.1f} m/s | "
              f"az err {d_az:+.1f}, elev {el(v):.1f} / tar {el(d):.1f} deg | "
              f"angle {math.degrees(math.acos(max(-1.0, min(1.0, cos)))):.1f} deg | rew {self.rew_buf[0].item():.3f}")
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_kick_observations(root_states, ball_states, tar_speed, tar_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    # 원본 compute_shoot_observations + 공 각속도 (heading 기준)
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_ball_pos = ball_states[:, 0:3] - root_pos
    local_ball_pos[..., -1] = ball_states[:, 2]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_states[:, 7:10])
    local_ball_ang_vel = quat_rotate(heading_rot, ball_states[:, 10:13])
    local_tar_dir = quat_rotate(heading_rot, tar_dir)
    return torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, tar_speed.unsqueeze(-1), local_tar_dir], dim=-1)


#@torch.jit.script
def compute_kick_reward(ball_vel, tar_speed, tar_dir, ground_tar, ball_height, height_free, height_max):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    # 뜬 공: 원본 compute_shoot_reward 에서 쓰던 항만 (ball_vel_reward_w = 1, 나머지 가중치 0)
    # 땅볼: 같은 식을 수평 성분만 x 높이 계수 (공 아래쪽 높이 height_free 이하 1, height_max 까지 선형으로 0)
    err = ball_vel / tar_speed.unsqueeze(1) - tar_dir
    air_rew = torch.exp(-10 * torch.pow(err, 2).sum(dim=-1))
    height_factor = torch.clamp(1 - (ball_height - height_free) / (height_max - height_free), 0.0, 1.0)
    ground_rew = torch.exp(-10 * torch.pow(err[:, 0:2], 2).sum(dim=-1)) * height_factor
    return torch.where(ground_tar, ground_rew, air_rew)
