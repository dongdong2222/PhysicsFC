import torch

import env.tasks.soccer_hlc.humanoid_dribble as humanoid_dribble
from env.tasks.soccer_hlc.ball_physics import GROUND_EPS
from utils import torch_utils

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

# 무작위 방향에서 오는 공을 원터치로 목표 방향/속력으로 보내는 task.
# HumanoidDribble 을 상속해 공 actor, 공 물리 프로파일, target 샘플링을 그대로 쓰고
# obs / reward / termination / 초기 상태만 바꾼다.
#
#   에피소드      3 s (90 step). 목표는 에피소드 내 고정 (headingChangeSteps > episodeLength)
#   obs (13)      공 위치 3 + 공 속도 3 + 공 각속도 3 + tar_dir 2 + tar_speed 1 + 접촉 1
#   reward        터치 전 거리 / 접촉 중 공 속도 / 터치 끝 공 속도(가중치 최대)
#   termination   터치 끝, 공이 relocate_dist 보다 멀어짐, 시간 초과
#   초기 상태     드리블식 스폰 또는 "좋은 터치로 끝난 에피소드의 마지막 상태" 버퍼

DIST_ERR_SCALE = 2.0

# 공이 바닥이 아닌 휴머노이드 body 에 닿았는지 보려고 쓰는 충돌 캡슐.
# Isaac Gym 에 asset 의 캡슐 치수를 런타임에 조회하는 API 가 없어서 mjcf/amp_humanoid_shoe.xml 에서
# 옮겨 적었다. body local frame 의 (p0, p1, radius) 이고, 구는 p0 == p1 으로 둔다.
# 발은 충돌체가 mesh 라서 MJCF 의 touch box site 로 근사했다 (long local-x 를 축, 나머지 half-extent 평균을 반지름).
_BALL_CONTACT_CAPSULES_LOCAL = {
    "pelvis": [
        ((0.0, 0.0, 0.07), (0.0, 0.0, 0.07), 0.09),
        ((0.0, 0.0, 0.205), (0.0, 0.0, 0.205), 0.07),
    ],
    "torso": [
        ((0.0, 0.0, 0.12), (0.0, 0.0, 0.12), 0.11),
        ((-0.0060125, -0.0457775, 0.2287955), (-0.016835, -0.128177, 0.2376182), 0.045),
        ((-0.0060125, 0.0457775, 0.2287955), (-0.016835, 0.128177, 0.2376182), 0.045),
    ],
    "head": [((0.0, 0.0, 0.175), (0.0, 0.0, 0.175), 0.095)],
    "right_upper_arm": [((0.0, 0.0, -0.05), (0.0, 0.0, -0.23), 0.045)],
    "right_lower_arm": [((0.0, 0.0, -0.0525), (0.0, 0.0, -0.1875), 0.04)],
    "right_hand": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.04)],
    "left_upper_arm": [((0.0, 0.0, -0.05), (0.0, 0.0, -0.23), 0.045)],
    "left_lower_arm": [((0.0, 0.0, -0.0525), (0.0, 0.0, -0.1875), 0.04)],
    "left_hand": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.04)],
    "right_thigh": [((0.0, 0.0, -0.06), (0.0, 0.0, -0.36), 0.055)],
    "right_shin": [((0.0, 0.0, -0.045), (0.0, 0.0, -0.355), 0.05)],
    "right_foot": [((0.135, 0.0, -0.0225), (-0.045, 0.0, -0.0225), 0.05)],
    "left_thigh": [((0.0, 0.0, -0.06), (0.0, 0.0, -0.36), 0.055)],
    "left_shin": [((0.0, 0.0, -0.045), (0.0, 0.0, -0.355), 0.05)],
    "left_foot": [((0.135, 0.0, -0.0225), (-0.045, 0.0, -0.0225), 0.05)],
}
_BALL_CONTACT_FLASH_COLOR = (1.0, 0.0, 0.0)
_BALL_DEFAULT_COLOR = (1.0, 1.0, 1.0)


class HumanoidBallMastery(humanoid_dribble.HumanoidDribble):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env_cfg = cfg["env"]

        self._dist_reward_w = env_cfg["distRewardW"]
        self._contact_reward_w = env_cfg["contactRewardW"]
        self._touch_reward_w = env_cfg["touchRewardW"]
        self._vel_err_scale = env_cfg["velErrScale"]
        self._vel_err_par_w = env_cfg["velErrParW"]
        self._vel_err_tan_w = env_cfg["velErrTanW"]
        self._touch_eval_rolling = env_cfg["touchEvalMode"] == "rolling"
        self._face_shape_w = env_cfg["faceShapeW"]
        self._face_touch_w = env_cfg["faceTouchW"]
        self._tar_facing_range = np.deg2rad(env_cfg["tarFacingRangeDeg"])

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        self._setup_ball_contact_detector(env_cfg)
        self._setup_init_state_buffer(env_cfg)
        return

    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 2 + 1 + 2 + 3 + 3 + 3 + 1 # tar_dir + tar_speed + tar_facing_dir + 공 위치/속도/각속도 + 접촉
        return obs_size

    ####################################################################
    # 접촉 감지
    ####################################################################

    def _setup_ball_contact_detector(self, env_cfg):
        self._ball_contact_force_threshold = float(env_cfg["ball_contact_force_threshold"])
        self._ball_contact_capsule_distance_threshold = float(env_cfg["ball_contact_capsule_distance_threshold"])
        self._ball_contact_velocity_change_threshold = float(env_cfg["ball_contact_velocity_change_threshold"])
        self._ball_radius = self._ball_profile["ball"]["radius"]

        capsule_body_names = list(env_cfg["ball_contact_capsule_body_names"])
        unknown_bodies = [name for name in capsule_body_names if name not in _BALL_CONTACT_CAPSULES_LOCAL]
        if unknown_bodies:
            raise ValueError("ball_contact_capsule_body_names contains {}, which have no entry in "
                             "_BALL_CONTACT_CAPSULES_LOCAL.".format(unknown_bodies))

        body_ids = self._build_contact_body_ids_tensor(capsule_body_names).tolist()
        cap_body_ids, p0, p1, radius = [], [], [], []
        for name, body_id in zip(capsule_body_names, body_ids):
            for cap_p0, cap_p1, cap_r in _BALL_CONTACT_CAPSULES_LOCAL[name]:
                cap_body_ids.append(body_id)
                p0.append(cap_p0)
                p1.append(cap_p1)
                radius.append(cap_r)
        self._contact_capsule_body_ids = torch.tensor(cap_body_ids, device=self.device, dtype=torch.long)
        self._contact_capsule_local_p0 = torch.tensor(p0, device=self.device, dtype=torch.float32)
        self._contact_capsule_local_p1 = torch.tensor(p1, device=self.device, dtype=torch.float32)
        self._contact_capsule_radius = torch.tensor(radius, device=self.device, dtype=torch.float32)

        # 기존 contact force 뷰는 humanoid body 수까지만 잘려 있어 공의 row 가 없다.
        # 같은 GPU 버퍼를 env 당 전체 body 폭으로 다시 wrap 한다 (새 버퍼가 아니라 같은 버퍼의 다른 view).
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        raw_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._contact_force_tensor_full = gymtorch.wrap_tensor(raw_contact_force_tensor).view(
            self.num_envs, bodies_per_env, 3)
        self._ball_contact_body_idx = self.gym.get_actor_rigid_body_index(
            self.envs[0], self._ball_handles[0], 0, gymapi.IndexDomain.DOMAIN_ENV)

        self._ball_non_ground_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._ball_contact_prev = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._touch_end = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._touch_quality = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._prev_vel_quality = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._prev_dist_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._prev_facing = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._ball_contact_color_is_flashed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._ball_contact_force_peak = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._ball_capsule_distance_min = torch.full((self.num_envs,), float("inf"),
                                                     device=self.device, dtype=torch.float32)
        self._ball_vel_xy_peak_delta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        return

    def _physics_step(self):
        # BallPhysicsMixin 의 루프에 서브스텝 피크 추적을 더한 것. control_freq_inv > 1 이면 한 컨트롤 스텝
        # 안에 simulate 가 여러 번 돌아서, 끝에 한 번만 읽으면 서브스텝 안에서 생겼다 사라지는 터치를 놓친다.
        # 첫 simulate 전에는 refresh 하지 않는다 (리셋이 방금 쓴 root state 를 덮어씀, BallPhysicsMixin 참고).
        self._ball_contact_force_peak.zero_()
        self._ball_capsule_distance_min.fill_(float("inf"))
        self._ball_vel_xy_peak_delta.zero_()
        substep_prev_vel_xy = self._ball_states[:, 7:9].clone()

        for i in range(self.control_freq_inv):
            self.render()
            if self._ball_friction is not None:
                self._ball_friction.apply(self._root_states)
            self.gym.simulate(self.sim)

            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            substep_force = torch.norm(self._contact_force_tensor_full[:, self._ball_contact_body_idx, :], dim=-1)
            self._ball_contact_force_peak = torch.maximum(self._ball_contact_force_peak, substep_force)

            substep_capsule_dist = self._compute_ball_capsule_distance()
            self._ball_capsule_distance_min = torch.minimum(self._ball_capsule_distance_min, substep_capsule_dist)

            substep_vel_xy = self._ball_states[:, 7:9]
            substep_delta_v = torch.norm(substep_vel_xy - substep_prev_vel_xy, dim=-1)
            self._ball_vel_xy_peak_delta = torch.maximum(self._ball_vel_xy_peak_delta, substep_delta_v)
            substep_prev_vel_xy = substep_vel_xy.clone()
        return

    def _compute_ball_capsule_distance(self):
        body_pos = self._rigid_body_pos[:, self._contact_capsule_body_ids, :] # (E, C, 3)
        body_rot = self._rigid_body_rot[:, self._contact_capsule_body_ids, :] # (E, C, 4) xyzw
        num_envs, num_bodies = body_pos.shape[0], body_pos.shape[1]

        local_p0 = self._contact_capsule_local_p0.unsqueeze(0).expand(num_envs, -1, -1)
        local_p1 = self._contact_capsule_local_p1.unsqueeze(0).expand(num_envs, -1, -1)
        flat_rot = body_rot.reshape(-1, 4)
        world_p0 = body_pos + quat_rotate(flat_rot, local_p0.reshape(-1, 3)).reshape(num_envs, num_bodies, 3)
        world_p1 = body_pos + quat_rotate(flat_rot, local_p1.reshape(-1, 3)).reshape(num_envs, num_bodies, 3)

        ball_pos = self._ball_states[:, 0:3].unsqueeze(1).expand(-1, num_bodies, -1)
        axis_dist = point_segment_distance(ball_pos, world_p0, world_p1)
        surface_dist = axis_dist - self._contact_capsule_radius.unsqueeze(0) - self._ball_radius
        return surface_dist.min(dim=-1)[0]

    def _update_ball_non_ground_contact(self):
        force_ok = self._ball_contact_force_peak > self._ball_contact_force_threshold
        distance_ok = self._ball_capsule_distance_min <= self._ball_contact_capsule_distance_threshold
        velocity_ok = self._ball_vel_xy_peak_delta >= self._ball_contact_velocity_change_threshold

        # progress_buf == 1 은 리셋 직후 첫 물리 스텝이라, 솔버가 침투를 푸는 동안 힘/속도가 튄다. 터치가 아니다.
        raw_contact = force_ok & distance_ok & velocity_ok
        self._ball_contact_prev[:] = self._ball_non_ground_contact
        self._ball_non_ground_contact = raw_contact & (self.progress_buf != 1)
        self._touch_end[:] = self._ball_contact_prev & ~self._ball_non_ground_contact
        return

    def _update_ball_contact_color(self):
        # set_rigid_body_color 에 batch 버전이 없어 env 수만큼 루프. 상태가 바뀐 env 만 호출.
        if self.headless:
            return
        want_flash = self._ball_non_ground_contact
        for env_id in range(self.num_envs):
            is_flashed = bool(want_flash[env_id].item())
            if is_flashed == bool(self._ball_contact_color_is_flashed[env_id].item()):
                continue
            color = _BALL_CONTACT_FLASH_COLOR if is_flashed else _BALL_DEFAULT_COLOR
            self.gym.set_rigid_body_color(self.envs[env_id], self._ball_handles[env_id], 0,
                                          gymapi.MESH_VISUAL, gymapi.Vec3(*color))
        self._ball_contact_color_is_flashed = want_flash.clone()
        return

    ####################################################################
    # obs / reward / termination
    ####################################################################

    def _compute_observations(self, env_ids=None):
        # 접촉 플래그가 obs / reward / reset 에서 모두 같은 step 의 값이 되도록 여기서 먼저 갱신한다.
        # Humanoid.post_physics_step 이 progress_buf += 1 -> _refresh_sim_tensors -> _compute_observations
        # 순서라, 이 시점이면 progress_buf 와 텐서가 둘 다 최신이다. env_ids 가 주어지는 건 리셋 경로다.
        if env_ids is None:
            self._update_ball_non_ground_contact()
            self._update_ball_contact_color()
        super()._compute_observations(env_ids)
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            tar_facing_dir = self._tar_facing_dir
            ball_pos = self._ball_pos
            ball_vel = self._ball_vel
            ball_ang_vel = self._ball_ang_vel
            contact = self._ball_non_ground_contact
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            tar_facing_dir = self._tar_facing_dir[env_ids]
            ball_pos = self._ball_pos[env_ids]
            ball_vel = self._ball_vel[env_ids]
            ball_ang_vel = self._ball_ang_vel[env_ids]
            contact = self._ball_non_ground_contact[env_ids]

        obs = compute_ball_mastery_observations(root_states, tar_dir, tar_speed, tar_facing_dir,
                                                ball_pos, ball_vel, ball_ang_vel, contact)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        ball_pos = self._ball_states[..., 0:3]
        ball_vel = self._ball_states[..., 7:10]
        ball_ang_vel = self._ball_states[..., 10:13]

        root_rot = self._humanoid_root_states[..., 3:7]

        r_dist, r_contact, r_touch, r_face, touch_quality, vel_quality, dist_quality, facing = \
            compute_ball_mastery_reward(
                root_pos, root_rot, ball_pos, ball_vel, ball_ang_vel,
                self._tar_dir, self._tar_speed, self._tar_facing_dir,
                self._ball_non_ground_contact, self._touch_end,
                self._prev_vel_quality, self._prev_dist_reward, self._prev_facing,
                self.progress_buf == 1, self._ball_radius, self._touch_eval_rolling,
                self._vel_err_scale, self._vel_err_par_w, self._vel_err_tan_w,
                self._dist_reward_w, self._contact_reward_w, self._touch_reward_w,
                self._face_shape_w, self._face_touch_w)

        self.rew_buf[:] = r_dist + r_contact + r_touch + r_face
        self._touch_quality[:] = touch_quality
        self._prev_vel_quality[:] = vel_quality
        self._prev_dist_reward[:] = dist_quality
        self._prev_facing[:] = facing
        self.extras["reward_terms"] = {"dist": r_dist, "contact": r_contact,
                                       "touch": r_touch, "face": r_face}
        return

    def _update_task(self):
        # 상속한 play 모드 분기는 조이스틱을 읽는다. 이 task 는 test 에서도 학습과 같은 랜덤 목표를 쓴다.
        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        self._axes = None # 조이스틱 입력을 쓰지 않는다 (위 주석 참고)
        super()._reset_task(env_ids)

        # 상속한 쪽은 facing 을 tar_dir 과 완전히 독립으로 뽑아, 1/3 이 백힐(각도차 120도 이상)이
        # 되어 원터치로는 사실상 불가능하다. tar_dir 기준 +-tarFacingRangeDeg 안에서만 뽑는다.
        n = len(env_ids)
        tar_theta = torch.atan2(self._tar_dir[env_ids, 1], self._tar_dir[env_ids, 0])
        face_theta = tar_theta + self._tar_facing_range * (2.0 * torch.rand(n, device=self.device) - 1.0)
        self._tar_facing_dir[env_ids] = torch.stack([torch.cos(face_theta), torch.sin(face_theta)], dim=-1)
        return

    def _compute_reset(self):
        # 시간 초과(+ 넘어짐, enableEarlyTermination 로 제어)
        super()._compute_reset()

        ball_dist = torch.norm(self._ball_states[..., :2] - self._humanoid_root_states[..., :2], dim=-1)
        done = ball_dist > self._relocate_dist

        if self._is_play:
            # test 에서는 터치로 끝내지 않고 그 자리에서 새 목표만 다시 뽑는다.
            # "터치 -> 새 목표 -> 다시 터치" 가 끊기지 않고 이어지는 것을 보려는 용도.
            new_tar_env_ids = self._touch_end.nonzero(as_tuple=False).flatten()
            if len(new_tar_env_ids) > 0:
                self._reset_task(new_tar_env_ids)
        else:
            done = done | self._touch_end

        self._terminate_buf[:] = torch.where(done, torch.ones_like(self._terminate_buf), self._terminate_buf)
        self.reset_buf[:] = torch.where(done, torch.ones_like(self.reset_buf), self.reset_buf)

        self._push_init_states(self._touch_end & (self._touch_quality > self._init_buf_quality_min))
        return

    ####################################################################
    # 초기 상태 버퍼
    ####################################################################

    def _setup_init_state_buffer(self, env_cfg):
        size = int(env_cfg["initBufferSize"])
        self._init_buf_size = size
        self._init_buf_dribble_ratio = float(env_cfg["initBufferDribbleRatio"])
        self._init_buf_quality_min = float(env_cfg["initBufferTouchRewardMin"])

        # 월드 좌표는 env 마다 원점이 달라서 그대로 쓰면 다른 env 로 순간이동한다.
        # 캐릭터 루트 xy 를 원점으로 옮겨 저장하고, 복원할 때 그 env 의 기본 스폰 xy 를 더한다.
        self._init_buf_root = torch.zeros(size, 13, device=self.device, dtype=torch.float32)
        self._init_buf_dof_pos = torch.zeros(size, self.num_dof, device=self.device, dtype=torch.float32)
        self._init_buf_dof_vel = torch.zeros(size, self.num_dof, device=self.device, dtype=torch.float32)
        self._init_buf_ball = torch.zeros(size, 13, device=self.device, dtype=torch.float32)
        self._init_buf_head = 0
        self._init_buf_count = 0

        self._init_src_env_ids = torch.zeros(0, device=self.device, dtype=torch.long)
        self._init_src_idx = torch.zeros(0, device=self.device, dtype=torch.long)
        return

    def _push_init_states(self, mask):
        env_ids = mask.nonzero(as_tuple=False).flatten()[:self._init_buf_size]
        n = len(env_ids)
        if n == 0:
            return

        slots = (torch.arange(n, device=self.device) + self._init_buf_head) % self._init_buf_size
        root = self._humanoid_root_states[env_ids].clone()
        ball = self._ball_states[env_ids].clone()
        root_xy = root[:, 0:2].clone()
        root[:, 0:2] = 0.0
        ball[:, 0:2] -= root_xy

        self._init_buf_root[slots] = root
        self._init_buf_dof_pos[slots] = self._dof_pos[env_ids]
        self._init_buf_dof_vel[slots] = self._dof_vel[env_ids]
        self._init_buf_ball[slots] = ball

        self._init_buf_head = (self._init_buf_head + n) % self._init_buf_size
        self._init_buf_count = min(self._init_buf_count + n, self._init_buf_size)
        return

    def _sample_init_source(self, env_ids):
        if self._init_buf_count == 0:
            self._init_src_env_ids = env_ids[:0]
            return
        use_buf = torch.rand(len(env_ids), device=self.device) >= self._init_buf_dribble_ratio
        self._init_src_env_ids = env_ids[use_buf]
        self._init_src_idx = torch.randint(0, self._init_buf_count, (len(self._init_src_env_ids),),
                                           device=self.device)
        return

    def _reset_actors(self, env_ids):
        self._sample_init_source(env_ids)
        # HumanoidDribble._reset_actors 는 IS_INIT_BY_FILE 경로와 매 리셋 print 가 있어서 건너뛴다.
        super(humanoid_dribble.HumanoidDribble, self)._reset_actors(env_ids)
        self._reset_ball(env_ids)

        self._ball_non_ground_contact[env_ids] = False
        self._ball_contact_prev[env_ids] = False
        self._touch_end[env_ids] = False
        return

    def _reset_default(self, env_ids):
        super()._reset_default(env_ids)
        ids = self._init_src_env_ids
        if len(ids) > 0:
            self._humanoid_root_states[ids] = self._init_buf_root[self._init_src_idx]
            self._humanoid_root_states[ids, 0:2] = self._initial_humanoid_root_states[ids, 0:2]
            self._dof_pos[ids] = self._init_buf_dof_pos[self._init_src_idx]
            self._dof_vel[ids] = self._init_buf_dof_vel[self._init_src_idx]
        return

    def _reset_ball(self, env_ids):
        super()._reset_ball(env_ids)
        ids = self._init_src_env_ids
        if len(ids) > 0:
            ball = self._init_buf_ball[self._init_src_idx].clone()
            ball[:, 0:2] += self._initial_humanoid_root_states[ids, 0:2]
            self._ball_states[ids] = ball
            self._target_states[ids, :3] = ball[:, 0:3]
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def point_segment_distance(point, seg_a, seg_b):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    ab = seg_b - seg_a
    t = ((point - seg_a) * ab).sum(dim=-1) / ab.pow(2).sum(dim=-1).clamp_min(1e-8)
    t = t.clamp(0.0, 1.0)
    closest = seg_a + t.unsqueeze(-1) * ab
    return torch.norm(point - closest, dim=-1)


#@torch.jit.script
def compute_ball_mastery_observations(root_states, tar_dir, tar_speed, tar_facing_dir,
                                      ball_pos, ball_vel, ball_ang_vel, contact):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)

    tar_face_dir3d = torch.cat([tar_facing_dir, torch.zeros_like(tar_facing_dir[..., 0:1])], dim=-1)
    local_tar_face_dir = quat_rotate(heading_rot, tar_face_dir3d)[..., 0:2]

    # ball - character pos
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    obs = torch.cat([local_tar_dir, tar_speed, local_tar_face_dir,
                     local_ball_pos, local_ball_vel, local_ball_ang_vel,
                     contact.unsqueeze(-1).float()], dim=-1)
    return obs


#@torch.jit.script
def compute_vel_quality(ball_vel_xy, tar_dir, tar_speed, vel_err_scale, par_w, tan_w):
    # type: (Tensor, Tensor, Tensor, float, float, float) -> Tensor
    # 드리블과 달리 오차를 tar_speed 로 나누지 않는다. 비율로 재면 같은 절대 오차라도 목표 속력이
    # 낮을수록 벌점이 폭발해서 저속 명령에 학습 신호가 남지 않는다.
    par_speed = torch.sum(tar_dir * ball_vel_xy, dim=-1)
    par_err = tar_speed - par_speed
    tan_vel = ball_vel_xy - par_speed.unsqueeze(-1) * tar_dir
    tan_err = torch.norm(tan_vel, dim=-1)
    return torch.exp(-vel_err_scale * (par_w * par_err * par_err + tan_w * tan_err * tan_err))


#@torch.jit.script
def compute_rolling_velocity(ball_pos, ball_vel, ball_ang_vel, radius):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    # 미끄러지던 공이 구름으로 전이한 뒤의 속도. 접촉점 기준 각운동량이 보존되므로 마찰계수와 무관하다.
    #   v_roll = (5 v - 2 R (z_hat x w)) / 7
    # 지면에 있는 공에만 유효해서, 떠 있으면 현재 속도를 그대로 쓴다. 전이 후의 굴림 마찰 감속은 무시한다.
    cross = torch.stack([-ball_ang_vel[..., 1], ball_ang_vel[..., 0]], dim=-1)
    v_roll = (5.0 * ball_vel[..., :2] - 2.0 * radius * cross) / 7.0
    on_ground = (ball_pos[..., 2] - radius) < GROUND_EPS
    return torch.where(on_ground.unsqueeze(-1), v_roll, ball_vel[..., :2])


#@torch.jit.script
def compute_ball_mastery_reward(root_pos, root_rot, ball_pos, ball_vel, ball_ang_vel,
                                tar_dir, tar_speed, tar_facing_dir,
                                contact, touch_end, prev_quality, prev_dist, prev_facing,
                                first_step, ball_radius, eval_rolling,
                                vel_err_scale, par_w, tan_w, dist_w, contact_w, touch_w,
                                face_shape_w, face_touch_w):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    # 터치 전: 공까지의 거리만. 공 속도는 아직 에이전트가 만든 값이 아니라 보상에 넣지 않는다.
    dist_err = torch.norm(ball_pos[..., :2] - root_pos[..., :2], dim=-1)
    dist_reward = torch.exp(-DIST_ERR_SCALE * dist_err)

    quality_now = compute_vel_quality(ball_vel[..., :2], tar_dir, tar_speed, vel_err_scale, par_w, tan_w)
    if eval_rolling:
        eval_vel_xy = compute_rolling_velocity(ball_pos, ball_vel, ball_ang_vel, ball_radius)
    else:
        eval_vel_xy = ball_vel[..., :2]
    quality_end = compute_vel_quality(eval_vel_xy, tar_dir, tar_speed, vel_err_scale, par_w, tan_w)

    # dist / 접촉 중 둘 다 "개선량" 을 준다. 합이 각각 (접촉 직전) - (에피소드 시작),
    # q(접촉 끝) - q(접촉 직전) 로 telescoping 되므로, 제자리에 머무르거나 접촉을 끝내지
    # 않고 끄는 쪽이 이득이 되는 일이 없다.
    # first_step 은 prev 가 직전 에피소드 값이라 제외한다 (contact 가 progress_buf==1 에서
    # 꺼지는 것과 같은 이유).
    before_touch = ~(contact | touch_end) & ~first_step
    r_dist = dist_w * (dist_reward - prev_dist) * before_touch.float()
    r_contact = contact_w * (quality_now - prev_quality) * contact.float()
    r_touch = touch_w * quality_end * touch_end.float()

    # facing: 캐릭터 정면과 목표 방향의 cos. dist 와 같은 개선량 + touch 와 같은 종료 보상.
    # q 에 곱하지 않고 더한다 (곱하면 희소 조건 두 개가 AND 로 묶여 초기 학습이 막힌다).
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing = torch.clamp_min(torch.sum(tar_facing_dir * facing_dir[..., 0:2], dim=-1), 0.0)
    r_face = (face_shape_w * (facing - prev_facing) * before_touch.float()
              + face_touch_w * facing * touch_end.float())

    return r_dist, r_contact, r_touch, r_face, quality_end, quality_now, dist_reward, facing
