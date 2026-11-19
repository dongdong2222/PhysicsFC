import torch
from torch.nn.functional import normalize as torch_norm

import env.tasks.soccer_hlc.humanoid_dribble as humanoid_dribble
import env.tasks.soccer_hlc.humanoid_dribble_fix as humanoid_dribble_fix
from utils import torch_utils

from isaacgym.torch_utils import *

# HumanoidDribble_Fix 를 상속한 face 방향 드리블. 바뀐 것만 적는다.
#   1. task obs 에 tar_facing_dir 다시 포함 (원본 HumanoidDribble 과 같은 14)
#   2. target: 리셋 때 캐릭터 -> 공 방향을 face dir, 그 반대를 ball dir 로. 속도는 [tarSpeedMin, tarSpeedMax].
#      에피소드 중에는 [headingChangeStepsMin, Max) step 마다 랜덤 방향을 뽑아 tar dir = face dir 로 두고, 속도도 다시 뽑는다.
#   3. ball 과 face 를 곱으로 묶음: BALL_FACE_REWARD_W * ball * (α + (1 - α) * face), α = FACE_FLOOR
#      face = exp(-k |θ|/π), θ = 캐릭터 heading 과 face dir 의 각도 차이. ball 은 Fix 와 같은 식 (가중치 전).
#   4. dist reward: k = DIST_ERR_SCALE. 공이 캐릭터 heading 기준 앞쪽 절반이면 exp(-k d),
#      뒤쪽 절반이면 -exp(-k d) * (|φ| - π/2) / (π/2) (옆 0 -> 정후방 -exp(-k d))
#      d 는 root 에서 heading 방향으로 DIST_FRONT_OFFSET 앞의 점까지 거리, 앞/뒤(φ) 판정은 root 기준.
#   humanoid reward 와 공 거리 종료(relocate_dist)는 Fix 그대로.

BALL_FACE_REWARD_W = None
FACE_FLOOR = None
FACE_ERR_SCALE = None
DIST_ERR_SCALE = None


class HumanoidDribbleFace(humanoid_dribble_fix.HumanoidDribble_Fix):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        global BALL_FACE_REWARD_W
        global FACE_FLOOR
        global FACE_ERR_SCALE
        global DIST_ERR_SCALE
        BALL_FACE_REWARD_W = cfg["env"]["BALL_FACE_REWARD_W"]
        FACE_FLOOR = cfg["env"]["FACE_FLOOR"]
        FACE_ERR_SCALE = cfg["env"]["FACE_ERR_SCALE"]
        DIST_ERR_SCALE = cfg["env"]["DIST_ERR_SCALE"]

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)
        return

    def get_task_obs_size(self):
        return humanoid_dribble.HumanoidDribble.get_task_obs_size(self)

    def _compute_task_obs(self, env_ids=None):
        return humanoid_dribble.HumanoidDribble._compute_task_obs(self, env_ids)

    def _update_task(self):
        # 에피소드 중 heading change step 마다 랜덤 방향을 tar dir 과 face dir 에 같이 넣고 속도도 다시 뽑는다.
        env_ids = (self.progress_buf >= self._heading_change_steps).nonzero(as_tuple=False).flatten()
        n = len(env_ids)
        if n == 0:
            return

        rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)
        root_pos = self._humanoid_root_states[env_ids, :2]
        ball_pos = self._ball_states[env_ids, :2]

        face_dir = torch_norm(ball_pos - root_pos, dim=-1)
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(n, device=self.device) + self._tar_speed_min
        change_steps = torch.randint(low=self._heading_change_steps_min, high=self._heading_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = -face_dir
        self._tar_facing_dir[env_ids] = face_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        ball_pos = self._ball_states[..., :3]

        r_ball, r_humanoid, _ = humanoid_dribble_fix.compute_heading_reward(root_pos, self._prev_root_pos, root_rot,
                                                                            self._tar_dir, self._tar_speed,
                                                                            ball_pos, self._prev_ball_pos, self.dt)
        # BALL_REWARD_W 는 1.0 이라 r_ball 은 가중치 전 값
        r_ball_face, r_dist = compute_face_dist_reward(root_pos, root_rot, self._tar_facing_dir, ball_pos, r_ball)
        self.rew_buf[:] = r_ball_face + r_humanoid + r_dist
        self.extras["reward_terms"] = {"ball_face": r_ball_face, "humanoid": r_humanoid, "dist": r_dist}  # logged by RLGPUAlgoObserver

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_face_dist_reward(root_pos, root_rot, tar_face_dir, ball_pos, ball_reward):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    heading = torch_utils.calc_heading(root_rot)

    # face reward
    tar_face_theta = torch.atan2(tar_face_dir[..., 1], tar_face_dir[..., 0])
    face_err = tar_face_theta - heading
    face_err = torch.atan2(torch.sin(face_err), torch.cos(face_err)) # [-pi, pi]
    face_reward = torch.exp(-FACE_ERR_SCALE * torch.abs(face_err) / np.pi)
    ball_face_reward = ball_reward * (FACE_FLOOR + (1.0 - FACE_FLOOR) * face_reward)

    # dist reward, 공이 캐릭터 뒤쪽 절반에 있으면 각도에 따라 패널티
    ball_delta = ball_pos[..., :2] - root_pos[..., :2]
    dist_err = torch.norm(ball_pos[..., :2] - humanoid_dribble.compute_dist_anchor(root_pos, root_rot), dim=-1)
    dist_reward = torch.exp(-DIST_ERR_SCALE * dist_err)

    ball_theta = torch.atan2(ball_delta[..., 1], ball_delta[..., 0]) - heading
    ball_theta = torch.abs(torch.atan2(torch.sin(ball_theta), torch.cos(ball_theta))) # [0, pi], 0 = 정면
    back_scale = (ball_theta - 0.5 * np.pi) / (0.5 * np.pi)
    dist_reward = torch.where(ball_theta > 0.5 * np.pi, -back_scale * dist_reward, dist_reward)

    return BALL_FACE_REWARD_W * ball_face_reward, humanoid_dribble.DIST_REWARD_W * dist_reward
