import pickle

import numpy as np
import torch
from torch.nn.functional import normalize as torch_norm

import env.tasks.soccer_hlc.humanoid_dribble as humanoid_dribble
from env.tasks.soccer_hlc.fall_init import FallInitMixin
from utils import torch_utils

from isaacgym.torch_utils import *

# HumanoidDribble 에서 아래 세 가지만 바꾼 버전. 나머지는 전부 상속.
#   1. (되돌림) tangent speed 는 원본과 같이 성분 합
#   2. task obs 에서 tar_facing_dir 제외 (보상 항이 없어 학습 신호가 없었음) -> 14 -> 12
#   3. dist reward 의 pos_err_scale 10.0 -> 2.0 (1m 이내에서 의미가 생기도록)
#   4. 공이 relocate_dist 보다 멀어지면 early termination
#   5. 리셋 때 env 마다 INIT_FILE_PROB 확률로 INIT_FILES (합쳐서 균등 샘플) 의 캐릭터 + 공 상태에서 시작, 나머지는 원본 기본 리셋
#   6. test 전용: FALL_INIT_PROB 확률로 넘어진 자세에서 시작 (fall_init.py)
#   (dist reward 기준점은 원본과 같이 compute_dist_anchor: root + DIST_FRONT_OFFSET * heading 방향)


class HumanoidDribble_Fix(FallInitMixin, humanoid_dribble.HumanoidDribble):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # 키가 없으면 (dribble_face 등) 파일을 쓰지 않는다
        self._init_file_prob = cfg["env"].get("INIT_FILE_PROB", 0.0)
        init_files = cfg["env"].get("INIT_FILES", []) if self._init_file_prob > 0 else []
        self._init_states = None  # super().__init__ 중 리셋은 기본 리셋만
        self._rb_reset = None      # (env_ids, 샘플 index): 리셋 직후 refresh 뒤에 덮어쓸 rigid body 상태
        self._setup_fall_init(cfg)
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                         device_type=device_type, device_id=device_id, headless=headless)

        # v2 파일 (root_canonical): root 는 원점, heading +x. rb_* 는 모든 파일에 있을 때만 쓴다 (없으면 다음 step 에 갱신)
        keys = ["root_state", "dof_pos", "dof_vel", "ball_state"]
        states = [pickle.load(open(f, "rb")) for f in init_files]
        if states and all("rb_pos" in st for st in states):
            keys += ["rb_pos", "rb_rot", "rb_vel", "rb_ang_vel"]
        self._init_states = {k: torch.tensor(np.concatenate([st[k] for st in states]), device=self.device, dtype=torch.float)
                             for k in keys} if states else None
        self._build_fall_states()
        return


    def get_task_obs_size(self):
        obs_size = 0
        if self._enable_task_obs:
            obs_size = 3 + 3 + 3 + 3 # move + 공 위치 + 공 속력 + (ang vel)
        return obs_size

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            ball_pos = self._ball_pos
            ball_vel = self._ball_vel
            ball_ang_vel = self._ball_ang_vel
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            ball_pos = self._ball_pos[env_ids]
            ball_vel = self._ball_vel[env_ids]
            ball_ang_vel = self._ball_ang_vel[env_ids]

        obs = compute_heading_observations(root_states, tar_dir, tar_speed, ball_pos, ball_vel, ball_ang_vel)
        return obs

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)  # 원본 기본 리셋 (캐릭터 기본 자세 + 공 환형 배치)
        if self._init_states is not None:
            file_ids = env_ids[torch.rand(len(env_ids), device=self.device) < self._init_file_prob]
            if len(file_ids) > 0:
                self._reset_from_file(file_ids)
        self._apply_fall_init(env_ids)
        return

    def _reset_from_file(self, env_ids):
        st = self._init_states
        s = torch.randint(0, st["ball_state"].shape[0], (len(env_ids),), device=self.device)
        self._humanoid_root_states[env_ids] = st["root_state"][s]
        self._dof_pos[env_ids] = st["dof_pos"][s]
        self._dof_vel[env_ids] = st["dof_vel"][s]
        self._ball_states[env_ids] = st["ball_state"][s]
        self._target_states[env_ids, :3] = self._ball_states[env_ids, :3]
        if "rb_pos" in st:
            self._rb_reset = (env_ids, s)
        return

    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        # 리셋 직후 refresh 는 시뮬레이션 전이라 이전 에피소드의 rigid body 상태를 다시 읽어 온다.
        # 그래서 _reset_actors 에서 넣으면 덮여 버리므로 refresh 뒤 (첫 관측 / AMP 관측 계산 전) 에 넣는다
        if self._rb_reset is not None:
            env_ids, s = self._rb_reset
            st = self._init_states
            self._rigid_body_pos[env_ids] = st["rb_pos"][s]
            self._rigid_body_rot[env_ids] = st["rb_rot"][s]
            self._rigid_body_vel[env_ids] = st["rb_vel"][s]
            self._rigid_body_ang_vel[env_ids] = st["rb_ang_vel"][s]
            self._rb_reset = None
        self._apply_fall_rigid_bodies()
        return

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        ball_pos = self._ball_states[..., :3]

        r1, r2, r3 = compute_heading_reward(root_pos, self._prev_root_pos, root_rot,
                                            self._tar_dir, self._tar_speed,
                                            ball_pos, self._prev_ball_pos, self.dt)
        self.rew_buf[:] = r1 + r2 + r3
        self.extras["reward_terms"] = {"ball": r1, "humanoid": r2, "dist": r3}  # logged by RLGPUAlgoObserver

        return

    def _compute_reset(self):
        # 넘어짐/최대 길이 판정은 원본 그대로 (enableEarlyTermination 가 넘어짐 쪽만 제어)
        super()._compute_reset()

        # 공을 relocate_dist 보다 멀리 놓치면 종료
        ball_dist = torch.norm(self._ball_states[..., :2] - self._humanoid_root_states[..., :2], dim=-1)
        ball_lost = ball_dist > self._relocate_dist

        self._terminate_buf[:] = torch.where(ball_lost, torch.ones_like(self._terminate_buf), self._terminate_buf)
        self.reset_buf[:] = torch.where(ball_lost, torch.ones_like(self.reset_buf), self.reset_buf)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def compute_heading_observations(root_states, tar_dir, tar_speed, ball_pos, ball_vel, ball_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_dir = quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    tar_speed = tar_speed.unsqueeze(-1)

    # ball - character pos
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    obs = torch.cat([local_tar_dir, tar_speed, local_ball_pos, local_ball_vel, local_ball_ang_vel], dim=-1)
    return obs


#@torch.jit.script
def compute_heading_reward(root_pos, prev_root_pos, root_rot, tar_dir, tar_speed, ball_pos, prev_ball_pos, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    vel_err_scale = 10.0
    pos_err_scale = 10.0
    tangent_err_w = 1.0

    # 비율
    ball_reward_w = humanoid_dribble.BALL_REWARD_W
    humanoid_reward_w = humanoid_dribble.HUMANOID_REWARD_W
    dist_reward_w = humanoid_dribble.DIST_REWARD_W

    # for ball reward
    delta_ball_pos = ball_pos - prev_ball_pos
    ball_vel = delta_ball_pos / dt
    tar_dir_speed = torch.sum(tar_dir * ball_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = ball_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.norm(tangent_vel, dim=-1)

    tar_vel_err = (tar_speed - tar_dir_speed) / tar_speed # for norm
    tangent_vel_err = tangent_speed / tar_speed # for norm
    ball_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err +
                        tangent_err_w * tangent_vel_err * tangent_vel_err))

    speed_mask = tar_dir_speed <= 0
    ball_reward[speed_mask] = 0

    # for character reward
    ball_humanoid_dir = torch_norm(ball_pos[..., :2] - root_pos[..., :2], dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(ball_humanoid_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * ball_humanoid_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.norm(tangent_vel, dim=-1)

    tar_vel_err = (tar_speed - tar_dir_speed) / tar_speed # for norm
    tangent_vel_err = tangent_speed / tar_speed # for norm
    humanoid_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err +
                        tangent_err_w * tangent_vel_err * tangent_vel_err))

    # ball pos err reward
    cur_ball_pos = ball_pos[..., :2]
    dist_err = torch.pow(cur_ball_pos - humanoid_dribble.compute_dist_anchor(root_pos, root_rot), 2).sum(dim=-1).sqrt() / tar_speed
    dist_reward = torch.exp(-pos_err_scale*dist_err)

    return ball_reward_w * ball_reward, humanoid_reward_w * humanoid_reward, dist_reward_w * dist_reward
