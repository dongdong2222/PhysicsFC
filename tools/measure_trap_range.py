# HumanoidTrap_Fix 정책이 받을 수 있는 초기 공 범위 측정. calm/run.py --test 를 그대로 돌리면서
# 리셋 때 샘플 값 (공 / 캐릭터 초기 상태) 과 종료 사유를 에피소드마다 기록한다. 학습 코드는 건드리지 않는다.
#   TRAP_MEASURE_N=200000 TRAP_MEASURE_OUT=trap_range.npz python tools/measure_trap_range.py --test --headless \
#       --task HumanoidTrap_Fix --cfg_env ... --checkpoint ... (calm/run.py --test 인자 그대로)
# 샘플 값은 _reset_ball_air_trapping / _reset_ball_ground 안의 torch.rand 호출을 순서대로 가로채 복원하고,
# 복원한 값으로 다시 계산한 발사 방향 / 착지점이 실제 값과 같은지 확인한다 (코드가 바뀌면 assert 로 멈춘다).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calm"))
import run  # isaacgym 을 torch 보다 먼저 import

import math
import numpy as np
import torch

from env.tasks.soccer_hlc import humanoid_trap_fix as tf
from utils import torch_utils

N_EPISODES = int(os.environ.get("TRAP_MEASURE_N", "200000"))
OUT = os.environ.get("TRAP_MEASURE_OUT", "trap_range.npz")
FIELDS = ["is_ground", "speed", "phi", "fall_dist", "fall_theta", "launch_theta", "arrive_speed", "arrive_time",
          "flight_time", "part", "root_x", "root_y", "root_vx", "root_vy", "root_yaw", "fall_x", "fall_y",
          "ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy", "ball_vz"]
Cls = tf.HumanoidTrap_Fix


def _capture_rand(method):
    # method 안의 torch.rand 결과를 순서대로 self._meas_rand 에 남긴다
    def wrapped(self, env_ids):
        calls, orig = [], torch.rand

        def rec(*a, **k):
            t = orig(*a, **k)
            calls.append(t)
            return t
        torch.rand = rec
        try:
            method(self, env_ids)
        finally:
            torch.rand = orig
        self._meas_rand = calls
    return wrapped


def _air_values(self, env_ids, u):
    assert len(u) == 5, f"air rand calls {len(u)}"
    v_min, v_max = self._air_speed_min, self._air_speed_max
    speed = (v_max - v_min) * u[0] + v_min
    fall_dist = u[1] * (2.0 + 2.0 * (speed - v_min) / (v_max - v_min))
    fall_theta = 2 * np.pi * u[2] - np.pi
    phi = (self._air_phi_max - self._air_phi_min) * u[3] + self._air_phi_min
    theta = 2 * np.pi * u[4] - np.pi
    launch_dir = -torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    fall_xy = 0.5 * self._humanoid_root_states[env_ids, 7:9] \
              + fall_dist.unsqueeze(-1) * torch.stack([torch.cos(fall_theta), torch.sin(fall_theta)], dim=-1)
    assert torch.allclose(launch_dir, self._launch_direc[env_ids], atol=1e-5)
    assert torch.allclose(fall_xy, self._fall_pos[env_ids], atol=1e-4)
    return dict(speed=speed, phi=phi, fall_dist=fall_dist, fall_theta=fall_theta,
                launch_theta=torch.atan2(launch_dir[:, 1], launch_dir[:, 0]),
                flight_time=2 * speed * torch.sin(phi) / tf.G)


def _ground_values(self, env_ids, u):
    assert len(u) == 7, f"ground rand calls {len(u)}"
    arrive_speed = (self._ground_speed_max - self._ground_speed_min) * u[0] + self._ground_speed_min
    arrive_time = (self._ground_t_max - self._ground_t_min) * u[1] + self._ground_t_min
    fall_dist = u[2] * torch.clamp(self._fall_dist_per_sec * arrive_time, max=self._fall_dist_cap)
    fall_theta = 2 * np.pi * u[3] - np.pi
    theta = 2 * np.pi * u[4] - np.pi
    launch_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    center = self._humanoid_root_states[env_ids, 0:2] \
             + self._humanoid_root_states[env_ids, 7:9] * torch.clamp(arrive_time, max=self._fall_lead_time_max).unsqueeze(-1)
    fall_xy = center + fall_dist.unsqueeze(-1) * torch.stack([torch.cos(fall_theta), torch.sin(fall_theta)], dim=-1)
    assert torch.allclose(launch_dir, self._launch_direc[env_ids], atol=1e-5)
    assert torch.allclose(fall_xy, self._fall_pos[env_ids], atol=1e-4)
    return dict(arrive_speed=arrive_speed, arrive_time=arrive_time, fall_dist=fall_dist, fall_theta=fall_theta,
                launch_theta=theta, flight_time=arrive_time)


def _record_reset(self, env_ids):
    if not hasattr(self, "_meas"):
        self._meas = {k: torch.full([self.num_envs], float("nan"), device=self.device) for k in FIELDS}
        self._meas_valid = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self._meas_rows = []
        self._meas_count = 0
    m = self._meas
    for k in FIELDS:
        m[k][env_ids] = float("nan")
    is_ground = self._is_ground[env_ids]
    for ids, name in ((env_ids[~is_ground], "air"), (env_ids[is_ground], "ground")):
        if len(ids) == 0:
            continue
        u = self._meas_calls.pop(name)
        vals = _air_values(self, ids, u) if name == "air" else _ground_values(self, ids, u)
        for k, v in vals.items():
            m[k][ids] = v
    root = self._humanoid_root_states[env_ids]
    ball = self._ball_states[env_ids]
    m["is_ground"][env_ids] = is_ground.float()
    m["part"][env_ids] = torch.argmax(self._onehot_body_part[env_ids], dim=-1).float()
    m["root_x"][env_ids], m["root_y"][env_ids] = root[:, 0], root[:, 1]
    m["root_vx"][env_ids], m["root_vy"][env_ids] = root[:, 7], root[:, 8]
    m["root_yaw"][env_ids] = torch_utils.calc_heading(root[:, 3:7])
    m["fall_x"][env_ids], m["fall_y"][env_ids] = self._fall_pos[env_ids, 0], self._fall_pos[env_ids, 1]
    for i, a in enumerate("xyz"):
        m[f"ball_{a}"][env_ids] = ball[:, i]
        m[f"ball_v{a}"][env_ids] = ball[:, 7 + i]
    self._meas_valid[env_ids] = True


def _wrap_ball(name, method):
    captured = _capture_rand(method)

    def wrapped(self, env_ids):
        captured(self, env_ids)
        self._meas_calls[name] = self._meas_rand
    return wrapped


_orig_reset_actors = Cls._reset_actors
_orig_compute_reset = Cls._compute_reset
Cls._reset_ball_air_trapping = _wrap_ball("air", Cls._reset_ball_air_trapping)
Cls._reset_ball_ground = _wrap_ball("ground", Cls._reset_ball_ground)


def _reset_actors(self, env_ids):
    assert self._air_init == "trapping", "측정 스크립트는 AIR_INIT: trapping 만 지원"
    self._meas_calls = {}
    _orig_reset_actors(self, env_ids)
    if len(env_ids) > 0:
        _record_reset(self, env_ids)


def _compute_reset(self):
    _orig_compute_reset(self)
    done = self.reset_buf.bool() & self._meas_valid
    if not done.any():
        return
    row = {k: v[done].cpu().numpy() for k, v in self._meas.items()}
    row["end_reason"] = self._end_reason[done].cpu().numpy()
    row["end_step"] = self.progress_buf[done].cpu().numpy()
    self._meas_rows.append(row)
    self._meas_valid[done] = False
    n_before = self._meas_count
    self._meas_count += int(done.sum())
    if self._meas_count // 10000 > n_before // 10000:
        print(f"[measure_trap_range] {self._meas_count} / {N_EPISODES} episodes", flush=True)
    if self._meas_count >= N_EPISODES:
        out = {k: np.concatenate([r[k] for r in self._meas_rows]) for k in self._meas_rows[0]}
        out["end_reason_names"] = np.array(tf.END_REASONS)
        out["lin_damp"] = np.array(self._lin_damp)
        out["ball_radius"] = np.array(self._ball_radius)
        out["dt"] = np.array(self.dt)
        out["max_episode_length"] = np.array(self.max_episode_length)
        np.savez(OUT, **out)
        print(f"[measure_trap_range] saved {self._meas_count} episodes to {OUT}", flush=True)
        os._exit(0)


Cls._reset_actors = _reset_actors
Cls._compute_reset = _compute_reset

# 정해진 에피소드 수를 모을 때까지 player 가 멈추지 않게 games_num 을 크게
_orig_load_cfg = run.load_cfg


def _load_cfg(args):
    cfg, cfg_train, logdir = _orig_load_cfg(args)
    cfg_train["params"]["config"].setdefault("player", {})["games_num"] = 10 ** 9
    return cfg, cfg_train, logdir


run.load_cfg = _load_cfg

if __name__ == "__main__":
    run.main()
