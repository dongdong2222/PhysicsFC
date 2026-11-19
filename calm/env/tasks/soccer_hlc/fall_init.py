"""Test-only fallen-state starts for the soccer Fix tasks.

cfg["env"]:
    FALL_INIT_PROB:   probability that a reset starts the humanoid from a fallen pose (0 = off). Ignored in training.
    FALL_GRACE_STEPS: for that many steps a fallen-start env does not end on fall / handball (hands touch the ground).

The fallen poses are made once at start-up like HumanoidAMPGetup._generate_fall_states: drop the humanoid with a random
root orientation and random PD targets and simulate. On reset the task's own reset (ball, init files) runs first, then
the humanoid root height / rotation / joints are replaced, keeping the reset's root xy and heading, so the ball stays
where the task put it relative to the character.

Usage in a task:
    __init__, before super().__init__:  self._setup_fall_init(cfg)
    __init__, at the end:               self._build_fall_states()
    _reset_actors, at the end:          self._apply_fall_init(env_ids)
    _refresh_sim_tensors, at the end:   self._apply_fall_rigid_bodies()
    fall / handball termination:        ... & ~self._in_fall_grace()
"""
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_mul, quat_rotate, to_torch

from utils import torch_utils

FALL_SIM_STEPS = 150  # same as HumanoidAMPGetup


class FallInitMixin:

    def _setup_fall_init(self, cfg):
        is_train = cfg["args"].train
        self._fall_init_prob = 0.0 if is_train else cfg["env"].get("FALL_INIT_PROB", 0.0)
        self._fall_grace_steps = cfg["env"].get("FALL_GRACE_STEPS", 0)
        self._fall_states = None  # super().__init__ 중 리셋은 기본 리셋만
        self._fall_rb_reset = None
        if is_train and cfg["env"].get("FALL_INIT_PROB", 0.0) > 0:
            print("[FallInit] FALL_INIT_PROB is test-only, ignored in training")
        return

    def _build_fall_states(self):
        self._fall_start = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self._fall_init_prob <= 0:
            return
        root = self._initial_humanoid_root_states.clone()
        root[:, 3:7] = torch.nn.functional.normalize(torch.randn_like(root[:, 3:7]), dim=-1)
        self._humanoid_root_states[:] = root
        self._dof_pos[:] = self._initial_dof_pos
        self._dof_vel[:] = 0
        actor_ids = self._humanoid_actor_ids
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
        rand_actions = to_torch(np.random.uniform(-0.5, 0.5, size=[self.num_envs, self.get_action_size()]), device=self.device)
        pd_tar = self._action_to_pd_targets(rand_actions)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pd_tar))
        for _ in range(FALL_SIM_STEPS):
            self.gym.simulate(self.sim)
        self._refresh_sim_tensors()

        # heading 을 뺀 자세로 저장: 리셋 때 그 env 의 heading 을 다시 곱한다
        root_pos = self._humanoid_root_states[:, 0:3]
        heading_inv = torch_utils.calc_heading_quat_inv(self._humanoid_root_states[:, 3:7])
        num_bodies = self._rigid_body_pos.shape[1]
        heading_inv_b = heading_inv.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
        rb_rel = self._rigid_body_pos.clone()
        rb_rel[..., 0:2] -= root_pos[:, None, 0:2]
        self._fall_states = {
            "root_z": root_pos[:, 2].clone(),
            "root_rot": quat_mul(heading_inv, self._humanoid_root_states[:, 3:7]),
            "dof_pos": self._dof_pos.clone(),
            "rb_pos": quat_rotate(heading_inv_b, rb_rel.reshape(-1, 3)).view(self.num_envs, num_bodies, 3),
            "rb_rot": quat_mul(heading_inv_b, self._rigid_body_rot.reshape(-1, 4)).view(self.num_envs, num_bodies, 4),
        }
        print(f"[FallInit] {self.num_envs} fallen poses, root height {float(root_pos[:, 2].min()):.2f}-{float(root_pos[:, 2].max()):.2f} m, "
              f"FALL_INIT_PROB {self._fall_init_prob}, FALL_GRACE_STEPS {self._fall_grace_steps}")
        return

    def _apply_fall_init(self, env_ids):
        if self._fall_states is None:
            return
        fall_mask = torch.rand(len(env_ids), device=self.device) < self._fall_init_prob
        self._fall_start[env_ids] = fall_mask
        ids = env_ids[fall_mask]
        if len(ids) == 0:
            return
        st = self._fall_states
        s = torch.randint(0, self.num_envs, (len(ids),), device=self.device)
        heading = torch_utils.calc_heading_quat(self._humanoid_root_states[ids, 3:7])
        self._humanoid_root_states[ids, 2] = st["root_z"][s]
        self._humanoid_root_states[ids, 3:7] = quat_mul(heading, st["root_rot"][s])
        self._humanoid_root_states[ids, 7:13] = 0
        self._dof_pos[ids] = st["dof_pos"][s]
        self._dof_vel[ids] = 0
        self._fall_rb_reset = (ids, s, heading, self._humanoid_root_states[ids, 0:2].clone())
        return

    def _apply_fall_rigid_bodies(self):
        # 리셋 직후 refresh 는 이전 에피소드 rigid body 를 읽어 오므로, 첫 관측용으로 넘어진 자세의 값을 넣는다
        if self._fall_rb_reset is None:
            return
        ids, s, heading, root_xy = self._fall_rb_reset
        st = self._fall_states
        num_bodies = st["rb_pos"].shape[1]
        heading_b = heading.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
        rb_pos = quat_rotate(heading_b, st["rb_pos"][s].reshape(-1, 3)).view(len(ids), num_bodies, 3)
        rb_pos[..., 0:2] += root_xy[:, None]
        self._rigid_body_pos[ids] = rb_pos
        self._rigid_body_rot[ids] = quat_mul(heading_b, st["rb_rot"][s].reshape(-1, 4)).view(len(ids), num_bodies, 4)
        self._rigid_body_vel[ids] = 0
        self._rigid_body_ang_vel[ids] = 0
        self._fall_rb_reset = None
        return

    def _in_fall_grace(self):
        if self._fall_states is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return self._fall_start & (self.progress_buf < self._fall_grace_steps)
