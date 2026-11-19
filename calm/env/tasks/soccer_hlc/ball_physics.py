"""Ball + ground physics profiles shared by the soccer tasks.

A profile is calm/data/cfg/soccer_hlc/ball/<name>.yaml with three sections:
    ball:        asset / shape settings of the ball
    ground:      plane settings (written into cfg["env"]["plane"], so the humanoid sees the same ground)
    mj_friction: MuJoCo-style torsional / rolling friction of the ball-ground pair, or null (off)
A task selects one with cfg["env"]["ballProfile"] and can patch single values with
cfg["env"]["ballOverrides"] (same layout, merged per section).

Usage in a task (see HumanoidDribble):
    class Task(BallPhysicsMixin, HumanoidAMPTask)
    __init__, before super().__init__:   self._setup_ball_profile(cfg)
    _load_obj_asset:                     self._ball_asset = self._create_ball_asset()
    after the state tensors exist:       self._init_ball_friction(ball_handles, ball_actor_ids)
"""
import math
import os
import tempfile

from isaacgym import gymapi, gymtorch
import torch
import yaml

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/cfg/soccer_hlc/ball")
BALL_ASSET_ROOT = "calm/data/assets/mjcf/"
BALL_ASSET_FILE = "soccer_ball.urdf"
GROUND_EPS = 0.02   # ball counts as on the ground when its bottom is below this height [m] (= contact_offset)


def load_ball_profile(name, overrides=None):
    with open(os.path.join(PROFILE_DIR, f"{name}.yaml"), "r") as f:
        profile = yaml.safe_load(f)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict) and isinstance(profile.get(section), dict):
            profile[section].update(values)
        else:
            profile[section] = values
    return profile


def write_ball_urdf(directory, ball):
    """Sphere URDF with explicit mass (density * volume) and inertia (inertia_ratio * m r^2)."""
    r = ball["radius"]
    mass = ball["density"] * 4.0 / 3.0 * math.pi * r ** 3
    inertia = ball["inertia_ratio"] * mass * r * r
    fname = "ball.urdf"
    with open(os.path.join(directory, fname), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<robot name="ball">
  <link name="ball">
    <inertial>
      <mass value="{mass:.8f}"/>
      <inertia ixx="{inertia:.10f}" ixy="0" ixz="0" iyy="{inertia:.10f}" iyz="0" izz="{inertia:.10f}"/>
    </inertial>
    <visual><geometry><sphere radius="{r}"/></geometry></visual>
    <collision><geometry><sphere radius="{r}"/></geometry></collision>
  </link>
</robot>
""")
    return fname


def create_ball_asset(gym, sim, ball):
    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = ball["angular_damping"]
    asset_options.linear_damping = ball["linear_damping"]
    asset_options.max_angular_velocity = ball["max_angular_velocity"]
    asset_options.density = ball["density"]
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    if ball.get("inertia_ratio") is None:
        # soccer_ball.urdf has no inertial: mass and inertia (solid sphere, 0.4 m r^2) come from the density
        asset_options.override_inertia = True
        asset = gym.load_asset(sim, BALL_ASSET_ROOT, BALL_ASSET_FILE, asset_options)
    else:
        # Explicit inertia (e.g. 2/3 = hollow shell) must be set on the asset: overriding it per actor
        # with set_actor_rigid_body_properties crashes GPU PhysX's narrowphase in Isaac Gym.
        urdf_dir = tempfile.mkdtemp(prefix="ball_profile_")
        asset = gym.load_asset(sim, urdf_dir, write_ball_urdf(urdf_dir, ball), asset_options)

    shape_property = gym.get_asset_rigid_shape_properties(asset)
    shape_property[0].restitution = ball["restitution"]
    shape_property[0].friction = ball["friction"]
    gym.set_asset_rigid_shape_properties(asset, shape_property)
    return asset


class MujocoFriction:
    """MuJoCo-style torsional + rolling contact friction for the balls, applied as external torques.

    MuJoCo (condim 6, default pyramidal cone) bounds each frictional dimension separately:
        |tau_torsion| <= mu_t * N        (spin about the contact normal)
        |tau_rolling| <= mu_r * N        (angular velocity tangent to the contact plane)
    mu_t / mu_r are lengths [m] (geom friction="slide torsional rolling"), the pair value is the
    element-wise max of both geoms, and the solver drives the relative angular velocity to zero when the
    bound allows it (stick) and saturates at the bound otherwise (slip). Here that is done explicitly:
        tau = -dir(w_rel) * min(mu * N, I_eff * |w_rel| / h)
    with I_eff = I for torsion and I + m r^2 for rolling (rolling about the contact point), h = sim dt.
    Sliding friction stays with PhysX. Only the ball-ground contact is modeled: contact normal = +z
    (flat ground at z = 0), N = m g while the ball bottom is within GROUND_EPS of the ground, 0 otherwise.
    The net contact force tensor is not used for N: with humanoids in the scene it reads 0.5x or ~28x
    m g for a ball resting on the ground (Isaac Gym GPU PhysX), which stops a rolling ball in one step.
    So a foot pressing the ball down does not raise N here.
    MuJoCo's soft-constraint softness (solref/solimp) and the elliptic cone are not reproduced.
    Everything is batched over all balls on the GPU: a few element-wise ops + one
    apply_rigid_body_force_tensors call per simulate().
    """

    def __init__(self, gym, sim, actor_ids, body_ids, mu_t, mu_r, inertia, mass, radius, h):
        self.gym, self.sim, self.h = gym, sim, h
        body_state = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        dev = body_state.device
        self.actor_ids = torch.tensor(actor_ids, dtype=torch.long, device=dev)
        self.body_ids = torch.tensor(body_ids, dtype=torch.long, device=dev)
        self.mu_t = torch.tensor(mu_t, dtype=torch.float32, device=dev)
        self.mu_r = torch.tensor(mu_r, dtype=torch.float32, device=dev)
        I = torch.tensor(inertia, dtype=torch.float32, device=dev)
        m = torch.tensor(mass, dtype=torch.float32, device=dev)
        self.radius = torch.tensor(radius, dtype=torch.float32, device=dev)
        self.weight = m * 9.81
        self.stop_t = I / h                                 # torque that stops 1 rad/s of spin in one step
        self.stop_r = (I + m * self.radius * self.radius) / h  # same for rolling
        self.forces = torch.zeros((body_state.shape[0], 3), dtype=torch.float32, device=dev)
        self.torques = torch.zeros_like(self.forces)

    def apply(self, root):
        on_ground = root[self.actor_ids, 2] - self.radius < GROUND_EPS
        n = self.weight * on_ground
        w = root[self.actor_ids, 10:13]
        wz = w[:, 2]
        tau_t = -torch.sign(wz) * torch.minimum(self.mu_t * n, self.stop_t * wz.abs())
        wt = w[:, :2]
        wt_norm = wt.norm(dim=-1)
        tau_r = torch.minimum(self.mu_r * n, self.stop_r * wt_norm) / wt_norm.clamp_min(1e-9)
        # 행 단위 대입: torch 1.10 은 deterministic 모드 (fifa.py) 에서 torques[ids, :2] = ... 같은 index + slice 대입이 실패한다
        self.torques[self.body_ids] = torch.cat([-wt * tau_r[:, None], tau_t[:, None]], dim=-1)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces),
                                                gymtorch.unwrap_tensor(self.torques), gymapi.ENV_SPACE)


class BallPhysicsMixin:
    """Puts a ball profile into a soccer task. Put it before the task base class."""

    _ball_friction = None

    def _setup_ball_profile(self, cfg):
        self._ball_profile = load_ball_profile(cfg["env"]["ballProfile"], cfg["env"].get("ballOverrides"))
        ground = self._ball_profile["ground"]
        cfg["env"]["plane"] = {"staticFriction": ground["static_friction"],
                               "dynamicFriction": ground["dynamic_friction"],
                               "restitution": ground["restitution"]}
        return

    def _create_ball_asset(self):
        return create_ball_asset(self.gym, self.sim, self._ball_profile["ball"])

    def _init_ball_friction(self, ball_handles, ball_actor_ids):
        mj = self._ball_profile.get("mj_friction")
        if mj is None:
            return
        body_ids, inertia, mass = [], [], []
        for env_ptr, handle in zip(self.envs, ball_handles):
            body_ids.append(self.gym.get_actor_rigid_body_index(env_ptr, handle, 0, gymapi.DOMAIN_SIM))
            body_property = self.gym.get_actor_rigid_body_properties(env_ptr, handle)[0]
            inertia.append(body_property.inertia.x.x)
            mass.append(body_property.mass)
        n = len(body_ids)
        self._ball_friction = MujocoFriction(self.gym, self.sim, ball_actor_ids.tolist(), body_ids,
                                             [mj["torsional"]] * n, [mj["rolling"]] * n, inertia, mass,
                                             [self._ball_profile["ball"]["radius"]] * n, self.sim_params.dt)
        return

    def _physics_step(self):
        if self._ball_friction is None:
            return super()._physics_step()
        for i in range(self.control_freq_inv):
            if i > 0:
                # The first simulate uses the tensor as it is: a refresh there could overwrite root states that
                # reset just wrote with set_actor_root_state_tensor_indexed.
                self.gym.refresh_actor_root_state_tensor(self.sim)
            self.render()
            self._ball_friction.apply(self._root_states)
            self.gym.simulate(self.sim)
        return


#@torch.jit.script
def compute_non_ground_impulse(prev_vel, vel, near_ground, mu, dt):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    # 이번 step 공이 받은 접촉 충격량 (질량당) J = Δv + g dt z 중 바닥으로 설명되지 않는 크기.
    # 바닥은 법선 성분 J_z >= 0, 마찰 |J_xy| <= mu J_z 인 원뿔 안의 충격량만 줄 수 있으므로 원뿔까지의 거리를 쓴다.
    # 바닥 근처가 아니면 J 전체. linear damping (step 당 c v dt, 약 0.1 m/s 이하) 은 무시한다.
    j = vel - prev_vel
    j[..., 2] += 9.81 * dt
    a = torch.norm(j[..., 0:2], dim=-1)
    b = j[..., 2]
    full = torch.norm(j, dim=-1)

    sin_c = mu / math.sqrt(1.0 + mu * mu)
    cos_c = 1.0 / math.sqrt(1.0 + mu * mu)
    inside = a <= mu * b
    along = a * sin_c + b * cos_c  # 원뿔 모선 방향 성분
    cone_dist = torch.where(along <= 0, full, torch.abs(a * cos_c - b * sin_c))
    cone_dist = torch.where(inside, torch.zeros_like(cone_dist), cone_dist)
    return torch.where(near_ground, cone_dist, full)
