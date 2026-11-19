"""FIFA turf ball tests, visualized side by side for each project's ball settings.

  Vertical rebound : ball bottom dropped from 2.00 m onto a drop pad -> first rebound height,
                     run on both pads in every lane:
                     steel: FIFA Quality Pro *ball* test, steel plate, 1.35 ~ 1.55 m at 20 C.
                         The pad's restitution is set equal to the ball's, so PhysX's
                         average-combined restitution is the ball's own (rigid plate).
                     turf:  FIFA *artificial turf* test, pad = that lane's ground,
                         FIFA Quality 0.60 ~ 1.00 m, FIFA Quality Pro 0.60 ~ 0.85 m
  Ball roll        : ball leaves a 1 m high ramp rolling at 3.43 m/s -> distance until it stops
                     FIFA Quality 4 ~ 10 m,      FIFA Quality Pro 4 ~ 8 m

  Spin (torsion)   : ball resting on the roll ground spinning about z at 10 rad/s -> time until < 0.1 rad/s

One lane per config, side by side along +y (every LANE_SPACING m). Each lane runs all tests at once:
  - drop balls : left column, steel pad (+y side) and turf pad (-y side), each next to a pole showing
                 that surface's FIFA rebound band
  - roll ball  : starts at x = 0 and rolls toward +x over green FIFA distance strips
  - spin ball  : just behind the roll start, spinning in place (white debug line shows its heading)

--mj_friction adds MuJoCo-style torsional and rolling contact friction to the balls (see MujocoFriction);
PhysX itself ignores Isaac Gym's rolling_friction / torsion_friction.
The Isaac Gym viewer shows the lanes with FIFA markers (markers never collide with the balls) and
live results as debug lines (red = first rebound apex, orange = latest apex, red post = where the roll
ball stopped). Every event is printed to the log, followed by a summary table, and a self-contained
HTML replay (--html) with the same markers, playback controls, event log and summary is written in
both viewer and --headless mode, so a server run can be watched in a browser afterwards.

Only the ball and ground settings differ per lane; all lanes share one sim
(dt = 1/60, substeps = --substeps). Ball shapes are plain spheres (= the URDF collision sphere).

Run inside the pulse-v1 container, from the PULSE root:
    python scripts/ball_fifa_test_vis.py                       # viewer, real time (+ log, HTML replay)
    python scripts/ball_fifa_test_vis.py --fast 4              # viewer, 4x speed
    python scripts/ball_fifa_test_vis.py --headless            # no viewer: log + HTML replay only
    python scripts/ball_fifa_test_vis.py --configs pulse fifa_proposal --html output/ball_test.html
    python scripts/ball_fifa_test_vis.py --configs physicsfc_dribble_tuned mj_friction --mj_friction
    CUDA_VISIBLE_DEVICES=1 python scripts/ball_fifa_test_vis.py   # pick the GPU
Viewer keys: R = restart, SPACE = pause
"""
import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np
from isaacgym import gymapi, gymtorch
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calm"))
from env.tasks.soccer_hlc.ball_physics import MujocoFriction  # noqa: E402  (shared with the training tasks)

# --- FIFA Quality Programme for Football Turf ------------------------------------------------
DROP_HEIGHT = 2.0                       # ball bottom above the ground [m]
REBOUND_CRITERIA = {
    # FIFA Quality Pro ball: 2 m drop onto a steel plate at 20 C (0.8 bar), 135 ~ 155 cm
    # extra_ticks: reference heights drawn in light green only (not used for the verdict);
    # 1.25 m = the same test's minimum at 5 C
    "steel": dict(label="steel plate (FIFA Quality Pro ball test)", q=(1.35, 1.55), pro=(1.35, 1.55),
                  extra_ticks=(1.25,)),
    # FIFA artificial turf: 2 m drop onto the turf (pad = that lane's ground)
    "turf": dict(label="lane ground (FIFA artificial turf test)", q=(0.60, 1.00), pro=(0.60, 0.85),
                 extra_ticks=()),
}
DROP_SURFACES = ("steel", "turf")                # both rebound tests run in every lane
MJ_ON = False                                   # --mj_friction
ROLL_Q = (4.0, 10.0)
ROLL_PRO = (4.0, 8.0)
RAMP_HEIGHT = 1.0
ROLL_V0 = math.sqrt(2 * 9.81 * RAMP_HEIGHT / (1 + 2.0 / 3.0))  # hollow ball rolling off the ramp, ~3.43 m/s

MAX_LOGGED_REBOUNDS = 5                # rebounds after this are tracked but not logged
SETTLE_APEX = 0.01                     # drop test ends once a rebound is lower than this [m]
STOP_SPEED = 0.02                       # roll ball counts as stopped below this [m/s]
LANE_END = 55.0                         # roll ball past this x has run off the lane
SPIN_W0 = 10.0                          # spin test: initial angular velocity about z [rad/s]
SPIN_STOP = 0.1                         # spin ball counts as stopped below this [rad/s]
MJ_DEFAULT_FRICTION = (1.0, 0.005, 0.0001)  # MuJoCo geom friction default: (slide, torsional, rolling)

# --- per-project ball / ground settings ------------------------------------------------------
# None = not set in that project's code -> Isaac Gym default
# (AssetOptions: linear_damping 0.0, angular_damping 0.5, max_angular_velocity 64;
#  shape: torsion 0, compliance 0)
CONFIGS = {
    "pulse": dict(
        label="PULSE soccer", src="humanoid_im_soccer_distill_getup.py / marladona",
        color=(0.15, 0.45, 1.0),
        radius=0.11, density=80.71, linear_damping=0.1, angular_damping=0.05, max_angular_velocity=100.0,
        friction=0.2, rolling_friction=None, torsion_friction=None, restitution=0.8, compliance=None,
        ground_friction=1.0, ground_restitution=0.2,
        mj_friction=(0, 0, 0),
        ground_mj_friction=(0,0,0),
    ),
    "mimic2dm": dict(
        label="mimic2DM dribble", src="humanoid_amp_tracking2d.py",
        color=(1.0, 0.55, 0.1),
        radius=0.12, density=50.0, linear_damping=0.0, angular_damping=0.5, max_angular_velocity=None,
        friction=1.0, rolling_friction=None, torsion_friction=None, restitution=0.8, compliance=0.2,
        ground_friction=1.0, ground_restitution=0.0,
        mj_friction=(0, 0, 0),
        ground_mj_friction=(0,0,0),
    ),
    "angular_damping": dict(
        label="angular_damping",
        src="physicsfc_dribble with angular_damping 0.5 -> 1.5, density 80.71, friction 0.2 (not used by any project)",
        color=(0.35, 0.85, 0.2),
        radius=0.11, density=80.71, linear_damping=0.1, angular_damping=1.5, max_angular_velocity=100.0,
        friction=0.2, rolling_friction=None, torsion_friction=None, restitution=0.8, compliance=None,
        ground_friction=1.0, ground_restitution=0.2,
        mj_friction=(0, 0, 0),
        ground_mj_friction=(0,0,0),

    ),
    # MuJoCo-style friction instead of damping-based rolling resistance (run with --mj_friction).
    # linear_damping 0.2 stays as air drag; angular_damping is small; rolling 0.010 m gives ~6 m at 3.43 m/s
    # together with the damping (constant decel ~0.64 m/s^2 from rolling friction: mu_r*g / (r*(1+beta)) ).
    "from_motor_control": dict(
        label="MuJoCo friction", src="physicsfc_dribble_tuned + MuJoCo torsion/rolling (needs --mj_friction)",
        color=(1.0, 0.85, 0.1),
        radius=0.11, density=80.71, linear_damping=0.0, angular_damping=0.0, max_angular_velocity=300.0,
        friction=0.2, rolling_friction=0.04, torsion_friction=0.05, restitution=0.25, compliance=None,
        ground_friction=1.0, ground_restitution=0.2,
        mj_friction=(1.0, 0.05, 0.04),
    ),
    "no_damping": dict(
        label="no_damping", src="no damping constraint",
        color=(1.0, 1.0, 1.0),
        radius=0.11, density=80.71, linear_damping=0.0, angular_damping=0.0, max_angular_velocity=300.0,
        friction=0.2, rolling_friction=0.04, torsion_friction=0.8, restitution=0.25, compliance=None,
        ground_friction=1.0, ground_restitution=0.2,
        mj_friction=(0, 0, 0),
        ground_mj_friction=(0,0,0),
    ),
    # Closest to a real match ball on turf (run with --mj_friction):
    #   FIFA size 5 ball (r 0.11 m, 0.43 kg = density 77.1), hollow shell inertia I = 2/3 m r^2,
    #   restitution 0.88 -> ~1.43 m on the steel plate (FIFA Quality Pro ball 1.35~1.55 m),
    #   turf restitution 0.35 -> pair 0.615 -> ~0.70 m on turf (FIFA turf Pro 0.60~0.85 m),
    #   sliding friction pair (0.2 + 1.0) / 2 = 0.6 (ball on grass ~0.4~0.6),
    #   linear_damping 0.1 as air drag (equals real drag ~0.013 v^2 at ~8 m/s), small angular_damping,
    #   max_angular_velocity 300 (rolls without slip up to ~33 m/s),
    #   MuJoCo rolling 0.015 m -> ~0.8 m/s^2 rolling resistance, ~6 m at 3.43 m/s with the damping,
    #   MuJoCo torsional 0.005 m (MuJoCo default; no measured value for a ball on turf).
    "real_soccer": dict(
        label="Real soccer (best guess)", src="FIFA ball/turf specs + MuJoCo friction (needs --mj_friction)",
        color=(1.0, 0.3, 0.6),
        radius=0.11, density=77.1, inertia_ratio=2.0 / 3.0,
        linear_damping=0.1, angular_damping=0.0, max_angular_velocity=300.0,
        friction=0.2, rolling_friction=None, torsion_friction=None, restitution=0.88, compliance=None,
        ground_friction=1.0, ground_restitution=0.35,
        mj_friction=(1.0, 0.005, 0.015),
    ),
}
DEFAULT_CONFIGS = ["pulse", "mimic2dm", "physicsfc_fifa", "physicsfc_dribble"]

# --- lane layout (env-local coordinates) -----------------------------------------------------
DROP_XY = {"steel": (-2.5, 0.8), "turf": (-2.5, -0.7)}
ROLL_XY = (0.0, -0.5)
POLE_XY = {s: (x, y + 0.3) for s, (x, y) in DROP_XY.items()}
GROUND_X = (-1.2, 57.0)      # roll ground (that project's ground settings)
DROP_PAD_X = (-4.0, -1.4)    # separate pads under the drop balls, split along y
DROP_PAD_Y = {"steel": (0.05, 1.4), "turf": (-1.4, 0.05)}
SPIN_XY = (-0.6, 0.6)        # spin ball, on the roll ground behind the roll start
GROUND_Y = (-1.4, 1.4)
LANE_SPACING = 4.0      # lane i is shifted by +i * LANE_SPACING in y. All lanes live in ONE env:
                        # Isaac Gym's env grid puts the 2nd env next to the 1st along x even with
                        # num_per_row=1, so the lane layout is done by hand instead.

# collision filter bits: shapes collide only if (filter_a & filter_b) == 0
BALL_FILTER = 0b001       # the two balls of a lane don't collide with each other
GROUND_FILTER = 0b110     # collides with balls
MARKER_FILTER = 0b111     # collides with nothing that moves


def write_ball_urdf(directory, name, c):
    """Sphere URDF with explicit mass (density * volume) and inertia (inertia_ratio * m r^2)."""
    r = c["radius"]
    mass = c["density"] * 4.0 / 3.0 * math.pi * r ** 3
    inertia = c["inertia_ratio"] * mass * r * r
    fname = f"ball_{name}.urdf"
    with open(os.path.join(directory, fname), "w") as f:
        f.write(f"""<?xml version="1.0"?>
<robot name="ball_{name}">
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


def pad_restitution(c, surface):
    # steel plate: equal to the ball's restitution, so the average-combined pair value is the ball's own
    return c["restitution"] if surface == "steel" else c["ground_restitution"]


def rebound_verdict(h, surface):
    if h is None:
        return "-"
    q, pro = REBOUND_CRITERIA[surface]["q"], REBOUND_CRITERIA[surface]["pro"]
    if h < q[0]:
        return "LOW"
    if h > q[1]:
        return "HIGH"
    return "OK (Pro)" if h <= pro[1] else "OK (Quality only)"


def roll_verdict(d, stopped):
    if not stopped:
        return "LONG (still rolling)" if d > ROLL_Q[1] else "still rolling"
    if d < ROLL_Q[0]:
        return "SHORT"
    if d > ROLL_Q[1]:
        return "LONG"
    return "OK (Pro)" if d <= ROLL_PRO[1] else "OK (Quality only)"


class DropTest:
    def __init__(self, surface):
        self.surface = surface
        self.apexes = []
        self.ascending = False
        self.cur_apex = 0.0
        self.prev_vz = 0.0
        self.last_event_t = 0.0
        self.done = False


class Lane:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.reset()

    def reset(self):
        self.drops = {s: DropTest(s) for s in DROP_SURFACES}
        self.roll_dist = 0.0
        self.roll_speed = ROLL_V0
        self.roll_stopped = False
        self.roll_off_lane = False
        self.roll_stop_t = None
        self.roll_next_mark = 0
        self.roll_next_report = 10.0
        self.spin_done = False
        self.spin_stop_t = None
        self.spin_w = SPIN_W0
        self.spin_reported_1s = False

    @property
    def done(self):
        return all(d.done for d in self.drops.values()) and (self.roll_stopped or self.roll_off_lane) and self.spin_done


EVENTS = []  # (t, lane name, kind, message) -- also shown in the HTML replay


def log(t, lane, kind, msg):
    print(f"[t={t:6.2f}s] {lane.cfg['label']:<24} {kind:<5} {msg}")
    EVENTS.append(dict(t=round(t, 3), lane=lane.name, kind=kind, msg=msg))


def update_drop(lane, drop, t, z_bottom, vz):
    if drop.done:
        return
    kind = drop.surface.upper()                           # log kind: STEEL / TURF
    q, pro = REBOUND_CRITERIA[drop.surface]["q"], REBOUND_CRITERIA[drop.surface]["pro"]
    # bounced off the ground; impacts slower than PhysX's bounce threshold (0.2 m/s) don't bounce,
    # which also ignores the resting ball's tiny contact jitter
    if drop.prev_vz < -0.2 and vz >= 0.0 and z_bottom < 0.05:
        drop.ascending, drop.cur_apex = True, z_bottom
    if drop.ascending:
        drop.cur_apex = max(drop.cur_apex, z_bottom)
        if vz <= 0.0:                                     # apex reached
            drop.ascending = False
            drop.apexes.append(drop.cur_apex)
            drop.last_event_t = t
            n = len(drop.apexes)
            if n == 1:
                e = math.sqrt(max(drop.cur_apex, 0.0) / DROP_HEIGHT)
                log(t, lane, kind, f"rebound #1 = {drop.cur_apex:.3f} m  (COR {e:.2f})  "
                    f"FIFA {q[0]:.2f}~{q[1]:.2f} / Pro ~{pro[1]:.2f} -> "
                    f"{rebound_verdict(drop.cur_apex, drop.surface)}")
            elif n <= MAX_LOGGED_REBOUNDS:
                log(t, lane, kind, f"rebound #{n} = {drop.cur_apex:.3f} m")
            if drop.cur_apex < SETTLE_APEX:   # nearly elastic pairs keep bouncing for a long time
                drop.done = True
                log(t, lane, kind, f"rebound below {SETTLE_APEX * 100:.0f} cm after {n} rebounds -> done")
                return
    drop.prev_vz = vz
    if not drop.ascending and t - drop.last_event_t > 1.0 and z_bottom < 0.005 and abs(vz) < 0.05:
        drop.done = True
        log(t, lane, kind, f"settled after {len(drop.apexes)} rebounds")


def update_roll(lane, t, dist, speed):
    if lane.roll_stopped or lane.roll_off_lane:
        return
    lane.roll_dist, lane.roll_speed = dist, speed
    marks = [ROLL_Q[0], ROLL_PRO[1], ROLL_Q[1]]
    while lane.roll_next_mark < len(marks) and dist >= marks[lane.roll_next_mark]:
        log(t, lane, "ROLL", f"passed {marks[lane.roll_next_mark]:4.1f} m  (speed {speed:.2f} m/s)")
        lane.roll_next_mark += 1
    if t >= lane.roll_next_report:
        log(t, lane, "ROLL", f"{dist:6.2f} m, speed {speed:.3f} m/s")
        lane.roll_next_report += 10.0
    if dist > LANE_END:
        lane.roll_off_lane = True
        log(t, lane, "ROLL", f"ran off the lane at {dist:.1f} m (speed {speed:.2f} m/s) -> LONG")
    elif speed < STOP_SPEED:
        lane.roll_stopped, lane.roll_stop_t = True, t
        log(t, lane, "ROLL", f"stopped at {dist:.2f} m  FIFA {ROLL_Q[0]:.0f}~{ROLL_Q[1]:.0f} / "
            f"Pro ~{ROLL_PRO[1]:.0f} -> {roll_verdict(dist, True)}")


def update_spin(lane, t, wz):
    if lane.spin_done:
        return
    lane.spin_w = wz
    if not lane.spin_reported_1s and t >= 1.0:
        lane.spin_reported_1s = True
        log(t, lane, "SPIN", f"omega_z after 1 s = {wz:.2f} rad/s (from {SPIN_W0:.0f})")
    if abs(wz) < SPIN_STOP:
        lane.spin_done, lane.spin_stop_t = True, t
        log(t, lane, "SPIN", f"stopped (< {SPIN_STOP} rad/s) after {t:.2f} s")


def mj_pair_friction(c):
    """MuJoCo combines two geoms' friction by element-wise max (equal priority)."""
    ball = c.get("mj_friction", MJ_DEFAULT_FRICTION)
    ground = c.get("ground_mj_friction", MJ_DEFAULT_FRICTION)
    return tuple(max(a, b) for a, b in zip(ball, ground))


def print_summary(lanes, t, masses):
    print(f"\n===== summary at t = {t:.1f}s  (drop {DROP_HEIGHT:.1f} m, roll v0 {ROLL_V0:.2f} m/s) =====")
    hdr = (f"{'config':<24}{'mass':>7} | "
           + "".join(f"{s + ' #1':>9} {'COR':>5}  {'verdict':<18}| " for s in DROP_SURFACES)
           + f"{'roll dist':>9} {'t_stop':>7}  {'verdict':<20}| {'spin stop':>9}")
    print(hdr)
    print("-" * len(hdr))
    for lane in lanes:
        reb = ""
        for s in DROP_SURFACES:
            ap = lane.drops[s].apexes
            h = ap[0] if ap else None
            cor = f"{math.sqrt(h / DROP_HEIGHT):.2f}" if h is not None else "-"
            hs = f"{h:.3f}" if h is not None else "-"
            reb += f"{hs:>9} {cor:>5}  {rebound_verdict(h, s):<18}| "
        stopped = lane.roll_stopped
        dist = f"{lane.roll_dist:.2f}" + ("" if stopped else "+")
        ts = f"{lane.roll_stop_t:.1f}" if stopped else "-"
        verdict = "LONG (off lane)" if lane.roll_off_lane else roll_verdict(lane.roll_dist, stopped)
        spin = f"{lane.spin_stop_t:.2f}s" if lane.spin_done else f">{t:.0f}s"
        print(f"{lane.cfg['label']:<24}{masses[lane.name]:>6.3f}k | {reb}"
              f"{dist:>9} {ts:>7}  {verdict:<20}| {spin:>9}")
    st, tf = REBOUND_CRITERIA["steel"], REBOUND_CRITERIA["turf"]
    print(f"FIFA: rebound steel {st['q'][0]:.2f}~{st['q'][1]:.2f} m, "
          f"turf {tf['q'][0]:.2f}~{tf['q'][1]:.2f} m (Pro ~{tf['pro'][1]:.2f}), "
          f"roll {ROLL_Q[0]:.0f}~{ROLL_Q[1]:.0f} m (Pro ~{ROLL_PRO[1]:.0f})   '+' = not stopped yet   "
          f"spin stop = {SPIN_W0:.0f} -> {SPIN_STOP} rad/s\n")


def write_html(path, lanes, masses, frames, record_fps, t_end, substeps):
    lane_data = []
    for lane in lanes:
        c = lane.cfg
        rebound = {}
        for s, d in lane.drops.items():
            h = d.apexes[0] if d.apexes else None
            rebound[s] = dict(h1=h, apexes=[round(a, 4) for a in d.apexes], verdict=rebound_verdict(h, s))
        lane_data.append(dict(
            name=lane.name, label=c["label"], src=c["src"],
            color="rgb({},{},{})".format(*[int(255 * v) for v in c["color"]]),
            radius=c["radius"], mass=round(masses[lane.name], 4),
            params={k: c[k] for k in ("linear_damping", "angular_damping", "max_angular_velocity", "friction",
                                      "rolling_friction", "torsion_friction", "restitution", "compliance",
                                      "ground_friction", "ground_restitution")},
            steel_pad_restitution=pad_restitution(c, "steel"),
            rebound=rebound,
            roll_dist=round(lane.roll_dist, 3), roll_stopped=lane.roll_stopped, roll_stop_t=lane.roll_stop_t,
            roll_off_lane=lane.roll_off_lane,
            spin_stop_t=lane.spin_stop_t,
            mj_friction=mj_pair_friction(c) if MJ_ON else None,
            roll_verdict="LONG (off lane)" if lane.roll_off_lane else roll_verdict(lane.roll_dist, lane.roll_stopped),
        ))
    data = dict(
        fifa=dict(drop_height=DROP_HEIGHT, surfaces=DROP_SURFACES,
                  rebound={s: dict(q=c["q"], pro=c["pro"], extra_ticks=c["extra_ticks"])
                           for s, c in REBOUND_CRITERIA.items()},
                  roll_q=ROLL_Q, roll_pro=ROLL_PRO, roll_v0=round(ROLL_V0, 3), ramp_height=RAMP_HEIGHT),
        sim=dict(dt="1/60", substeps=substeps, fps=record_fps, t_end=round(t_end, 3), mj_friction=MJ_ON,
                 spin_w0=SPIN_W0, spin_stop=SPIN_STOP),
        lanes=lane_data, frames=frames, events=EVENTS,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(HTML_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"HTML replay written to {os.path.abspath(path)}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FIFA Ball Tests</title>
<style>
:root{--bg:#f5f6f8;--panel:#fff;--text:#1d2330;--muted:#5d6676;--line:#d6dae1;--grid:#e8ebef;
 --pro:#2f9e52;--q:#a6e08f;--bad:#d1372b;--good:#14935a;--warn:#c27a00;--ground:#b9bec6;--start:#1d2330}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1218;--panel:#171b23;--text:#e6e9ef;
 --muted:#9aa3b2;--line:#2b323d;--grid:#222934;--pro:#2f9e52;--q:#5f9e4f;--bad:#ff6b5e;--good:#3ccf8e;--warn:#f0b13c;
 --ground:#3d4450;--start:#e6e9ef}}
:root[data-theme="dark"]{--bg:#0f1218;--panel:#171b23;--text:#e6e9ef;--muted:#9aa3b2;--line:#2b323d;--grid:#222934;
 --pro:#2f9e52;--q:#5f9e4f;--bad:#ff6b5e;--good:#3ccf8e;--warn:#f0b13c;--ground:#3d4450;--start:#e6e9ef}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 2px}
.sub{color:var(--muted);margin:0 0 12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;position:sticky;top:0;z-index:2}
button,select{font:inherit;padding:5px 12px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--text);cursor:pointer}
input[type=range]{flex:1;min-width:140px}
.time{font-variant-numeric:tabular-nums;min-width:110px;text-align:right}
canvas{width:100%;display:block}
h2{font-size:15px;margin:0 0 8px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--muted);font-size:12px;margin-top:8px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:5px}
.ln{display:inline-block;width:16px;height:0;border-top:2px solid;vertical-align:3px;margin-right:5px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}.warn{color:var(--warn);font-weight:600}
.scroll{overflow-x:auto}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:0;border:1px solid var(--muted)}
#log{height:240px;overflow-y:auto;font:12px/1.55 ui-monospace,Menlo,Consolas,monospace;white-space:pre}
#log div.future{opacity:.28}
#log div.now{background:color-mix(in srgb,var(--q) 25%,transparent)}
</style></head><body><div class="wrap">
<h1>FIFA 공 물리 시험: 프로젝트별 공 설정 비교</h1>
<p class="sub" id="sub"></p>
<div class="panel controls">
  <button id="play">▶ 재생</button><button id="restart">⟲ 처음부터</button>
  <label>속도 <select id="speed"><option value="0.5">0.5</option><option value="1">1</option><option value="2" selected>2</option>
  <option value="4">4</option><option value="8">8</option><option value="16">16</option></select> x</label>
  <input type="range" id="scrub" min="0" max="1000" value="0" aria-label="time">
  <span class="time" id="time"></span>
</div>
<div class="panel"><h2 id="dropTitle_steel"></h2><canvas id="drop_steel"></canvas>
  <div class="legend"><span><span class="sw" style="background:var(--pro)"></span>FIFA Quality Pro 구간</span>
  <span><span class="ln" style="border-color:var(--start);border-top-style:dashed"></span>낙하 시작 높이 (공 바닥)</span>
  <span><span class="ln" style="border-color:var(--q)"></span>1.25 m: 같은 시험의 5°C 기준 하한 (참고선, 판정에는 미사용)</span>
  <span><span class="ln" style="border-color:#e5322d"></span>첫 리바운드 최고점 (FIFA 판정값)</span>
  <span><span class="ln" style="border-color:#f08c00"></span>가장 최근 리바운드 최고점</span></div></div>
<div class="panel"><h2 id="dropTitle_turf"></h2><canvas id="drop_turf"></canvas>
  <div class="legend"><span><span class="sw" style="background:var(--pro)"></span>FIFA Quality Pro 구간 (인조잔디)</span>
  <span><span class="sw" style="background:var(--q)"></span>FIFA Quality 전용 구간 (인조잔디)</span>
  <span><span class="ln" style="border-color:var(--start);border-top-style:dashed"></span>낙하 시작 높이 (공 바닥)</span>
  <span><span class="ln" style="border-color:#e5322d"></span>첫 리바운드 최고점 (FIFA 판정값)</span>
  <span><span class="ln" style="border-color:#f08c00"></span>가장 최근 리바운드 최고점</span></div></div>
<div class="panel"><h2 id="rollTitle"></h2><canvas id="roll"></canvas>
  <div class="legend"><span><span class="sw" style="background:var(--pro)"></span>FIFA Quality Pro 구간</span>
  <span><span class="sw" style="background:var(--q)"></span>FIFA Quality 전용 구간</span>
  <span><span class="ln" style="border-color:#e5322d"></span>정지 위치</span>
  <span>공 위의 선 = 회전 표시</span></div></div>
<div class="panel"><h2>결과 요약</h2><div class="scroll"><table id="summary"></table></div></div>
<div class="panel"><h2>이벤트 로그 (현재 시점까지 진하게 표시)</h2><div id="log"></div></div>
<div class="panel"><h2>레인별 설정 (공 / 바닥)</h2><div class="scroll"><table id="params"></table></div>
  <p class="sub" style="margin:8px 0 0">"default" = 해당 프로젝트 코드에서 설정하지 않아 Isaac Gym 기본값 사용
  (linear_damping 0.0, angular_damping 0.5, max_angular_velocity 64). rolling_friction / compliance는 PhysX에서 효과 없음.</p></div>
</div>
<script>
const D = __DATA__;
const F = D.fifa, L = D.lanes, FR = D.frames, N = FR.t.length, FPS = D.sim.fps;
const T_END = FR.t[N - 1];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const RED = '#e5322d', ORANGE = '#f08c00';
const RS = F.rebound.steel, RT = F.rebound.turf;
document.getElementById('sub').textContent =
  `낙하 시험: 공 바닥 ${F.drop_height.toFixed(2)} m에서 강철판(기준 ${RS.q[0]}~${RS.q[1]} m)과 각 레인 바닥(인조잔디 기준 ${RT.q[0]}~${RT.q[1]} m, Pro ${RT.pro[0]}~${RT.pro[1]} m) 위로 각각 낙하, 첫 리바운드 높이  ·  ` +
  `굴림 시험: ${F.ramp_height} m 경사로 출발 속도 ${F.roll_v0} m/s로 구르기 시작, 정지 거리 (FIFA ${F.roll_q[0]}~${F.roll_q[1]} m, Pro ~${F.roll_pro[1]} m)  ·  ` +
  `Isaac Gym PhysX, dt ${D.sim.dt}, substeps ${D.sim.substeps}, 시뮬레이션 ${D.sim.t_end.toFixed(1)} s`;
document.getElementById('dropTitle_steel').textContent =
  `①-a 리바운드 시험: 강철판, 공 자체 규격 (FIFA Quality Pro 공 ${RS.q[0]}~${RS.q[1]} m, 옆에서 본 모습)`;
document.getElementById('dropTitle_turf').textContent =
  `①-b 리바운드 시험: 각 레인 바닥, FIFA 인조잔디 규격 (Quality ${RT.q[0]}~${RT.q[1]} m, Quality Pro ${RT.pro[0]}~${RT.pro[1]} m, 옆에서 본 모습)`;
document.getElementById('rollTitle').textContent = '② 굴림 시험 (위에서 본 모습)';

let t = 0, playing = false, speed = 2, last = null;
const frameAt = tt => Math.max(0, Math.min(N - 1, Math.round(tt * FPS)));
const verdictClass = v => !v || v === '-' ? '' : v.startsWith('OK') ? 'ok' : (v.startsWith('still') ? 'warn' : 'bad');
const KO = {'LOW':'낮음 (기준 미달)','HIGH':'높음 (기준 초과)','OK (Pro)':'통과 (Pro)','OK (Quality only)':'통과 (Quality만)',
  'SHORT':'짧음 (기준 미달)','LONG':'김 (기준 초과)','LONG (still rolling)':'김 (아직 구르는 중)','still rolling':'아직 구르는 중',
  'LONG (off lane)':'김 (레인 이탈)','-':'-'};
const ko = v => KO[v] || v;

function setupCanvas(cv, cssH) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth;
  cv.style.height = cssH + 'px'; cv.width = Math.round(w * dpr); cv.height = Math.round(cssH * dpr);
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); return [g, w, cssH];
}
function text(g, s, x, y, color, align = 'left', size = 12, weight = 400) {
  g.fillStyle = color; g.font = `${weight} ${size}px system-ui,"Noto Sans KR",sans-serif`; g.textAlign = align; g.fillText(s, x, y);
}
function apexesUntil(lane, tt, surface) {  // apexes already reached at time tt, from the event log
  return D.events.filter(e => e.lane === lane.name && e.kind === surface.toUpperCase() && e.t <= tt && e.msg.startsWith('rebound #'))
    .map(e => parseFloat(e.msg.split('=')[1]));
}

function drawDrop(surface) {
  const C = F.rebound[surface], DR = FR['drop_' + surface];
  const cv = document.getElementById('drop_' + surface);
  const [g, W, H] = setupCanvas(cv, 380);
  const padL = 46, padB = 58, padT = 14, zMax = F.drop_height + 0.25;
  const plotH = H - padB - padT, y = z => padT + plotH * (1 - z / zMax);
  const colW = (W - padL) / L.length, k = plotH / zMax;
  const fi = frameAt(t);
  g.strokeStyle = css('--grid'); g.lineWidth = 1;
  for (let z = 0; z <= zMax + 1e-6; z += 0.25) {
    g.beginPath(); g.moveTo(padL, y(z)); g.lineTo(W, y(z)); g.stroke();
    text(g, z.toFixed(2), padL - 6, y(z) + 4, css('--muted'), 'right', 11);
  }
  text(g, 'm', padL - 6, padT - 2, css('--muted'), 'right', 11);
  L.forEach((lane, i) => {
    const x0 = padL + i * colW, cx = x0 + colW / 2, bw = Math.min(colW * 0.7, 150);
    g.fillStyle = css('--pro'); g.globalAlpha = 0.35;
    g.fillRect(cx - bw / 2, y(C.pro[1]), bw, y(C.pro[0]) - y(C.pro[1]));
    g.fillStyle = css('--q');
    g.fillRect(cx - bw / 2, y(C.q[1]), bw, y(C.pro[1]) - y(C.q[1]));
    g.globalAlpha = 1;
    g.strokeStyle = css('--q'); g.lineWidth = 2;
    (C.extra_ticks || []).forEach(z => { g.beginPath(); g.moveTo(cx - bw / 2, y(z)); g.lineTo(cx + bw / 2, y(z)); g.stroke(); });
    g.strokeStyle = css('--start'); g.setLineDash([5, 4]); g.lineWidth = 1.2;
    g.beginPath(); g.moveTo(cx - bw / 2, y(F.drop_height)); g.lineTo(cx + bw / 2, y(F.drop_height)); g.stroke(); g.setLineDash([]);
    g.fillStyle = css('--ground'); g.fillRect(x0 + 6, y(0), colW - 12, 5);
    const ap = apexesUntil(lane, t, surface);
    if (ap.length) {
      g.strokeStyle = RED; g.lineWidth = 2.5; g.beginPath(); g.moveTo(cx - bw / 2 - 6, y(ap[0])); g.lineTo(cx + bw / 2 + 6, y(ap[0])); g.stroke();
      text(g, ap[0].toFixed(3) + ' m', cx + bw / 2 + 4, y(ap[0]) - 5, RED, 'right', 11, 600);
      if (ap.length > 1) { g.strokeStyle = ORANGE; g.lineWidth = 1.5; g.beginPath(); g.moveTo(cx - bw / 2, y(ap[ap.length - 1])); g.lineTo(cx + bw / 2, y(ap[ap.length - 1])); g.stroke(); }
    }
    const zb = DR[fi][i], r = lane.radius * k;
    g.fillStyle = lane.color; g.strokeStyle = css('--text'); g.lineWidth = 1;
    g.beginPath(); g.arc(cx, y(zb) - r, r, 0, 2 * Math.PI); g.fill(); g.stroke();
    text(g, lane.label, cx, H - padB + 22, css('--text'), 'center', 12, 600);
    text(g, `현재 ${zb.toFixed(3)} m` + (ap.length ? `  ·  1차 ${ap[0].toFixed(3)} m` : ''), cx, H - padB + 38, css('--muted'), 'center', 11);
    if (ap.length) { const v = lane.rebound[surface].verdict; text(g, ko(v), cx, H - padB + 53, css(verdictClass(v) === 'ok' ? '--good' : '--bad'), 'center', 11, 600); }
  });
}

const X_MAX = Math.min(56, Math.max(12, Math.ceil((Math.max(...L.map(l => l.roll_dist)) + 3) / 5) * 5));
function drawRoll() {
  const cv = document.getElementById('roll');
  const rowH = 54, padT = 24, labelW = Math.min(170, document.getElementById('roll').clientWidth * 0.3);
  const [g, W, H] = setupCanvas(cv, padT + rowH * L.length + 6);
  const padR = 14, plotW = W - labelW - padR, x = d => labelW + plotW * d / X_MAX;
  const fi = frameAt(t);
  for (let d = 0; d <= X_MAX; d += 5) {
    g.strokeStyle = css('--grid'); g.lineWidth = 1; g.beginPath(); g.moveTo(x(d), padT - 4); g.lineTo(x(d), H); g.stroke();
    text(g, d + ' m', x(d), padT - 8, css('--muted'), 'center', 11);
  }
  L.forEach((lane, i) => {
    const y0 = padT + i * rowH, cy = y0 + rowH / 2;
    g.fillStyle = css('--ground'); g.globalAlpha = 0.5; g.fillRect(x(0), y0 + 8, plotW, rowH - 16); g.globalAlpha = 1;
    g.fillStyle = css('--pro'); g.globalAlpha = 0.6; g.fillRect(x(F.roll_pro[0]), y0 + 8, x(F.roll_pro[1]) - x(F.roll_pro[0]), rowH - 16);
    g.fillStyle = css('--q'); g.fillRect(x(F.roll_pro[1]), y0 + 8, x(F.roll_q[1]) - x(F.roll_pro[1]), rowH - 16); g.globalAlpha = 1;
    g.strokeStyle = css('--start'); g.lineWidth = 2; g.beginPath(); g.moveTo(x(0), y0 + 6); g.lineTo(x(0), y0 + rowH - 6); g.stroke();
    const d = FR.roll[fi][i], v = FR.speed[fi][i];
    const stopped = lane.roll_stopped && t >= lane.roll_stop_t;
    if (stopped) { g.strokeStyle = RED; g.lineWidth = 2.5; g.beginPath(); g.moveTo(x(lane.roll_dist), y0 + 4); g.lineTo(x(lane.roll_dist), y0 + rowH - 4); g.stroke(); }
    const r = Math.max(6, lane.radius * plotW / X_MAX), bx = x(Math.min(d, X_MAX)), ang = d / lane.radius;
    g.fillStyle = lane.color; g.strokeStyle = css('--text'); g.lineWidth = 1;
    g.beginPath(); g.arc(bx, cy, r, 0, 2 * Math.PI); g.fill(); g.stroke();
    g.beginPath(); g.moveTo(bx, cy); g.lineTo(bx + r * Math.cos(ang), cy + r * Math.sin(ang)); g.stroke();
    text(g, lane.label, 6, cy - 3, css('--text'), 'left', 12, 600);
    const info = stopped ? `정지 ${lane.roll_dist.toFixed(2)} m · ${ko(lane.roll_verdict)}` : `${d.toFixed(2)} m · ${v.toFixed(2)} m/s`;
    text(g, info, 6, cy + 13, stopped ? css(verdictClass(lane.roll_verdict) === 'ok' ? '--good' : '--bad') : css('--muted'), 'left', 11, stopped ? 600 : 400);
  });
}

function buildTables() {
  const s = document.getElementById('summary');
  const SN = {steel: '강철판', turf: '바닥(인조잔디 기준)'};
  const rebCells = (l, s) => { const r = l.rebound[s];
    return `<td class="num">${r.h1 == null ? '-' : r.h1.toFixed(3) + ' m'}</td>
      <td class="num">${r.h1 == null ? '-' : Math.sqrt(r.h1 / F.drop_height).toFixed(2)}</td>
      <td class="${verdictClass(r.verdict)}">${ko(r.verdict)}</td>`; };
  s.innerHTML = '<tr><th>설정</th><th>질량</th>' +
    F.surfaces.map(sf => `<th>${SN[sf]} 첫 리바운드</th><th>반발계수</th><th>판정</th>`).join('') +
    '<th>굴림 거리</th><th>정지 시각</th><th>굴림 판정</th><th>제자리 회전 정지</th></tr>' +
    L.map(l => `<tr><td><span class="dot" style="background:${l.color}"></span>${l.label}</td>
      <td class="num">${l.mass.toFixed(3)} kg</td>
      ${F.surfaces.map(sf => rebCells(l, sf)).join('')}
      <td class="num">${l.roll_dist.toFixed(2)} m${l.roll_stopped ? '' : '+'}</td>
      <td class="num">${l.roll_stopped ? l.roll_stop_t.toFixed(1) + ' s' : '-'}</td>
      <td class="${verdictClass(l.roll_verdict)}">${ko(l.roll_verdict)}</td>
      <td class="num">${l.spin_stop_t == null ? `> ${D.sim.t_end.toFixed(0)} s` : l.spin_stop_t.toFixed(2) + ' s'}</td></tr>`).join('') +
    `<tr><td colspan="${6 + 3 * F.surfaces.length}" style="color:var(--muted)">제자리 회전: ${D.sim.spin_w0} rad/s에서 ${D.sim.spin_stop} rad/s 미만이 될 때까지 · MuJoCo 방식 torsion/rolling 마찰 ${D.sim.mj_friction ? 'ON' : 'OFF'} · FIFA 기준: 리바운드 강철판 ${RS.q[0]}~${RS.q[1]} m, 인조잔디 ${RT.q[0]}~${RT.q[1]} m (Pro ${RT.pro[0]}~${RT.pro[1]} m), 굴림 ${F.roll_q[0]}~${F.roll_q[1]} m (Pro ~${F.roll_pro[1]} m). "+" = 시뮬레이션 종료 시점까지 멈추지 않음</td></tr>`;
  const keys = ['linear_damping','angular_damping','max_angular_velocity','friction','restitution','torsion_friction','rolling_friction','compliance','ground_friction','ground_restitution','steel_pad_restitution'];
  const p = document.getElementById('params');
  p.innerHTML = '<tr><th>설정</th><th>출처</th><th>반지름</th><th>질량</th>' + keys.map(k => `<th>${k}</th>`).join('') + '</tr>' +
    L.map(l => `<tr><td><span class="dot" style="background:${l.color}"></span>${l.label}</td><td>${l.src}</td>
      <td class="num">${l.radius}</td><td class="num">${l.mass.toFixed(3)}</td>` +
      keys.map(k => { const v = k in l.params ? l.params[k] : l[k]; return `<td class="num">${v == null ? 'default' : v}</td>`; }).join('') + '</tr>').join('');
}

const logEl = document.getElementById('log');
const label = Object.fromEntries(L.map(l => [l.name, l.label]));
logEl.innerHTML = D.events.map((e, i) => `<div data-i="${i}">[t=${e.t.toFixed(2).padStart(6)}s] ${label[e.lane].padEnd(24)} ${e.kind.padEnd(5)} ${e.msg.replace(/</g,'&lt;')}</div>`).join('');
let lastShown = -2;
function updateLog() {
  let cur = -1;
  D.events.forEach((e, i) => { if (e.t <= t) cur = i; });
  if (cur === lastShown) return;
  lastShown = cur;
  logEl.querySelectorAll('div').forEach((el, i) => { el.className = i > cur ? 'future' : (i === cur ? 'now' : ''); });
  const el = logEl.querySelector(`div[data-i="${Math.max(cur, 0)}"]`);
  if (el) logEl.scrollTop = Math.max(0, el.offsetTop - logEl.offsetTop - logEl.clientHeight + 40);
}

const playBtn = document.getElementById('play'), scrub = document.getElementById('scrub'), timeEl = document.getElementById('time');
function render() {
  F.surfaces.forEach(drawDrop); drawRoll(); updateLog();
  scrub.value = Math.round(1000 * t / T_END);
  timeEl.textContent = `${t.toFixed(2)} / ${T_END.toFixed(1)} s`;
}
function tick(now) {
  if (playing) {
    if (last != null) t = Math.min(T_END, t + (now - last) / 1000 * speed);
    last = now;
    if (t >= T_END) { playing = false; playBtn.textContent = '▶ 재생'; }
    render();
  }
  requestAnimationFrame(tick);
}
playBtn.onclick = () => { if (t >= T_END) t = 0; playing = !playing; last = null; playBtn.textContent = playing ? '❚❚ 일시정지' : '▶ 재생'; };
document.getElementById('restart').onclick = () => { t = 0; last = null; render(); };
document.getElementById('speed').onchange = e => { speed = parseFloat(e.target.value); };
scrub.oninput = () => { t = T_END * scrub.value / 1000; last = null; render(); };
window.addEventListener('resize', render);
buildTables(); render(); requestAnimationFrame(tick);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    ap.add_argument("--headless", action="store_true", help="run without the Isaac Gym viewer (log + HTML only)")
    ap.add_argument("--mj_friction", action="store_true",
                    help="add MuJoCo-style torsional/rolling contact friction to the balls (off: PhysX only)")
    ap.add_argument("--html", default="output/ball_fifa_test/ball_fifa_test.html", help="HTML replay output path")
    ap.add_argument("--record_fps", type=int, default=30, help="frames per sim second stored in the HTML replay")
    ap.add_argument("--fast", type=int, default=1, help="viewer: sim steps per rendered frame (speed-up factor)")
    ap.add_argument("--max_time", type=float, default=120.0, help="sim seconds before the summary is forced")
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--device", type=int, default=0,
                    help="CUDA device index for physics, counted among visible GPUs (pick the GPU with CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--graphics_device", type=int, default=0)
    ap.add_argument("--exit_on_done", action="store_true")
    args = ap.parse_args()
    global MJ_ON
    MJ_ON = args.mj_friction
    n_gpu = torch.cuda.device_count()
    if args.device >= n_gpu:
        raise SystemExit(f"--device {args.device} is not available: only {n_gpu} visible GPU(s) "
                         f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}). "
                         f"With CUDA_VISIBLE_DEVICES set, the chosen GPU is --device 0.")

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 1.0 / 60.0
    sp.substeps = args.substeps
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.use_gpu_pipeline = True
    sp.physx.use_gpu = True
    sp.physx.num_threads = 4
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 4
    sp.physx.num_velocity_iterations = 0
    sp.physx.contact_offset = 0.02
    sp.physx.rest_offset = 0.0
    sp.physx.bounce_threshold_velocity = 0.2
    sp.physx.max_depenetration_velocity = 10.0
    sp.physx.default_buffer_size_multiplier = 10.0
    graphics = -1 if args.headless else args.graphics_device
    sim = gym.create_sim(args.device, graphics, gymapi.SIM_PHYSX, sp)
    assert sim is not None, "failed to create sim"

    # --- assets ---
    def box_asset(sx, sy, sz):
        o = gymapi.AssetOptions()
        o.fix_base_link = True
        o.disable_gravity = True
        return gym.create_box(sim, sx, sy, sz, o)

    ball_assets, ground_assets, pad_assets = {}, {}, {}
    urdf_dir = tempfile.mkdtemp(prefix="ball_fifa_test_")
    for name in args.configs:
        c = CONFIGS[name]
        o = gymapi.AssetOptions()
        o.density = c["density"]
        for k in ("linear_damping", "angular_damping", "max_angular_velocity"):
            if c[k] is not None:
                setattr(o, k, c[k])
        if c.get("inertia_ratio") is None:
            a = gym.create_sphere(sim, c["radius"], o)    # solid ball: I = 0.4 m r^2
        else:
            # Explicit inertia (e.g. 2/3 = hollow shell) must be set on the asset: overriding it per actor
            # with set_actor_rigid_body_properties crashes GPU PhysX's narrowphase in Isaac Gym.
            a = gym.load_asset(sim, urdf_dir, write_ball_urdf(urdf_dir, name, c), o)
        sh = gym.get_asset_rigid_shape_properties(a)
        for k in ("friction", "rolling_friction", "torsion_friction", "restitution", "compliance"):
            if c[k] is not None:
                setattr(sh[0], k, c[k])
        gym.set_asset_rigid_shape_properties(a, sh)
        ball_assets[name] = a

        g = box_asset(GROUND_X[1] - GROUND_X[0], GROUND_Y[1] - GROUND_Y[0], 0.2)
        gsh = gym.get_asset_rigid_shape_properties(g)
        gsh[0].friction = c["ground_friction"]
        gsh[0].restitution = c["ground_restitution"]
        gym.set_asset_rigid_shape_properties(g, gsh)
        ground_assets[name] = g

        for s in DROP_SURFACES:
            pad = box_asset(DROP_PAD_X[1] - DROP_PAD_X[0], DROP_PAD_Y[s][1] - DROP_PAD_Y[s][0], 0.2)
            psh = gym.get_asset_rigid_shape_properties(pad)
            psh[0].friction = c["ground_friction"]
            psh[0].restitution = pad_restitution(c, s)
            gym.set_asset_rigid_shape_properties(pad, psh)
            pad_assets[name, s] = pad

    pole_w = 0.05
    tick = box_asset(0.7, 0.02, 0.012)
    pole_pro, pole_q = {}, {}
    for s in DROP_SURFACES:
        q, pro = REBOUND_CRITERIA[s]["q"], REBOUND_CRITERIA[s]["pro"]
        pole_pro[s] = box_asset(pole_w, pole_w, pro[1] - pro[0])
        pole_q[s] = box_asset(pole_w, pole_w, q[1] - pro[1]) if q[1] > pro[1] else None
    pole_base = box_asset(pole_w * 0.6, pole_w * 0.6, DROP_HEIGHT + 0.1)
    strip_w = 0.6
    strip_pro = box_asset(ROLL_PRO[1] - ROLL_PRO[0], strip_w, 0.004)
    strip_q = box_asset(ROLL_Q[1] - ROLL_PRO[1], strip_w, 0.004)
    roll_line = box_asset(0.04, strip_w + 0.2, 0.008)

    GREEN_DARK, GREEN_LIGHT = gymapi.Vec3(0.05, 0.55, 0.15), gymapi.Vec3(0.55, 0.95, 0.45)
    WHITE, GRAY = gymapi.Vec3(1, 1, 1), gymapi.Vec3(0.45, 0.45, 0.45)

    def add_marker(env, asset, x, y, z, color, group):
        p = gymapi.Transform()
        p.p = gymapi.Vec3(x, y, z)
        h = gym.create_actor(env, asset, p, "marker", group, MARKER_FILTER, 0)
        gym.set_rigid_body_color(env, h, 0, gymapi.MESH_VISUAL, color)

    # --- one env, one lane per config along +y (collision group = lane index) ---
    lanes, ball_ids, ball_handles = [], [], []
    mj_actor_ids, mj_body_ids, mj_mu_t, mj_mu_r, mj_inertia, mj_mass, mj_radius = [], [], [], [], [], [], []
    n_lanes = len(args.configs)
    env = gym.create_env(sim, gymapi.Vec3(DROP_PAD_X[0], -2.0, 0.0),
                         gymapi.Vec3(GROUND_X[1], (n_lanes - 1) * LANE_SPACING + 2.0, 3.0), 1)
    for i, name in enumerate(args.configs):
        c = CONFIGS[name]
        ly = i * LANE_SPACING
        lanes.append(Lane(name, c))
        col = gymapi.Vec3(*c["color"])

        gp = gymapi.Transform()
        gp.p = gymapi.Vec3(0.5 * (GROUND_X[0] + GROUND_X[1]), ly, -0.1)
        gh = gym.create_actor(env, ground_assets[name], gp, f"{name}_ground", i, GROUND_FILTER, 0)
        for s in DROP_SURFACES:
            pp = gymapi.Transform()
            pp.p = gymapi.Vec3(0.5 * (DROP_PAD_X[0] + DROP_PAD_X[1]), ly + 0.5 * (DROP_PAD_Y[s][0] + DROP_PAD_Y[s][1]), -0.1)
            ph = gym.create_actor(env, pad_assets[name, s], pp, f"{name}_drop_pad_{s}", i, GROUND_FILTER, 0)
            gym.set_rigid_body_color(env, ph, 0, gymapi.MESH_VISUAL,
                                     gymapi.Vec3(0.62, 0.66, 0.74) if s == "steel" else gymapi.Vec3(0.3, 0.3, 0.3))
        gym.set_rigid_body_color(env, gh, 0, gymapi.MESH_VISUAL,
                                 gymapi.Vec3(0.18 + 0.12 * c["color"][0], 0.18 + 0.12 * c["color"][1],
                                             0.18 + 0.12 * c["color"][2]))

        r = c["radius"]
        ids = []
        _, mu_t, mu_r = mj_pair_friction(c)
        tests = [(f"drop_{s}", DROP_XY[s], DROP_HEIGHT + r) for s in DROP_SURFACES]
        tests += [("roll", ROLL_XY, r), ("spin", SPIN_XY, r)]
        for kind, (x, y), z in tests:
            p = gymapi.Transform()
            p.p = gymapi.Vec3(x, y + ly, z)
            h = gym.create_actor(env, ball_assets[name], p, f"{name}_ball_{kind}", i, BALL_FILTER, 0)
            gym.set_rigid_body_color(env, h, 0, gymapi.MESH_VISUAL, col)
            ids.append(gym.get_actor_index(env, h, gymapi.DOMAIN_SIM))
            if kind == "drop_steel":
                ball_handles.append(h)
            rbp = gym.get_actor_rigid_body_properties(env, h)[0]
            mj_actor_ids.append(ids[-1])
            mj_body_ids.append(gym.get_actor_rigid_body_index(env, h, 0, gymapi.DOMAIN_SIM))
            mj_mu_t.append(mu_t)
            mj_mu_r.append(mu_r)
            mj_inertia.append(rbp.inertia.x.x)
            mj_mass.append(rbp.mass)
            mj_radius.append(r)
        ball_ids.append(ids)

        # rebound markers per pad: pole (dark green = Pro band, light green = Quality only) + height ticks
        for s in DROP_SURFACES:
            q, pro = REBOUND_CRITERIA[s]["q"], REBOUND_CRITERIA[s]["pro"]
            px, py = POLE_XY[s][0], POLE_XY[s][1] + ly
            add_marker(env, pole_base, px, py, 0.5 * (DROP_HEIGHT + 0.1), GRAY, i)
            add_marker(env, pole_pro[s], px, py + 0.001, 0.5 * (pro[0] + pro[1]), GREEN_DARK, i)
            if pole_q[s] is not None:
                add_marker(env, pole_q[s], px, py + 0.001, 0.5 * (pro[1] + q[1]), GREEN_LIGHT, i)
            ticks = {DROP_HEIGHT: WHITE, q[0]: GREEN_DARK, pro[1]: GREEN_DARK}
            if q[1] > pro[1]:
                ticks[q[1]] = GREEN_LIGHT
            for hz in REBOUND_CRITERIA[s]["extra_ticks"]:
                ticks[hz] = GREEN_LIGHT
            for hz, cc in ticks.items():
                add_marker(env, tick, px, py, hz, cc, i)

        # roll markers: start line, Pro strip (4~8 m), Quality-only strip (8~10 m), gray line every 5 m
        rx, ry = ROLL_XY[0], ROLL_XY[1] + ly
        add_marker(env, roll_line, rx, ry, 0.004, WHITE, i)
        add_marker(env, strip_pro, rx + 0.5 * (ROLL_PRO[0] + ROLL_PRO[1]), ry, 0.002, GREEN_DARK, i)
        add_marker(env, strip_q, rx + 0.5 * (ROLL_PRO[1] + ROLL_Q[1]), ry, 0.002, GREEN_LIGHT, i)
        for d in range(5, int(LANE_END) + 1, 5):
            add_marker(env, roll_line, rx + d, ry, 0.004, GRAY, i)

    gym.prepare_sim(sim)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    init_root = root.clone()
    all_ball_ids = torch.tensor([j for ids in ball_ids for j in ids], dtype=torch.int32, device=root.device)
    n_drop = len(DROP_SURFACES)
    drop_idx = {s: torch.tensor([ids[k] for ids in ball_ids], dtype=torch.long, device=root.device)
                for k, s in enumerate(DROP_SURFACES)}
    roll_idx = torch.tensor([ids[n_drop] for ids in ball_ids], dtype=torch.long, device=root.device)
    spin_idx = torch.tensor([ids[n_drop + 1] for ids in ball_ids], dtype=torch.long, device=root.device)
    mjf = MujocoFriction(gym, sim, mj_actor_ids, mj_body_ids, mj_mu_t, mj_mu_r, mj_inertia, mj_mass,
                         mj_radius, sp.dt) if MJ_ON else None
    radii = torch.tensor([CONFIGS[n]["radius"] for n in args.configs], device=root.device)
    start_x = init_root[roll_idx, 0].clone()

    masses = {}
    for h, lane in zip(ball_handles, lanes):
        masses[lane.name] = gym.get_actor_rigid_body_properties(env, h)[0].mass

    def reset_all():
        root[:] = init_root
        root[roll_idx, 7] = ROLL_V0
        root[roll_idx, 11] = ROLL_V0 / radii          # omega_y: rolling without slip toward +x
        root[spin_idx, 12] = SPIN_W0                  # spin in place about z
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root),
                                                gymtorch.unwrap_tensor(all_ball_ids), len(all_ball_ids))
        for lane in lanes:
            lane.reset()
        EVENTS.clear()
        for key in frames:
            frames[key].clear()

    print()
    for s in DROP_SURFACES:
        q, pro = REBOUND_CRITERIA[s]["q"], REBOUND_CRITERIA[s]["pro"]
        print(f"FIFA ball tests: drop from {DROP_HEIGHT:.2f} m onto {REBOUND_CRITERIA[s]['label']} "
              f"(target {q[0]:.2f}~{q[1]:.2f} m" + (f", Pro ~{pro[1]:.2f} m)" if q[1] > pro[1] else ")"))
    print(f"FIFA ball tests: roll from a {RAMP_HEIGHT:.0f} m ramp "
          f"(v0 = {ROLL_V0:.2f} m/s, rolling) | dt 1/60, substeps {args.substeps}, GPU PhysX")
    for i, lane in enumerate(lanes):
        c = lane.cfg
        fmt = lambda v: "default" if v is None else f"{v:g}"
        print(f"  lane {i}: {c['label']:<24} y={i * LANE_SPACING:4.1f} m  color={c['color']}  [{c['src']}]")
        print(f"          r={c['radius']} m, mass={masses[lane.name]:.3f} kg, "
              f"I/(m r^2)={c.get('inertia_ratio') or 0.4:.3f}, lin_damp={fmt(c['linear_damping'])}, "
              f"ang_damp={fmt(c['angular_damping'])}, max_ang_vel={fmt(c['max_angular_velocity'])}, "
              f"friction={c['friction']}, restitution={c['restitution']} | ground friction="
              f"{c['ground_friction']}, restitution={c['ground_restitution']} | drop pad restitution "
              + ", ".join(f"{s} {pad_restitution(c, s)} (-> pair {0.5 * (c['restitution'] + pad_restitution(c, s)):.2f})"
                          for s in DROP_SURFACES))
        if MJ_ON:
            _, mu_t, mu_r = mj_pair_friction(c)
            print(f"          MuJoCo friction (max of ball/ground): torsional {mu_t:g} m, rolling {mu_r:g} m")
    print("  (Isaac Gym rolling_friction / torsion_friction / compliance have no effect in PhysX)")
    print(f"  MuJoCo-style torsional/rolling friction: {'ON' if MJ_ON else 'OFF'} (--mj_friction)\n")

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        assert viewer is not None, "failed to create viewer (DISPLAY / X11 / Vulkan?) -- run with --headless"
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "pause")
        my = 0.5 * (n_lanes - 1) * LANE_SPACING
        span = (n_lanes - 1) * LANE_SPACING + 4.0
        gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(-9.0, my - 0.9 * span, 3.5 + 0.25 * span),
                                  gymapi.Vec3(4.0, my, 0.5))

    frames = dict(t=[], roll=[], speed=[], **{f"drop_{s}": [] for s in DROP_SURFACES})
    record_every = max(1, int(round(1.0 / (sp.dt * args.record_fps))))

    def record(t):
        x = root[roll_idx, 0] - start_x
        v = torch.norm(root[roll_idx, 7:10], dim=-1)
        frames["t"].append(round(t, 4))
        for s in DROP_SURFACES:
            d = root[drop_idx[s], 2] - radii
            frames[f"drop_{s}"].append([round(float(a), 4) for a in d])
        frames["roll"].append([round(float(a), 4) for a in x])
        frames["speed"].append([round(float(a), 4) for a in v])

    reset_all()
    t, step, paused, summarized = 0.0, 0, False, False
    record(t)

    def draw_lines():
        gym.clear_lines(viewer)
        verts, cols = [], []
        spin_state = root[spin_idx, :7].cpu().numpy()
        for i, lane in enumerate(lanes):
            ly = i * LANE_SPACING
            px_, py_, pz_, qx, qy, qz, qw = spin_state[i]
            hx, hy = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz)   # body x axis in world
            nrm = max(math.hypot(hx, hy), 1e-6)
            verts += [[px_, py_, pz_ + 0.12], [px_ + 0.35 * hx / nrm, py_ + 0.35 * hy / nrm, pz_ + 0.12]]
            cols.append([1.0, 1.0, 1.0])
            for s, drop in lane.drops.items():
                x0, y0 = DROP_XY[s][0], DROP_XY[s][1] + ly
                if drop.apexes:
                    verts += [[x0 - 0.4, y0, drop.apexes[0]], [x0 + 0.4, y0 + 0.3, drop.apexes[0]]]
                    cols.append([1.0, 0.1, 0.1])
                    if len(drop.apexes) > 1:
                        verts += [[x0 - 0.3, y0, drop.apexes[-1]], [x0 + 0.3, y0 + 0.3, drop.apexes[-1]]]
                        cols.append([1.0, 0.6, 0.1])
            if lane.roll_stopped:
                xs, yr = ROLL_XY[0] + lane.roll_dist, ROLL_XY[1] + ly
                for dy in (-0.4, 0.4):
                    verts += [[xs, yr + dy, 0.0], [xs, yr + dy, 0.8]]
                    cols.append([1.0, 0.1, 0.1])
                verts += [[xs, yr - 0.4, 0.8], [xs, yr + 0.4, 0.8]]
                cols.append([1.0, 0.1, 0.1])
        if cols:
            gym.add_lines(viewer, env, len(cols), np.array(verts, dtype=np.float32).reshape(-1),
                          np.array(cols, dtype=np.float32).reshape(-1))

    while True:
        if viewer is not None:
            if gym.query_viewer_has_closed(viewer):
                break
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value > 0 and evt.action == "reset":
                    reset_all()
                    t, step, summarized = 0.0, 0, False
                    record(t)
                    print("\n----- restart -----\n")
                elif evt.value > 0 and evt.action == "pause":
                    paused = not paused

        n_steps = 0 if (paused or summarized) else (args.fast if viewer is not None else 60)
        for _ in range(n_steps):
            if mjf is not None:
                mjf.apply(root)
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.refresh_actor_root_state_tensor(sim)
            t += sp.dt
            step += 1
            if step % record_every == 0:
                record(t)
            d = {s: root[drop_idx[s]].cpu().numpy() for s in DROP_SURFACES}
            rl = root[roll_idx].cpu().numpy()
            dist = (root[roll_idx, 0] - start_x).cpu().numpy()
            wz = root[spin_idx, 12].cpu().numpy()
            for i, lane in enumerate(lanes):
                r = lane.cfg["radius"]
                for s in DROP_SURFACES:
                    update_drop(lane, lane.drops[s], t, float(d[s][i, 2] - r), float(d[s][i, 9]))
                update_roll(lane, t, float(dist[i]), float(np.linalg.norm(rl[i, 7:10])))
                update_spin(lane, t, float(wz[i]))
            if not summarized and (all(l.done for l in lanes) or t >= args.max_time):
                print_summary(lanes, t, masses)
                if args.html:
                    write_html(args.html, lanes, masses, frames, round(1.0 / (sp.dt * record_every)),
                               t, args.substeps)
                summarized = True
                break

        if viewer is not None:
            gym.step_graphics(sim)
            draw_lines()
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        if summarized and (viewer is None or args.exit_on_done):
            break

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
