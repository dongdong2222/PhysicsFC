# Trap_Fix 계획 — HumanoidTrapping_3 를 고친 트래핑 태스크

이 문서는 아직 계획 단계다. 구현은 이 계획(특히 §6 결정 필요 항목)을 확인받은 뒤 진행한다.

- 대상 원본: [humanoid_trapping_3.py](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py), [humanoid_trapping_3.yaml](calm/data/cfg/soccer_hlc/humanoid_trapping_3.yaml)
- 새로 만들 것 (Dribble_Fix 와 같은 방식, 원본은 건드리지 않음)
  - `calm/env/tasks/soccer_hlc/humanoid_trap_fix.py` — `HumanoidTrap_Fix(BallPhysicsMixin, HumanoidTrapping_3)`
  - `calm/data/cfg/soccer_hlc/humanoid_trap_fix.yaml`
  - [parse_task.py](calm/utils/parse_task.py) 에 import 1줄, [run_train_hlc.sh](tools/run_train_hlc.sh) 에 주석 블록, [train_config.md](train_config.md) 변경 이력

---

## 1. 현재 Trapping_3 동작 정리 (코드 기준)

### 1.1 한 에피소드의 흐름

1. 리셋: [_reset_actors:683](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L683) 에서 `ratios = [1.0, .0]` 로 env 를 공중볼 / 땅볼로 나눈다 → 지금은 전부 공중볼.
   - 나누는 방식이 `sizes = int(r * n)` 이라, 리셋 배치가 1개면 `int(0.5) = 0` 이 되어 항상 마지막 그룹(땅볼)으로 간다. 비율을 바꾸면 작은 배치에서 쏠린다.
   - 캐릭터는 `IS_INIT = True` 라 `blackhole/init_states/move_ball_state.pkl` 의 상태에서 시작한다 (root xy 는 0 으로, 속도는 그대로).
2. `_reset_task` 에서 `_touched`, `_counter` 등을 0 으로. `_reset_task` 는 heading change step(100–200)마다도 불리지만 에피소드가 60 step 이라 사실상 리셋 때만 불린다.
3. 매 step `_compute_reward` 에서 접촉 판정(`_touched`) → 보상, `_compute_reset` 에서 종료 판정.
4. `episodeLength: 60` (control 30 Hz) = **2.0 s** 에서 timeout.

### 1.2 공중볼 초기화 `_reset_ball_new_2` ([753-845](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L753-L845))

| 단계 | 내용 |
|---|---|
| 공 속도 | v ∈ [10, 15] m/s |
| 착지점 | 캐릭터 기준 전 방향, 거리 U(0, `2 + 2·ratio`) (ratio = 속도 정규화) + `root_vel · 0.5` |
| 발사각 | φ ∈ [15, 60]° |
| 진공 포물선 | `T = 2 v sinφ / g`, `R = v cosφ T`. 착지점에서 랜덤 방향 θ 로 R 떨어진 곳 (z = 0.115) 에서 착지점을 향해 발사 |
| 감쇠 보정 | `speed *= 1 + 0.025 T` (공 linear damping 0.1 에 대한 경험식) |
| 각속도 | 리셋하지 않음 (이전 에피소드 값이 남음) |
| 목표 부위 | 6 개 one-hot 중 랜덤: [torso, head, r_shin, r_foot, l_shin, l_foot] (body 1, 2, 10, 11, 13, 14) |

범위: T 0.53–2.65 s, R 5.1–22.9 m. **T 가 2 s 를 넘는 경우(v·sinφ > 9.8)가 있어 공이 떨어지기 전에 timeout 된다.**

감쇠 보정의 정확도 (수치 시뮬레이션, PhysX 식 `v *= 1 − c·h`, h = 1/120 가정):

| v, φ | 감쇠 0 | 0.1, 보정 없음 | 0.1 + 현재 보정 | 0.1 + 해석해 |
|---|---|---|---|---|
| 10, 45° | −0.06 m | −0.96 | −0.31 | −0.06 |
| 15, 45° | −0.09 | −3.01 | −0.95 | −0.09 |
| 15, 60° | −0.06 | −3.09 | −0.98 | −0.06 |

- 현재 보정은 필요한 양의 1/3 정도라 체공이 긴 공은 최대 약 1 m 짧게 떨어진다.
- 감쇠 0 에서도 남는 −4~−12 cm 는 semi-implicit Euler 적분 때문이다: 매 substep 속도를 먼저 갱신해 연속 포물선보다 `½ g h t` 만큼 낮게 가고, 약 1 substep 일찍 착지한다 (`−v cosφ · h`).

### 1.3 땅볼 초기화 `_reset_ball_new_ground` ([883-963](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L883-L963), 현재 비활성)

| 단계 | 내용 |
|---|---|
| 공 속도 | v ∈ [5, 13] |
| 착지점 (`fall_pos`) | 공중볼과 같은 방식. 땅볼에서는 멈추는 곳이 아니라 **지나가는 점** |
| 발사각 | φ ∈ [0, 1]° (vz ≤ 0.23 m/s, 사실상 수평) |
| 발사 거리 | `clip(v + U(−0.5, 0.5), 2)` — 포물선이 아닌 경험식 ("감속이 없으면 약 1 s 뒤 통과") |
| T, 감쇠 보정 | T ≤ 0.046 s 라 보정은 0.1 % 이하, 사실상 효과 없음 |
| 마찰 감속 | 고려하지 않음. 실제로는 처음에 미끄러지다가 (μ = 0.6, 공 0.2 + 바닥 1.0) 약 5/7 속도에서 구르기 시작 |
| 목표 부위 | env 앞 절반 r_foot, 뒤 절반 l_foot (`n // 2` 라 배치가 1이면 항상 l_foot) |
| 기타 | `_launch_direc` 을 정규화하지 않음 (공중볼은 정규화), 각속도 리셋 안 함 |

### 1.4 관측 (`IS_SIMPLE`, `IS_TARGET_PART`)

`compute_heading_observations_simple_onehot`: heading 기준 local 공 위치(z 는 절대 높이) 3 + 공 속도 3 + 공 각속도 3 + 목표 부위 one-hot 6 = **15**. 착지점은 관측에 없다.

### 1.5 보상 `compute_heading_reward` ([1210-1271](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L1210-L1271))

```
reward = exp(-5 · |rb_pos[목표 부위] - ball_pos|)        # 3D, 접촉 전후 동일
reward[facing · (-launch_direc) < 0.5] = 0               # 공이 오는 방향 ±60° 를 보지 않으면 0
```

- 거리 기준이 **body 원점(관절)** 이다. foot = 발목, shin = 무릎, torso = 허리 위(가슴 구 중심보다 0.12 m 아래), head = 목. 정확히 접촉해도 거리가 0 이 안 되고, 특히 shin 은 무릎까지 거리라 정강이 중간으로 받으면 오히려 손해다.
- 3D 거리라 공이 멀 때 (d > 1 m → 0.007 이하) 신호가 거의 없다. 착지점으로 미리 이동하라는 항이 없다.
- facing 은 hard cut 이라 각도 차이가 커지면 gradient 가 없다. 땅볼은 `launch_direc` 이 정규화되지 않아 cut 이 cos > 0.04–0.11 로 사실상 꺼진다.
- 접촉 후 보상(`fall_pos_reward`)은 주석 처리돼 있어 접촉 후에도 같은 식이다. `_target_fall_pos` 를 접촉 순간 랜덤으로 정하지만 (`IS_TARGET_FALL = False` 라) 관측·보상 어디에도 쓰이지 않는다.
- `print('reward:', ...)` 가 매 step 실행된다 ([648](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L648)): 로그 폭주 + GPU 동기화.

### 1.6 종료 `compute_humanoid_reset` ([1274-1335](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L1274-L1335))

`contactBodies` = [r_foot, l_foot, r_thigh, r_shin, l_thigh, l_shin]. 이 외 body (pelvis, torso, head, 팔 6개) 를 "비접촉 body" 라 부른다. body 번호는 0 pelvis, 1 torso, 2 head, 3–5 오른팔(상완, 전완, 손), 6–8 왼팔, 9 r_thigh, 10 r_shin, 11 r_foot, 12 l_thigh, 13 l_shin, 14 l_foot.

| 조건 | 식 | 대상 | 문제 |
|---|---|---|---|
| 넘어짐 `has_fallen` | 비접촉 body 에 힘 > 0.1 **그리고** 비접촉 body 높이 < 0.15 | 둘 다 | 없음 (유지) |
| `tar_fail` | 공 xy 힘 > 1 **그리고** 비접촉 body 에 힘 > 1 | 둘 다 | torso / head 가 목표인데 비접촉 body 라 목표로 받아도 실패. 공과 다른 body 의 접촉을 구분 못 함 (손이 땅에 닿은 상태에서 발로 받아도 실패) |
| `contact` | 비접촉 body 에 힘 > 0.1 | 둘 다 | 위의 상위 집합 (임계 0.1 < 1). 결과적으로 **torso / head 목표(공중볼의 1/3)는 받는 순간 무조건 실패**. 손 접촉을 막는 목적과도 다름 (pelvis 도 실패) |
| `ball_fall` | 공 중심 z < 0.13 | 공중 | 착지 전 판정으로는 맞음. 단 트래핑 후 공이 땅에 내려와도 실패로 처리됨 |
| `touched >= 2` | 접촉 후 10 step | 둘 다 | 아래 1.7 |
| 땅볼 통과 | 이동 거리 > `initial_dist × 1.2` | 땅 | 발사 거리가 길면 착지점을 한참 지나야 끝남 |

`has_fallen` 과 `tar_fail` 은 `contact` 에 다 포함되므로 실제로 작동하는 것은 `contact`, `ball_fall`, `touched >= 2`, 땅볼 통과, timeout 이다.

### 1.7 트래핑 판단 ([598-602](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L598-L602))

```
is_contact = (공의 net contact force > 0.1) and (공 높이 > 0.2)
touched 0 -> 1 (접촉), 이후 매 step counter++, counter >= 10 -> touched = 2 -> 종료(실패로 기록)
```

- **어느 body 에 닿았는지 보지 않는다.** 공중볼이 아무 body 에 닿기만 하면 10 step 뒤 종료. 받았는지(공이 통제되는지)도 보지 않는다.
- **땅볼은 판정이 안 된다.** 공 중심이 0.11 이라 `공 높이 > 0.2` 를 영원히 만족 못 한다. 또 굴러가는 공은 바닥 수직항력이 net contact force 에 항상 섞여서 힘으로는 사람 접촉을 구분할 수 없다. 땅볼은 통과 판정이나 timeout 으로만 끝난다.
- 성공 / 실패 모두 `terminate = 1` 로 끝나서 둘의 차이가 보상에 없다.

### 1.8 그 밖의 작은 문제

- `_reset_task` 의 `self._ball_contact_flag = torch.zeros([n])` 가 전체 버퍼를 크기 n 짜리로 덮어쓴다 (다음 `_compute_reward` 에서 다시 만들어져 실제 피해는 없음).
- 원본 공은 `soccer_ball.urdf` 를 직접 불러온다 (friction 0.2, restitution 0.8, damping 0.1 / 0.05, MuJoCo friction 없음).

---

## 2. Trap_Fix 설계

### 2.1 클래스 구조

```python
class HumanoidTrap_Fix(BallPhysicsMixin, humanoid_trapping_3.HumanoidTrapping_3):
```

| 메서드 | 내용 |
|---|---|
| `__init__` | `_setup_ball_profile(cfg)` → super → `_init_ball_friction(self._ball_handles, self._ball_actor_ids)` → 공 모델 상수 계산 (§2.2) → `max_episode_length` 계산 (§2.6) |
| `_load_obj_asset` | 원본을 복사하되 공만 `self._create_ball_asset()` |
| `_reset_actors` | env 마다 Bernoulli(`AIR_RATIO`) 로 공중 / 땅볼 분기 |
| `_reset_ball_air`, `_reset_ball_ground` | §2.3, §2.4 |
| `_reset_task` | 상태 머신 버퍼 초기화 (§2.8) |
| `_compute_reward`, `_compute_reset` | §2.5, §2.7–2.9. 원본 jit 함수는 쓰지 않고 새 함수 |

관측 함수와 IS_INIT 초기 상태는 원본 그대로 쓴다. 원본 모듈의 `IS_*` 전역 플래그는 바꾸지 않는다.

### 2.2 공 프로필: `linear_damping_and_mjfriction`

[linear_damping_and_mjfriction.yaml](calm/data/cfg/soccer_hlc/ball/linear_damping_and_mjfriction.yaml) 의 **실제 값** (yaml 상단 주석과 다른 곳이 있어 값 기준으로 정리):

| 항목 | 값 | 비고 |
|---|---|---|
| 반경, 질량 | 0.11 m, 0.45 kg (density 80.71) | |
| 관성 | `inertia_ratio: null` → 꽉 찬 구 0.4 m r² | 주석은 "hollow 2/3" 라고 되어 있음 |
| linear / angular damping | 0.1 / 0.0 | |
| 미끄럼 마찰 | 공 0.7, 바닥 1.0 → 쌍 **0.85** (PhysX 평균) | 주석은 "(0.2 + 1.0)/2 = 0.6" |
| 반발 | 공 0.85, 바닥 0.2 → 쌍 **0.525** | 주석은 0.88 |
| MuJoCo friction | rolling 0.012 m, torsional 0.004 m | 구름 감속 `0.764 + 0.0714 v` m/s² (아래) |

땅볼 발사 속도 계산(§2.4)에 쓰는 구름 감속 상수는 하드코딩하지 않고 프로필에서 계산한다:
`J = 1 + I / (m r²)` (꽉 찬 구 1.4), `a_r = mu_r · g / (r · J)`, `k = linear_damping / J` → a_r = 0.764 m/s², k = 0.0714 /s.

프로필 주석 수정은 이 작업 범위 밖이라 그대로 두고 보고만 한다.

### 2.3 공중볼

**샘플링 순서** (T 를 먼저 구해야 착지 범위를 정할 수 있으므로 φ 를 먼저 뽑는다. T 는 착지점과 무관):

1. v ~ U(8, 15), φ ~ U(10°, 60°)
2. `T = 2 v sinφ / g`, `R = v² sin2φ / g` (공칭 진공값)
3. `max_fall_dist = min(FALL_DIST_PER_SEC · T, FALL_DIST_CAP)` (기본 3.0 m/s, 4.0 m)
   - `FALL_DIST_PER_SEC`: 체공 시간 동안 캐릭터가 평균적으로 옮겨 갈 수 있는 속도. 착지점이 "T 초 안에 걸어가거나 뛰어가서 닿을 수 있는 거리" 안에 오도록 한다. 3 m/s 는 달리기 약 4 m/s 에서 반응·감속·받는 자세를 잡는 시간을 뺀 값으로 잡은 초기값이다.
   - `FALL_DIST_CAP`: 체공이 긴 공에서 범위가 끝없이 커지지 않게 하는 상한. 원본 최대값 4 m 를 그대로 썼다. 이보다 멀면 과제가 "받기" 보다 "달려가기" 가 된다.
   - 이렇게 구하는 이유: 받을 수 있는지는 공이 떨어지기까지 남은 시간으로 정해지므로, 범위를 T 에 비례시키는 것이 가장 직접적인 기준이다. 원본과의 비교는 아래 표에 있다.
4. 착지점 = root + `root_vel · min(T, FALL_LEAD_TIME_MAX)` + U(0, max_fall_dist) · (랜덤 방향) (기본 0.5 s)
5. 발사점 = 착지점 − R · (cos θ, sin θ), θ 랜덤, z = r (0.11)
6. 발사 속도 (감쇠 해석해 + 적분 보정):

```python
c = 공 linear_damping                      # 0.1
f = 1 - exp(-c T)
v_h = c R / f                              # 수평, 착지점 방향
v_z = g T / f - g / c + 0.5 g h            # h = sim dt / substeps = 1/120
```

- 선형 감쇠 운동방정식 `dv/dt = −c v − g ẑ` 의 해에서, T 초 뒤 수평 R, 높이 0 에 오는 초기 속도다. 착지점과 T 는 그대로이고 실제 발사 속도와 각도만 조금 커진다 (예: v 15, φ 45° → 16.1 m/s, 43.0°).
- `½ g h` 는 semi-implicit Euler 의 `−½ g h t` 낙하를 상쇄한다 (0.04 m/s).
- 수치 모델 기준 해석해 단독 오차는 −0.06~−0.09 m (적분 오차만 남음). `½ g h` 까지 넣은 결과는 §4 에서 Isaac Gym 으로 확인한다.

| 범위 (v 8–15, φ 10–60°) | 최소 | 최대 |
|---|---|---|
| T | 0.28 s (v 8, φ 10°) | 2.65 s (v 15, φ 60°) |
| R | 2.2 m (v 8, φ 10°) | 22.9 m (v 15, φ 45°) |
| 최고점 높이 | 0.10 m (공 중심 0.21 m) | 8.6 m |
| max_fall_dist | 0.85 m (T 0.28 s) | 4.0 m (T ≥ 1.33 s) |

**착지 범위를 T 에 비례시키는 이유** (원본 `2 + 2·ratio`, ratio = 속도 정규화와 비교):

원본 식에서 바뀌는 것은 두 가지다. 기준을 속도에서 T 로 바꾸고, 상수항 2 m 를 없애 원점을 지나는 비례식으로 만든다. 상수는 기울기(`FALL_DIST_PER_SEC`, "체공 1 초당 허용 이동 거리")와 상한(`FALL_DIST_CAP`, 원본 최대값 4 m) 두 개다.

| 예 | T | 원본 (속도 기준) | T 비례 `min(3T, 4)` | 필요한 평균 이동 속도 (원본 최대 거리) |
|---|---|---|---|---|
| v 8, φ 10° | 0.28 s | 0–2.0 m | 0–0.85 m | 7.1 m/s |
| v 15, φ 15° | 0.79 s | 0–4.0 m | 0–2.4 m | 5.1 m/s |
| v 8, φ 60° | 1.41 s | 0–2.0 m | 0–4.0 m | 1.4 m/s |
| v 15, φ 60° | 2.65 s | 0–4.0 m | 0–4.0 m | 1.5 m/s |

- 원본은 짧은 공(위 두 줄)에 캐릭터가 낼 수 없는 속도를 요구하고, 체공이 긴 v 8, φ 60° 공에는 오히려 범위를 좁게 준다. 도달 가능성은 T 로 정해지므로 T 기준이 맞다.
- 상수항이 있으면 (`2 + 2·(T 정규화)` 같은 형태) T 가 짧을 때도 최소 2 m 가 남아 같은 문제가 생긴다. 0 에서 시작하는 비례식이어야 짧은 공의 범위가 0 쪽으로 줄어든다.
- ratio 는 샘플링 범위 (`AIR_SPEED_MIN/MAX`) 로 정규화한 값이라, 범위를 바꾸면 같은 공의 착지 범위가 달라진다. T 비례식은 공 자체의 물리량만 본다.
- 기울기 3 m/s 는 달리기 4 m/s 에서 반응 시간을 뺀 평균값으로 잡은 초기값이다. 이전 안의 반응 시간, 하한 상수는 비례식 하나로 대신해 없앴다.

**착지점 중심을 `root + root_vel · min(T, 0.5)` 로 두는 이유** (구현 후 변경: 처음에는 `root_vel · T` 였다. 초기 상태 속도 분포로 계산하면 초기 위치에서 착지점까지 공중볼 최대 16.9 m, 땅볼 13.5 m 이고 4 m 를 넘는 경우가 약 30 % 라 너무 멀었다. 원본과 같은 0.5 s 상한을 두면 최대 약 6.7 m. 아래는 처음 안의 근거로, 속도를 반영하는 이유 자체는 그대로다):

- IS_INIT 초기 상태 (`move_ball_state.pkl`, 40960 개) 의 root 수평 속도는 중앙값 1.8 m/s, 75 % 3.3, 90 % 4.2, 최대 5.3 m/s 다. 대부분 움직이는 상태에서 시작한다.
- 위 범위식은 정지 상태에서 출발한다고 보고 정한 값이다. 4 m/s 로 달리는 캐릭터에게 뒤쪽 착지점은 멈추고 돌아서야 해서 같은 거리라도 훨씬 멀다. 그래서 중심을 "지금 속도로 T 초 동안 가면 있을 위치" 로 옮기면 범위식의 가정 (그 자리에서 출발) 과 맞는다.
- 상수 0.5 s 는 T 와 무관해서 짧은 공에는 과하고 긴 공에는 모자란다. 빼 버리면 (중심 = root) 빠르게 달리는 초기 상태에서 뒤쪽 착지점이 도달 불가능해진다.

**각속도**: 매 리셋마다 각 축 U(−`AIR_SPIN_MAX`, +`AIR_SPIN_MAX`) 로 새로 뽑는다 (기본 30 rad/s ≈ 5 회전/s). 원본처럼 리셋하지 않으면 이전 에피소드의 값 (정지 후 0, 트래핑 직후 값 등) 이 그대로 남아 분포를 통제할 수 없다. PhysX 에는 Magnus 효과가 없어 공중 궤적과 착지점은 바뀌지 않고, 몸에 닿을 때의 튐만 다양해진다.

**`AIR_INIT: trapping` (선택, 2026-11-06 추가):** 공중볼을 HumanoidTrapping_3 `_reset_ball_new_2` 방식으로 초기화한다. 속도 U(`AIR_SPEED_MIN`, `AIR_SPEED_MAX`), 각 U(`AIR_PHI_MIN`, `AIR_PHI_MAX`) (원본 10–15 m/s, 15–60°), 착지 반경 U(0, 2 + 2·(v−v_min)/(v_max−v_min)), 중심 root_vel·0.5 (원본처럼 root 위치는 더하지 않음, IS_INIT 리셋은 root xy = 0), 진공 포물선 + 속도 × (1 + 0.025T) 보정, 공 높이 0.115, 각속도는 이전 값 유지. 원본과 다른 점은 목표 부위가 `TARGET_POINT` 를 따르는 것뿐이다. linear damping 0.1 프로필에서 실측 착지점은 지정 착지점보다 발사 쪽으로 중앙 0.42 m, 최대 1.0 m 짧다 (원본 값 기준). `AIR_SPIN_MAX`, `FALL_*` 는 쓰지 않는다.

### 2.4 땅볼

착지점을 정확히 맞추지 않는다. 통과 실패를 착지점과 무관하게 판정하므로 (§2.7), 공이 "대략 그쪽으로, 대략 그 속도로" 오면 된다. 이전 안의 슬립 모델, 이분법, 옆 휨 보정은 모두 뺐다.

1. 도착 속도 v_a ~ U(3, 15), 도착 시간 T ~ U(`GROUND_T_MIN`, `GROUND_T_MAX`) (기본 0.5, 2.0 s)
2. 순수 구름 닫힌 식으로 발사 속도와 거리:
   ```
   v0 = (v_a + a_r/k) · e^{kT} − a_r/k
   D  = (v0 + a_r/k) · (1 − e^{−kT}) / k − a_r T / k
   ```
3. 조준점 (fall_pos) = 공중볼과 같은 방식 (`max_fall_dist(T)`, 중심 `root + root_vel · min(T, FALL_LEAD_TIME_MAX)`)
4. 발사점 = 조준점 − D · (cos θ, sin θ), z = r, 속도 v0 로 조준점 방향, vz = 0
5. 각속도 = 구름 각속도 (v0 / r, 진행 방향에 수직인 수평축) + 수평 두 축에 U(±`GROUND_SPIN_NOISE`) + 수직축 U(±`GROUND_SPIN_Z_MAX`)
   - 수평 노이즈는 슬립을 만들어 공을 조금 감속시키거나 옆으로 휘게 한다. 이 효과는 보정하지 않는다.
   - 수직축 회전은 경로를 바꾸지 않는다 (torsional friction 으로만 줄어듦).

**노이즈에 따른 오차** (substep 적분 모델, v_a 3–15, T 0.5–2.0 의 모서리 조합에서 최대값):

| `GROUND_SPIN_NOISE` | 도착 속도 오차 | 진행 방향 위치 오차 | 옆 휨 |
|---|---|---|---|
| ±5 rad/s | 0.16 m/s | 0.30 m | 0.28 m |
| **±10 rad/s (기본)** | **0.32 m/s** | **0.59 m** | **0.55 m** |
| ±20 rad/s | 0.66 m/s | 1.16 m | 1.09 m |

노이즈가 0 이면 오차는 1 cm 이하다. 도착 속도 3–15 m/s 범위는 거의 그대로 유지된다.

| 범위 (노이즈 0) | v0 (발사 속도) | D (발사 거리) |
|---|---|---|
| v_a 3, T 0.5 | 3.5 m/s | 1.6 m |
| v_a 3, T 2.0 | 5.1 | 8.1 |
| v_a 15, T 0.5 | 15.9 | 7.7 |
| v_a 15, T 2.0 | 19.0 | 33.9 |

**목표 부위**: 양발 중 Bernoulli(0.5) (배치 크기와 무관하게 반반).

필요한 상수: v_a 범위 2, T 범위 2, 스핀 2 (원본 땅볼과 비슷한 개수). a_r, k 는 프로필에서 계산한다 (§2.2).

### 2.5 트래핑 판정

body 별 기하 근접 판정은 쓰지 않는다. "어느 body 에 닿았나" 대신 "공이 사람에게 잡혔나" 만 판정하고, 금지 부위 (팔) 는 그 body 의 contact force 로 따로 잡는다 (§2.7).

| | 판정 | 근거 |
|---|---|---|
| 공중볼 | (공 net contact force (control step 중 simulate() 마다 읽은 최대) > `BALL_FORCE_THR` (원본 0.1) 이고, (공이 바닥 근처가 아니거나 다리 표면 근처) | 공중에서는 공에 닿을 수 있는 것이 사람뿐이다. 원본의 `공 높이 > 0.2` 는 뺀다. 바닥 근처에서는 net force 로 바닥과 몸을 구분할 수 없어서, 공 표면이 다리 (허벅지, 정강이 capsule, 발은 중심 (0.045, 0, −0.02) 반지름 0.10 구 근사) 표면에서 `TRAP_NEAR_MARGIN` (0.05 m) 안이면 트래핑으로 본다. 이때는 다리가 공을 막았을 수 있으므로 자유낙하 외삽을 쓰지 않는다 (2026-11-05, 발로 낮게 받으면 LANDED 로 끝나던 문제). 또는 바닥 외 충격량 > `TRAP_IMPULSE_THR` (0.5 m/s) (아래, 2026-11-08) |
| 땅볼 | 한 control step 사이 공 수평 속도 변화 `|v_t − v_{t−1}|_xy > GROUND_DV_THR` (기본 1.0 m/s) | 바닥에서는 net force 에 수직항력이 섞여 힘으로 구분할 수 없다. 물리 감속은 구름 최대 약 2.1 m/s² (v 19 m/s, 0.07 m/s/step), 슬립 μg 8.3 m/s² (0.28 m/s/step) 라 1.0 m/s/step (30 m/s²) 이면 사람 접촉만 걸린다 |

**공중볼에서 땅과 닿는 경우**: 공이 바닥에 닿는 step 에는 바닥 반력 때문에 net force 도 생긴다. 바닥 접촉은 공 중심이 r + contact_offset (0.13) 안에 들어올 때만 생기므로, 같은 step 에 먼저 "착지 → 실패" (§2.7) 를 판정하고 그다음에 접촉을 판정하면 바닥 반력이 트래핑으로 잘못 잡히지 않는다. 트래핑 이후에는 바닥에 닿아도 상관없다.

판정 순서 (매 step, phase 0 에서): 넘어짐 → 핸드볼 → (공중) 착지 → (공중) 공 접촉력 / (땅볼) 속도 변화 → (땅볼) 통과.

**공중볼 접촉력은 simulate() 마다 읽고, 충격량 판정을 OR 로 더한다 (2026-11-08)**: 트래핑 후에도 LANDED 로 끝나는 문제.
- 원인: 공 접촉력을 control step 끝 (PhysX 4 번 = simulate 2 번 × substep 2 중 마지막) 에서만 읽어서, 중간에 다리에 맞고 튀어 나간 공은 접촉력 0 → phase 0 그대로 → 떨어지면 LANDED.
- 지금 판정: `air_trap = (max_simulate 접촉력 > BALL_FORCE_THR & (다리 근처 | ~바닥 근처)) | (바닥 외 충격량 > TRAP_IMPULSE_THR)`. 충격량은 Kick_Fix 와 같은 식 (`compute_non_ground_impulse`, `ball_physics.py`): `J = Δv + g dt ẑ` 에서 바닥 근처일 수 있으면 바닥 마찰 원뿔까지의 거리, 아니면 |J|.
- `TRAP_MIN_HEIGHT` (기본 0.2, null 이면 끔): 위 air_trap 에 `공 중심 z > TRAP_MIN_HEIGHT` 를 AND 로 더한다 (Trapping_3 의 높이 조건). 켜면 다리 근처 바닥 바운스 (하프발리) 와 발 높이에서 받는 공은 트래핑이 아니고, 바닥에 닿으면 LANDED.
- 측정 (`trapfix-joint-airlegacy-rewardz`, 공중볼만, 1024 env, 약 2000 에피소드, 유령 공 정답 gt = |v_진짜 − v_유령|, gt > 1 m/s 를 터치로 봄):

| 판정 | recall (gt > 1) | gt < 0.05 인데 판정 |
|---|---|---|
| step 끝 접촉력 (이전) | 0.46 | 197 step |
| simulate 마다 접촉력 (2 번) | 0.73 | 284 step |
| 충격량 > 0.5 | 0.94 | 0 |
| simulate 마다 접촉력 \| 충격량 (채택) | 0.99 | 284 step |

  - 이전 판정에서 LANDED 의 28 % (381 / 1383) 는 착지 전에 gt > 1 터치가 있었다.
  - gt < 0.05 인데 판정되는 것은 거의 모두 공이 다리 표면 5 cm 안 (높이 중앙값 0.13 m) 에서 접촉력이 잡힌 경우로, 이전 판정에도 있던 다리 근처 규칙의 결과다 (측정 중 phase 가 안 바뀌어 한 에피소드에서 여러 step 이 세어졌을 수 있음).
  - 4 번 다 읽으려면 dt 1/120, substeps 1, controlFrequencyInv 4 로 바꿔야 한다 (recall 0.98, 충격량과 OR 0.997). 물리가 바뀌어 같은 정책의 success 400 → 314 라 재학습이 필요해서 쓰지 않았다.
  - 학습 속도 (4096 env, GPU 공유 상태라 잡음 큼): 기존 fps total 약 23k, simulate 마다 읽기 약 19–20k.

### 2.6 에피소드 길이

`__init__` 에서 설정값으로 계산해 `self.max_episode_length` 를 덮어쓰고 출력한다 (yaml `episodeLength` 는 무시하고 주석으로 표시).

```
T_air_max    = 2 · AIR_SPEED_MAX · sin(AIR_PHI_MAX) / g                     # 2.65 s
T_ground_max = GROUND_T_MAX + PASS_DIST / GROUND_ARRIVE_SPEED_MIN           # 2.0 + 0.33 s
episode      = ceil((max(T_air_max, T_ground_max) + TRAP_WINDOW_STEPS · dt + EPISODE_MARGIN) / dt)
```

기본값이면 (2.65 + 0.33 + 0.3) / (1/30) → **99 step (3.3 s)**. 해석해 보정은 T 를 그대로 두므로 이 계산이 유지된다. 땅볼은 스핀 노이즈로 도착이 조금 늦을 수 있어 `EPISODE_MARGIN` 으로 흡수한다.

### 2.7 종료 조건

| 조건 | 공중 | 땅볼 | 내용 |
|---|---|---|---|
| 넘어짐 | ○ | ○ | 원본 `has_fallen` 그대로 |
| 핸드볼 | ○ | ○ | `HANDBALL_BODIES` 중 하나라도 contact force > `HANDBALL_FORCE_THR` → 실패. 원본 `tar_fail`, `contact` 를 대체. 그 외 body (torso, head, pelvis, 다리) 는 목표가 아니어도 닿아도 된다 |
| 착지 | ○ | | phase 0 에서 공 중심 z < r + GROUND_EPS (0.13) → 실패. 직전 step 상태로 자유낙하를 외삽해 이번 step 안에 0.13 아래로 내려왔어야 하는 경우도 착지로 본다 (substep 2 개라 바닥에 튕겨 0.13 위로 다시 올라간 채 step 이 끝나는 경우가 있음, 구현 중 실측으로 확인). 같은 step 에 공중볼 트래핑이 참이면 트래핑이 우선 |
| 통과 | | ○ | phase 0 에서 공이 캐릭터를 지나감 → 실패 (아래) |
| 트래핑 창 종료 | ○ | ○ | 트래핑 판정 (§2.5) 후 `TRAP_WINDOW_STEPS` 가 지나면 종료 |
| timeout | ○ | ○ | §2.6 길이. 정상이면 일어나지 않음 |

- 모든 종료는 `terminate = 1` 이다 (원본과 같음).
- `HANDBALL_BODIES` 기본값: 양팔 6 개 (upper_arm, lower_arm, hand). 팔은 넘어질 때 말고는 바닥에 닿지 않으므로, 팔의 contact force 는 거의 공 접촉이다. 넘어지면서 팔이 바닥에 닿아도 어차피 실패라 문제없다.

**땅볼 통과 판정** (착지점과 무관):

```
p = (ball_xy − root_xy) · v̂_ball_xy        # 공 진행 방향으로 본 캐릭터 → 공 거리
통과 ⇔ phase 0 이고 p > PASS_DIST (기본 1.0 m)
```

- 리셋 직후 공은 캐릭터 쪽으로 오므로 p < 0 이다. 공이 캐릭터 옆이나 몸 근처를 지나 진행 방향으로 1 m 더 가면 p > 1 이 된다. 공이 어디를 조준했든, 스핀으로 얼마나 휘었든 같은 식으로 판정된다.
- 캐릭터가 공 경로에서 멀리 떨어져 있어도 공이 캐릭터 위치를 (진행 방향 기준으로) 지나는 순간 판정되므로 놓치는 경우가 없다.
- 공이 거의 멈춘 경우 (`|v_ball_xy| < 0.5 m/s`) 에는 방향이 불안정하므로 판정하지 않는다. v_a ≥ 3 이라 캐릭터 근처에서 멈출 일은 없고, 그래도 남는 경우는 timeout 이 처리한다.
- 캐릭터가 공을 뒤에서 쫓아가는 경우도 공이 1 m 넘게 앞서 나가면 실패다 (받는 동작이 아니므로 의도된 동작).

### 2.8 트래핑 상태 머신

```
phase 0 (트래핑 전) ── 트래핑 판정 (§2.5) ──▶ phase 1 (창, TRAP_WINDOW_STEPS step)
   │ 넘어짐 / 핸드볼 / 착지 / 통과 → 실패            │ 넘어짐 / 핸드볼 → 실패
   ▼                                                 ▼ 창 종료
 종료                                               종료
```

- 원본에도 성공 보상은 없었다. 원본은 접촉 뒤 10 step 동안 같은 거리 보상을 받고 무조건 종료했다. Trap_Fix 도 보너스 없이, 창 안의 접촉 후 보상 (§2.9) 으로만 잘 받은 경우와 아닌 경우를 구분한다.
- 성공 여부는 **로그용으로만** 기록한다. 창 마지막 step 에 `|v_ball_xy − v_root_xy| < SUCCESS_REL_SPEED` (기본 1.5 m/s) 이고 `|ball_xy − root_xy| < SUCCESS_DIST` (기본 1.0 m) 이면 성공.
- 버퍼: `_phase`, `_window_count`, `_end_reason` (로그용 정수: 성공, 창 종료 실패, 넘어짐, 핸드볼, 착지, 통과, timeout).

### 2.9 보상

**트래핑 전 (phase 0)**:

```
r_pre = face · exp(−PRE_NEAR_SCALE · d_near)
face  = FACE_FLOOR + (1 − FACE_FLOOR) · exp(−FACE_ERR_SCALE · |θ| / π)
```

- `d_near`: 공 중심과 목표 body 의 **geom 중심점** 사이 3D 거리. 원본은 관절 원점 (발목, 무릎, 목 등) 까지 거리였다. geom 중심점은 body 좌표계의 고정 offset 이다: torso (0, 0, 0.12), head (0, 0, 0.175), thigh (0, 0, −0.21), shin (0, 0, −0.2), foot (0, 0, −0.05, foot mesh 의 위치. 구현 때 mesh 중심으로 확인). 판정이 아니라 보상 shaping 이라 근사여도 된다.
- `TARGET_POINT` (yaml): `geom` 이면 위의 geom 중심점, `joint` 이면 원본처럼 관절 원점 (body 원점) 까지 거리. `joint` 에서는 허벅지 (관절 원점 = 고관절) 를 공중볼 목표에서 빼고 앞 6 개 부위만 뽑는다. one-hot 크기는 8 그대로 (허벅지 칸은 항상 0).
- **목표 부위에 허벅지 추가** (공중볼): 원본 one-hot 6 개 [torso, head, r_shin, r_foot, l_shin, l_foot] 뒤에 [r_thigh, l_thigh] 를 붙여 **8 개**로 한다. 떨어지는 공을 허벅지로 받는 것은 흔한 트래핑이고, 새 판정 (§2.5) 에서 허벅지 접촉은 이미 허용된다. 공중볼은 8 개 중 균일하게 뽑는다. 땅볼은 원본과 같이 양발 (인덱스 3, 5) 만 쓴다. 기존 순서를 유지해 앞 6 개의 의미는 원본과 같다. 관측은 15 → 17 이 된다 (§2.10).
- 가로채기 위치 항 (`d_int`, 공이 올 자리와 캐릭터의 xy 거리) 은 넣지 않는다. 원본처럼 `d_near` 하나로 시작한다. `exp(−5 · d_near)` 는 공이 1 m 넘게 떨어져 있으면 0.007 이하라 체공 초반에는 신호가 거의 없으므로, 착지점까지 이동을 잘 못 배우면 그때 다시 검토한다.
- `θ`: 캐릭터 heading 과 `−launch_dir` (정규화, 공중 / 땅볼 모두) 의 각도.
- **facing 을 hard cut 이 아니라 연속 곱으로 하는 이유**: hard cut (원본: |θ| > 60° 면 보상 0) 은 "공을 보지 않으면 보상 없음" 이 분명하다는 장점이 있다. 하지만 공이 뒤나 옆에서 오면 (θ 가 균일해 약 2/3 가 |θ| > 60°) 돌아서는 동안 계속 0 이라, 어느 쪽으로 돌아야 하는지 신호가 없다. 우연히 60° 안으로 들어올 때까지 탐색에 맡기게 된다. 연속 곱은 돌수록 보상이 커져 회전 방향을 알려 준다.
  - hard cut 의 의도 (안 보면 사실상 0) 를 살리려면 `FACE_FLOOR = 0` 으로 둔다. `FACE_ERR_SCALE` 4 기준 |θ| = 60° 에서 0.26, 90° 에서 0.14, 180° 에서 0.02 다. 기본값을 `FACE_FLOOR: 0.0` 으로 바꿨다.
  - hard cut 을 쓸지는 §6 에 남긴다.

**트래핑 후 (phase 1)**: 공과 캐릭터의 수평 속도 차이가 작을수록 크다.

```
r_post = POST_W · exp(−POST_VEL_SCALE · |v_ball_xy − v_root_xy|)
```

- `POST_VEL_Z: True` 면 공 z 속도를 더해 `|(v_ball_xy − v_root_xy, v_ball_z)|` 를 쓴다 (기본 False). z 는 root 기준이 아니라 공의 절대 z 속도라, 발 위에서 튀거나 떠 있는 공도 감점된다.
- `POST_TARGET_VEL: True` 면 트래핑 후 목표 공 속도를 쓴다 (기본 False, 2026-11-10). 리셋 때 방향 = 공이 날아온 쪽 (`-launch_dir`) ± `POST_TAR_AZ_RANGE` (180°), 속력 U(`POST_TAR_SPEED_MIN`, `POST_TAR_SPEED_MAX`) (0–3 m/s) 를 뽑아 에피소드 동안 고정. 보상은 `|v_ball_xy − v_tar|` (`v_root` 대신, `POST_VEL_Z` 는 그대로 적용), 관측에 heading 기준 목표 방향 2 + 목표 속력 1 을 붙인다 (17 → 20). 성공 판정 (`SUCCESS_REL_SPEED`, `SUCCESS_DIST`) 은 바꾸지 않았다.

- 원본에 주석 처리된 식은 `exp(−5 · |ball_vel − root_vel|²)` (3D) 였는데, `root_vel = root_pos − prev_root_pos` 로 dt 로 나누지 않아 단위가 틀려 있었다. 여기서는 root state 의 속도를 쓰고, 수평 성분만, 제곱 없이 쓴다 (기본 scale 1.0 → 차이 1 m/s 에서 0.37).
- 제곱 없이 써도 음수는 나오지 않는다. `|v_ball_xy − v_root_xy|` 는 벡터 크기라 항상 0 이상이고, `exp(−k · x)` 는 (0, 1] 이다.
- 제곱을 쓰지 않는 이유는 꼬리 때문이다. 트래핑 직후에는 공이 3–15 m/s 로 들어와 속도 차이가 크다. 제곱형은 차이가 클 때 너무 빨리 0 이 되어, 창 초반에는 공을 죽이는 정도에 따른 차이가 보상에 거의 없다.

  | 속도 차이 | 0.5 | 1 | 2 | 3 | 5 m/s |
  |---|---|---|---|---|---|
  | `exp(−1.0 · x)` (제안) | 0.61 | 0.37 | 0.14 | 0.050 | 0.007 |
  | `exp(−0.5 · x²)` | 0.88 | 0.61 | 0.14 | 0.011 | 0.000004 |

  제곱형은 0 근처에서 평평해 (차이 0.5 와 0 의 보상 차이가 작음) 마지막 정리 단계 신호도 약하다. 반대로 작은 오차를 관대하게 넘기고 싶다면 제곱형이 맞다.
- 공이 캐릭터와 같은 속도로 함께 멀어지는 경우도 보상을 받지만, 받은 직후 10 step 안에 그렇게 되기는 어렵다. 거리는 성공 로그에서만 본다.

`extras["reward_terms"]` 에 pre / post 를, `extras` 에 종료 사유별 비율과 성공률을 넣어 RLGPUAlgoObserver 로 기록한다.

### 2.10 관측

원본 `compute_heading_observations_simple_onehot` 를 그대로 쓰고, 목표 부위 one-hot 만 6 → 8 (허벅지 추가, §2.9) 이라 관측은 **17** 이다 (공 위치 3 + 속도 3 + 각속도 3 + one-hot 8). 기존 Trapping_3 체크포인트와는 관측 크기가 달라 이어서 학습할 수 없다. phase 플래그는 넣지 않는다.

### 2.11 yaml 키 (기본값 제안)

```yaml
  ballProfile: linear_damping_and_mjfriction
  ballOverrides: {}

  AIR_RATIO: 0.5                 # 리셋 때 공중볼 확률 (나머지 땅볼)

  # 공중볼
  AIR_SPEED_MIN: 8.0             # 공칭 발사 속도 [m/s] (감쇠 보정 전)
  AIR_SPEED_MAX: 15.0
  AIR_PHI_MIN: 10.0              # 발사각 [deg]
  AIR_PHI_MAX: 60.0
  AIR_SPIN_MAX: 30.0             # 각 축 U(-max, max) [rad/s]

  # 땅볼
  GROUND_ARRIVE_SPEED_MIN: 3.0   # 조준점 도착 속도 [m/s]
  GROUND_ARRIVE_SPEED_MAX: 15.0
  GROUND_T_MIN: 0.5              # 도착 시간 [s]
  GROUND_T_MAX: 2.0
  GROUND_SPIN_NOISE: 10.0        # 구름 각속도에 더하는 수평 두 축 노이즈 U(-max, max) [rad/s]
  GROUND_SPIN_Z_MAX: 15.0        # 수직축 U(-max, max) [rad/s]
  PASS_DIST: 1.0                 # 공이 진행 방향으로 캐릭터를 이만큼 지나면 실패 [m]

  # 착지점 범위: max_fall_dist = min(FALL_DIST_PER_SEC * T, FALL_DIST_CAP), 중심 = root + root_vel * min(T, FALL_LEAD_TIME_MAX)
  FALL_DIST_PER_SEC: 3.0
  FALL_DIST_CAP: 4.0
  FALL_LEAD_TIME_MAX: 0.5

  # 트래핑 판정 / 종료
  BALL_FORCE_THR: 0.1            # 공중볼: control step 중 simulate() 마다 읽은 공 net contact force 최대 [N]
  TRAP_IMPULSE_THR: 0.5          # 공중볼: 또는 바닥 외 공 충격량 [m/s]
  TRAP_MIN_HEIGHT: 0.2           # 공중볼: 트래핑은 공 중심 높이 > 이 값일 때만 [m], null 이면 안 봄
  GROUND_DV_THR: 1.0             # 땅볼: control step 당 공 수평 속도 변화 [m/s]
  TRAP_NEAR_MARGIN: 0.05         # 공중볼: 공 표면이 다리 표면에서 이 거리 안이면 바닥 근처 접촉도 트래핑 [m]
  HANDBALL_BODIES: ["right_upper_arm", "right_lower_arm", "right_hand",
                    "left_upper_arm", "left_lower_arm", "left_hand"]
  HANDBALL_FORCE_THR: 1.0        # [N]
  TRAP_WINDOW_STEPS: 10
  EPISODE_MARGIN: 0.3            # episodeLength 는 이 값들로 계산 (§2.6)

  # 성공 로그 (보상에는 안 씀)
  SUCCESS_REL_SPEED: 1.5
  SUCCESS_DIST: 1.0

  # 보상
  PRE_NEAR_SCALE: 10.0
  FACE_FLOOR: 0.0                # 0 = 공을 등지면 사실상 0 (hard cut 과 비슷하지만 회전 방향 신호는 남음)
  FACE_ERR_SCALE: 4.0
  POST_W: 1.0
  POST_VEL_SCALE: 1.0
  POST_VEL_Z: False
```

`contactBodies`, `terminationHeight` 는 넘어짐 판정용으로 원본 그대로 둔다. sim 설정 (substeps 2, TGS, `default_buffer_size_multiplier`) 도 원본 그대로다.

---

## 3. 작업 순서

1. 클래스 / yaml 골격, 공 프로필 연결, parse_task 등록 → 확인: 환경이 뜨고 관측 크기 17
2. 공중볼 리셋 (§2.3) → 확인: §4-1
3. 땅볼 리셋 (§2.4) → 확인: §4-2
4. 에피소드 길이 (§2.6) → 확인: 출력값 99, phase 0 timeout 0 건
5. 트래핑 판정과 종료, 상태 머신 (§2.5, §2.7, §2.8) → 확인: §4-3, §4-4
6. 보상 (§2.9) → 확인: 각 항이 정렬 / 트래핑에 따라 예상대로 변하는지 play 모드에서 로그
7. 짧은 학습으로 종료 사유 분포와 성공률 확인 → train_config.md 에 기록

---

## 4. 검증 계획

Isaac Gym (컨테이너, `singularity exec --nv --writable`) 에서 측정한다.

1. **공중볼 착지 오차**: 캐릭터를 멀리 두고 리셋 분포에서 수천 개를 쏜다. 공 중심이 z = r 을 지나는 시점의 xy 와 착지점 차이가 평균 < 0.05 m, 최대 < 0.15 m 인지, 착지 시각과 T 의 차이도 기록.
2. **땅볼 도착**: 조준점을 지날 때 속도 분포가 3–15 m/s 를 덮는지, 도착 시각이 T 근처인지. 위치는 정확히 맞출 필요가 없으므로 오차 분포만 기록한다 (§2.4 표와 비슷한지).
3. **트래핑 판정**:
   - 땅볼을 캐릭터 없이 굴려서 `GROUND_DV_THR` 오검출이 0 인지 (슬립 구간 포함).
   - 공중볼이 땅에 닿는 step 이 트래핑이 아니라 착지로 잡히는지.
   - 랜덤 정책으로 돌려 사람 접촉이 트래핑으로 잡히는지 play 화면과 대조.
4. **종료**: 랜덤 정책과 기존 Trapping_3 체크포인트로 종료 사유 분포를 본다. 예: 목표가 torso 인 에피소드가 접촉 즉시 실패하지 않는지, 땅볼 통과가 공이 몸을 지난 직후에 잡히는지.댈

검증 스크립트는 `tools/` 에 둔다.

---

## 5. 원본 대비 바뀌는 것 요약

| 항목 | Trapping_3 | Trap_Fix |
|---|---|---|
| 공 | soccer_ball.urdf 직접 (μ 쌍 0.6, e 0.8) | `linear_damping_and_mjfriction` (μ 쌍 0.85, e 쌍 0.525, MuJoCo rolling) |
| 공중 / 땅 비율 | 1 : 0, `int(r n)` 분할 | 0.5 : 0.5, env 별 Bernoulli |
| 공중볼 | v 10–15, φ 15–60, 착지 범위 2 + 2·(속도 비율), 중심 + root_vel·0.5, 경험식 보정 (최대 −1 m) | v 8–15, φ 10–60, 착지 범위 min(3T, 4), 중심 + root_vel·T, 해석해 + ½gh |
| 공 각속도 | 리셋 안 함 | 매 리셋 랜덤 (공중 ±30, 땅볼 구름 + 노이즈) |
| 땅볼 | 경험식 거리, 마찰 무시 | 도착 속도 3–15, 순수 구름 닫힌 식, 정확한 조준은 하지 않음 |
| 에피소드 | 60 step (2 s), T 최대 2.65 s | 설정값으로 계산 (기본 99 step) |
| 트래핑 판정 | 공 net force + 높이 > 0.2, 땅볼 불가 | 공중: 공 net force (착지 먼저 판정), 땅볼: 공 속도 급변 |
| 실패 | 비접촉 body (torso, head 포함) 접촉 | 넘어짐, 핸드볼 (팔 contact force), 착지, 땅볼 통과 (캐릭터 기준) |
| 성공 | 없음 (10 step 뒤 무조건 종료) | 없음 (창 종료로 끝). 성공은 로그로만 |
| 목표 부위 | 공중 6 개 (torso, head, 정강이, 발), 땅볼 양발 | 공중 8 개 (허벅지 추가), 땅볼 양발 |
| 트래핑 전 보상 | 관절 원점까지 3D 거리, facing hard cut | geom 중심까지 거리, facing 연속 곱 (floor 0) |
| 트래핑 후 보상 | 트래핑 전과 같은 식 | 공과 캐릭터의 수평 속도 차이 |
| 관측 | 15 | 17 (one-hot 8) |

---

## 6. 결정 필요 항목

메모로 정해진 것: 목표 외 부위는 허용하고 팔만 실패 (핸드볼로 통합), 성공 보너스 없음, phase 플래그 없음, 공중볼 각속도 랜덤, 땅볼 정확한 조준 없음, 기하 근접 판정 없음, 트래핑 후 보상은 속도 차이, 땅볼 목표 부위는 양발만, 착지점 중심은 `root + root_vel · min(T, 0.5)` (구현 후 `root_vel · T` 에서 변경), 공중볼 목표 부위에 허벅지 추가 (관측 17).

남은 것:

1. **facing**: 연속 곱 + `FACE_FLOOR` 0 (제안, 회전 방향 신호) vs 원본 hard cut (|θ| > 60° → 0).
2. **뒤에서 오는 공** (대화에서 제안, 아직 미반영): 모션 데이터 기준 180° 회전에 0.7–1.0 s 가 걸려, 뒤에서 오는 T < 1 s 공 (전체의 약 15 %) 은 받을 수 없다.
   - 오는 방향 α 에 따라 최소 체공시간 `T_req = T_MIN + TURN_TIME · α / π` (기본 0.3, 0.7 s) 를 두고, 공중볼은 φ 하한, 땅볼은 T 하한으로 맞출지.
   - 트래핑 판정 순간 공이 heading 기준 뒤쪽 반구에 있으면 (등에 맞음) 실패로 할지.
3. **`FALL_DIST_PER_SEC` 3.0 m/s**: 초기값이다. 학습 초반 착지 실패 비율을 보고 조정.
4. **임계값** (`BALL_FORCE_THR` 0.1, `GROUND_DV_THR` 1.0, `HANDBALL_FORCE_THR` 1.0): §4-3 측정 후 확정.
