# PhysicsFC 학습 설정 보고서 (LLC + HLP 5종)

작성일: 2026-10-28 · 기준: 현재 working tree 코드 (커밋 `0f789ef` + 미커밋 수정분)

## 0. 읽기 전에

- **근거**: 모든 내용은 현재 저장소 코드와 yaml에서 읽은 값이다. git 이력은 `Initial commit` 하나뿐이라, 배포된 체크포인트(`calm/calm_exp/checkpoints/*/default.pth`)를 학습할 때 이 코드가 그대로였는지는 확인할 수 없다.
- **물리 실측값**: 질량, 관성, 마찰, 반발 등은 같은 에셋과 같은 `AssetOptions`로 Isaac Gym(`physicsfc` conda env)에서 직접 불러와 읽은 값이다. CPU 파이프라인과 GPU 파이프라인에서 결과가 같았다. 측정 스크립트는 `probe_phys.py`, `probe_roll.py`(scratchpad)이다.
- **표기**: `[코드]`는 코드를 읽어서 확인한 내용, `[실측]`은 Isaac Gym에서 읽은 값, `[미확인]`은 확인하지 못한 내용이다.
- 정책 1 step = 1/30 s이다 (sim dt 1/60, `controlFrequencyInv: 2`). 따라서 에피소드 길이 N step = N/30 초다.

### 학습 구성 요약

| | LLC | Move | Dribble | Kick | Trap | Ball Mastery |
|---|---|---|---|---|---|---|
| `--task` | `HumanoidAMPGetup` | `HumanoidMove` | `HumanoidDribble` | `HumanoidBallPos_1` | `HumanoidTrapping_3` | `HumanoidBallMastery` |
| cfg_env | `humanoid_calm.yaml` | `soccer_hlc/humanoid_move.yaml` | `soccer_hlc/humanoid_dribble.yaml` | `soccer_hlc/humanoid_ball_pos_task_1.yaml` | `soccer_hlc/humanoid_trapping_3.yaml` | `soccer_hlc/humanoid_ball_mastery.yaml` |
| cfg_train | `calm_humanoid.yaml` | `hrl_humanoid.yaml` | 〃 | 〃 | 〃 | 〃 |
| 캐릭터 에셋 | `amp_humanoid.xml` | `amp_humanoid_shoe.xml` | 〃 | 〃 | 〃 | 〃 |
| 에피소드 길이 | 300 (10 s) | 50000 (≈27.8 min) | 300 (10 s) | 120 (4 s) | 60 (2 s) | 45 (1.5 s) |
| 초기 상태 | `Hybrid` + getup | `Default` (T-pose) | `Default` | `Default` | `Default`, 단 pkl에서 로드 (`IS_INIT=True`) | `Default` 20 % + 상태 버퍼 80 % |
| early termination | ON | **OFF** | **OFF** | ON | ON | 터치 종료 / 공 3 m |
| 공 | 없음 | 없음 | 있음 | 있음 | 있음 | 있음 |
| task obs 크기 | – | 5 | 14 | 10 | 15 | 15 |
| 전체 obs 크기 | 223 | 228 | 237 | 233 | 238 | 238 |

---

## 1. 물리 파라미터 비교

### 1.1 시뮬레이션 공통 (5개 환경 모두 동일)

`sim:` 블록은 5개 yaml에서 값이 모두 같다.

| 항목 | 값 |
|---|---|
| sim dt / substeps | 1/60 s / 2 (물리 스텝 1/120 s) |
| 제어 주기 | 30 Hz (`controlFrequencyInv: 2`) |
| 중력 | (0, 0, −9.81) |
| solver | TGS (`solver_type: 1`), position iter 4, velocity iter 0 |
| contact_offset / rest_offset | 0.02 / 0.0 |
| bounce_threshold_velocity | 0.2 m/s (이보다 느린 충돌에는 restitution이 적용되지 않음) |
| max_depenetration_velocity | 10.0 |
| default_buffer_size_multiplier | 10.0 |

### 1.2 바닥 (ground plane)

| 항목 | LLC | Move | Dribble | Kick | Trap |
|---|---|---|---|---|---|
| static friction | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dynamic friction | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| restitution | **0.0** | **0.0** | **0.2** | **0.2** | **0.2** |

- 바닥은 `gym.add_ground`로 만든 무한 평면이다 ([humanoid.py:197](calm/env/tasks/humanoid.py#L197)).
- Dribble은 2026-10-29부터 바닥 값을 task yaml의 `plane:`이 아니라 공 물리 프로파일의 `ground:`에서 읽는다(1.4절). `real_soccer`와 `dribble_legacy` 모두 1.0 / 1.0 / 0.2라서 값은 바뀌지 않았다. 실제 잔디 기준은 restitution 0.35지만, 캐릭터도 같은 바닥을 쓰므로 0.2를 유지했다.
- 테스트용으로 만든 `humanoid_calm_rest02.yaml`은 LLC 설정에서 restitution만 0.2로 바꾼 것이고, 학습에는 쓰이지 않았다.

### 1.3 캐릭터

| 항목 | LLC | Move / Dribble / Kick / Trap (HLP 공통) | (참고) 데모 `fifa.py` |
|---|---|---|---|
| 에셋 | `amp_humanoid.xml` | `amp_humanoid_shoe.xml` | `amp_humanoid_shoe.xml` |
| 발 형상 | box 0.177×0.09×0.055 | shoe mesh (`shoe_small.obj`) | shoe mesh |
| 발 질량 (좌/우) [실측] | 1.000 / 1.000 kg | 0.483 / 0.484 kg | 〃 |
| 전체 질량 [실측] | **48.89 kg** | **47.86 kg** | 47.86 kg |
| friction [실측] | 1.0 | 1.0 | 1.0 (코드에서 지정) |
| restitution [실측] | 0.0 | 0.0 | 0.0 (코드에서 지정) |
| torsion_friction | 에셋 0.05 → **actor 0** [실측] | 〃 | 〃 |
| rolling_friction | 에셋 0.05 → **actor 0** [실측] | 〃 | 에셋 0.5로 지정 → actor 0 |
| compliance [실측] | 0 | 0 | 0 |
| thickness [실측] | 0.02 | 0.02 | – |
| angular_damping (body) | **0.01** | **0.01** | **0.5** |
| linear_damping (body) | 0.0 (기본값) | 0.0 | 0.0 |
| max_angular_velocity | 100 | 100 | 100 |
| max_linear_velocity | 1000 (기본값) | 1000 | – |
| DOF 구동 | PD (`DOF_MODE_POS`) | 〃 | 〃 |
| DOF effort limit [실측] | 3.4e38 (사실상 무제한) | 〃 | 〃 |
| DOF velocity limit [실측] | 100 | 100 | 100 |
| DOF friction [실측] | 0 | 0 | 0 |

- **body별 damping**: angular_damping 0.01은 [humanoid.py:258](calm/env/tasks/humanoid.py#L258)의 `AssetOptions` 값 하나가 15개 body 전부에 적용된다. body마다 다르게 주는 코드는 없다.
- **에셋 차이**: LLC는 box 발 에셋으로, HLP는 shoe mesh 발 에셋으로 학습했다. 두 에셋은 발 형상과 발 질량만 다르고, 나머지 body 질량과 관절 값은 같다 [실측]. 그래서 HLP를 학습할 때 LLC는 자신이 학습한 적 없는 발 형상과 질량으로 구동되었다.
- **rolling/torsion friction은 시뮬레이션에 적용되지 않는다** [실측]. MJCF의 `friction="1.0 0.05 0.05"`가 에셋에는 들어가지만, actor를 만든 뒤 읽으면 0이 나온다. 동작 실험으로도 확인했다.
  - 공을 수직축으로 회전시키거나 굴렸을 때, torsion·rolling 값을 0, 0.05~0.2, 1.0으로 바꿔도 2초 동안 각속도와 속도가 전혀 줄지 않았다.
  - 에셋 단계 설정과 actor 단계 설정, CPU와 GPU 파이프라인 모두 같은 결과였다.
  - 대조 실험에서 일반 마찰은 정상 작동했다.
  - 바닥이 평평한 발은 일반 마찰만으로 수직축 회전이 멈추므로(box 발 실험, 0.5 s) 캐릭터에는 실질적인 영향이 없다.

**관절 PD 게인 (MJCF stiffness/damping이 그대로 kp/kd로 쓰임, 모든 환경 동일)** [실측]

| 관절 | kp | kd | armature |
|---|---|---|---|
| abdomen x/y/z | 1000 | 100 | 0.02 |
| neck x/y/z | 100 | 10 | 0.01 |
| shoulder x/y/z | 400 | 40 | 0.02 |
| elbow | 300 | 30 | 0.01 |
| hip x/y/z | 500 | 50 | 0.02 |
| knee | 500 | 50 | 0.02 |
| ankle x/y/z | 400 | 40 | 0.01 |

**body 질량** [실측]: pelvis 9.996, torso 12.012, head 3.882, upper_arm 1.499, lower_arm 1.000, hand 0.500, thigh 4.502, shin 3.000, foot 1.000(LLC) / 0.48(HLP) kg

### 1.4 공

LLC와 Move에는 공이 없다. Kick과 Trap의 공 에셋은 `soccer_ball.urdf`(충돌 구 r=0.11, `<inertial>` 주석 처리됨)이고, `override_inertia=True`로 불러온다.
Dribble은 2026-10-29부터 공 물리 프로파일 `real_soccer`를 쓴다(아래 "공 물리 프로파일" 참고). 표의 Dribble 열에서 괄호 안 "이전" 값은 변경 전 설정이며, `dribble_legacy` 프로파일로 그대로 재현된다.

| 항목 | Dribble (`real_soccer`) | Kick | Trap | (참고) 데모 `fifa.py` |
|---|---|---|---|---|
| 반지름 | 0.11 m | 0.11 m | 0.11 m | 0.11 m |
| density 옵션 | 77.1 (2026-10-29 변경, 이전 80.71 / 2026-10-28 이전 1000) | 80.71 (2026-10-28 변경, 이전 1000) | 80.71 | 80.71 |
| 질량 | 0.430 kg [실측] (이전 0.450 / 5.575 kg) | 0.450 kg (이전 5.575 kg) | 0.450 kg [실측] | 0.450 kg |
| 관성 Ixx=Iyy=Izz | 0.003468 = (2/3)·m·r², 속이 빈 공 [실측] (이전 0.002178 = (2/5)·m·r² / 0.02698) | 0.002178 (이전 0.02698) | 0.002178 [실측] | 0.002178 |
| linear_damping | 0.1 (이전 0.2) | 0.1 | 0.1 | 0.1 |
| angular_damping | **0.0** (이전 0.5) | 0.05 | 0.05 | 0.05 |
| max_angular_velocity | **300** (이전 100) | 100 | 100 | 100 |
| max_linear_velocity | 1000 (기본값) | 1000 | 1000 | – |
| friction | 0.2 (이전 0.5) | 0.2 | 0.2 | 0.2 |
| restitution | **0.88** (이전 1.0) | 0.8 | 0.8 | 0.8 |
| rolling_friction | 지정 안 함 → 0 [실측] (이전 0.5 → actor 0) | 0.2 → actor 0 | 0.2 → actor 0 | 0.2 → actor 0 |
| torsion_friction [실측] | 0 | 0 | 0 | 0 |
| compliance [실측] | 0 | 0 | 0 | 0 |
| thickness [실측] | 0 | 0 | 0 | – |
| MuJoCo식 torsion / rolling (외부 토크) | **0.005 / 0.015 m** (이전 없음) | 없음 | 없음 | 없음 |
| 코드 위치 | [real_soccer.yaml](calm/data/cfg/soccer_hlc/ball/real_soccer.yaml), [ball_physics.py](calm/env/tasks/soccer_hlc/ball_physics.py) | [humanoid_ball_pos_task_1.py:124](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L124) | [humanoid_trapping_3.py:219](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L219) | [fifa.py:287](calm/fifa.py#L287) |

- Kick/Trap의 관성 실측값은 균일 구 공식 (2/5)·m·r²와 일치한다. Dribble(`real_soccer`)은 실제 공처럼 속이 빈 구 (2/3)·m·r²이다.
- **공의 rolling_friction 설정은 효과가 없다** [실측, 1.3절 참고]. 바닥에서 구르는 공의 감속과 수직축 회전의 감쇠는 `linear_damping`과 `angular_damping`에서만 나온다.
- **공 질량**: 2026-10-28에 Dribble과 Kick에도 `density = 80.71`을 지정해 세 태스크 모두 0.45 kg이 되었다. 변경 전에는 density를 지정하지 않아 기본값 1000이 적용되어 5.575 kg(실제 축구공 약 0.43 kg의 13배)이었다. 기존 Dribble/Kick 체크포인트는 변경 전 5.575 kg 공으로 학습된 것으로 보인다. 변경 후 값은 Trap 실측값과 같은 옵션이라 같은 질량으로 보았고, 따로 재측정하지는 않았다.
- **Dribble 공의 다른 설정**: damping, 마찰, 반발, 관성, MuJoCo식 마찰이 Dribble만 나머지와 다르다 (2026-10-29 이전에도 lin 0.2, ang 0.5, friction 0.5, restitution 1.0으로 달랐다).

**공 물리 프로파일** (2026-10-29 추가, 현재 Dribble만 사용)
- 공과 바닥 설정을 `calm/data/cfg/soccer_hlc/ball/<이름>.yaml`에 모으고, task yaml에서 `ballProfile: <이름>`으로 고른다. 값 몇 개만 바꿀 때는 `ballOverrides`에 같은 구조로 적는다 (예: `{mj_friction: null}`로 MuJoCo식 마찰 끄기).
  - `ball:` 공 asset 설정. `inertia_ratio`가 있으면 그 관성으로 URDF를 만들어 불러온다 (actor 단계에서 관성을 바꾸면 GPU PhysX가 crash하므로 asset 단계에서 넣는다). null이면 기존처럼 꽉 찬 구.
  - `ground:` 바닥 마찰과 반발. `cfg["env"]["plane"]`에 덮어써지므로 캐릭터도 같은 바닥을 쓴다.
  - `mj_friction:` 공과 바닥 사이의 MuJoCo식 torsion/rolling 마찰. null이면 끈다.
- 프로파일: `real_soccer` (현재 Dribble), `dribble_legacy` (2026-10-29 이전 Dribble 설정, 기존 체크포인트 확인용).
- 코드: [ball_physics.py](calm/env/tasks/soccer_hlc/ball_physics.py)의 `BallPhysicsMixin`. 다른 태스크에 붙일 때는 클래스 상속, `__init__`에서 `_setup_ball_profile`, 공 asset 생성, state tensor 생성 뒤 `_init_ball_friction` 호출만 추가한다.

**MuJoCo식 torsion/rolling 마찰** [코드, 실측]
- PhysX에는 torsion/rolling 마찰이 없어서 공에만 외부 토크로 넣는다. 캐릭터에는 아무 힘도 가하지 않는다.
  ```
  N     = m·g  (공 바닥이 지면에서 2 cm 이내일 때), 아니면 0
  τ_z   = −sign(ω_z) · min(μ_t·N, I·|ω_z|/h)                  # 수직축 회전 (torsion)
  τ_xy  = −ω_xy/|ω_xy| · min(μ_r·N, (I + m r²)·|ω_xy|/h)       # 구름 (rolling)
  h = sim dt (1/60), simulate 호출마다(제어 step당 2번) 적용
  ```
  - min의 두 번째 항은 한 step 안에 회전을 정확히 멈추는 크기로, 정지한 공이 반대로 떨리지 않게 한다.
  - 다룬 접촉은 공과 바닥뿐이다. 발이 공을 위에서 눌러도 N은 커지지 않고, 발과 공 사이의 torsion도 없다.
  - N에 net contact force를 쓰지 않는 이유: 캐릭터가 넘어져 있는 env에서는 바닥에 가만히 있는 공의 net contact force가 m·g, 0.5배, 약 28배(약 118 N) 사이를 오갔다 [실측]. 이 값을 쓰면 구르는 공이 한 step 만에 멈췄다.
- 실측 (Dribble env, 캐릭터에서 떨어진 공, 3.43 m/s로 굴림 / 10 rad/s 제자리 회전):

  | 설정 | 굴러간 거리 | 회전 정지 (< 0.1 rad/s) |
  |---|---|---|
  | `real_soccer` | 6.26 m (3.8 s) | 1.63 s |
  | `real_soccer`, mj_friction 끔 | 15 s 동안 33.9 m+ (멈추지 않음) | 멈추지 않음 |
  | `dribble_legacy` | 15 s 동안 11.8 m+ (멈추지 않음) | 9.2 s |

  - `real_soccer`는 FIFA 굴림 기준(4~10 m, Pro 4~8 m)을 통과하고, [ball_fifa_test_vis.py](ball_fifa_test_vis.py)의 결과(6.26 m, 1.63 s)와 같다. 이 스크립트도 같은 `MujocoFriction` 클래스를 import해서 쓴다.
  - env 계산 시간: 1024 env × 200 step 기준 5.07 s(끔) → 5.25 s(켬), 약 3% 증가.

### 1.5 충돌 필터 [코드]

같은 env의 actor들은 같은 collision group을 쓰고, filter 비트가 겹치면(AND ≠ 0) 서로 충돌하지 않는다.

| actor | filter | 비고 |
|---|---|---|
| 캐릭터 | 0b01 | 자기 자신과는 충돌 안 함 |
| 공 | 0b10 | 캐릭터와 충돌함 |
| heading marker, 목표 표시 구 | 0b11 | 아무것과도 충돌 안 함 (시각화용) |
| Kick 목표 오브젝트 (`soccer_target.urdf`) | 2 | 충돌 반경 0, mass 0, 고정 |

### 1.6 물리 차이 요약

| 차이 | 영향 범위 |
|---|---|
| 캐릭터 발: box(1.0 kg) vs shoe mesh(0.48 kg) | LLC ↔ HLP 전체 |
| 바닥 restitution 0.0 vs 0.2 | LLC·Move ↔ Dribble·Kick·Trap |
| 공 질량 5.575 vs 0.45 kg (2026-10-28 0.45 kg으로 통일, 기존 체크포인트는 변경 전 조건) | Dribble·Kick ↔ Trap·데모 |
| 공 damping, friction, restitution, 관성, MuJoCo식 마찰 (2026-10-29 Dribble을 `real_soccer`로 변경) | Dribble ↔ Kick·Trap·데모 |
| 캐릭터 angular_damping 0.01 vs 0.5 | 학습 전체 ↔ 데모 |
| 캐릭터 rolling_friction 0.05(에셋) vs 0.5(데모 지정) | actor 단계에서 둘 다 0으로 읽힘 |

`[미확인]` PhysX가 두 물체의 friction과 restitution을 합치는 방식(평균, 최소 등)은 확인하지 않았다.

---

## 2. 학습 하이퍼파라미터 (cfg_train)

### 2.1 LLC — `calm_humanoid.yaml`

| 항목 | 값 |
|---|---|
| 알고리즘 | CALM (PPO + conditional discriminator + encoder) |
| actor/critic MLP | [1024, 1024, 512] relu, separate |
| disc / conditional_disc / enc MLP | [1024, 1024, 512] |
| σ | 고정, log σ = −2.9 |
| lr | 2e-5 (constant) |
| γ / τ | 0.99 / 0.95 |
| horizon / minibatch / mini_epochs | 32 / 16384 / 6 |
| e_clip / critic_coef / bounds_loss_coef | 0.2 / 5 / 10 |
| mixed_precision | True |
| latent_dim | 64 |
| latent 교체 주기 | 10~150 step 사이 무작위 |
| reward 가중치 | task 0.0, disc 0.0, **conditional_disc 1.0** |
| conditional_disc_reward_scale | 2 |
| enc_regularization_coeff | 0.1 |
| amp batch / minibatch / buffer | 512 / 4096 / 20000 |
| motion_file | README는 `soccer_90.yaml`, `tools/run_train_soccer.sh`는 `soccer_9.yaml` — 실제로 어느 쪽을 썼는지는 [미확인] |

> ⚠ **`disc_coef: 5관`**: 이 값은 문자열 `'5관'`으로 파싱된다. `disc_reward_w=0`이면 `disc_loss=0`이 되는데, 이때 `'5관' * 0 = ''`이 되고 `tensor + ''`에서 `TypeError`가 난다(재현 확인). 따라서 현재 파일 그대로는 LLC의 첫 업데이트에서 학습이 중단될 것으로 예상된다 ([calm_agent.py:415](calm/learning/calm_agent.py#L415)).

### 2.2 HLP 공통 — `hrl_humanoid.yaml`

| 항목 | 값 |
|---|---|
| actor/critic MLP | [1024, 512] relu, separate |
| action | 64차원 latent (LLC 입력으로 쓰기 전에 L2 정규화) |
| σ | 고정, log σ = −2.3 |
| lr | 2e-5 (constant) |
| γ / τ | 0.99 / 0.95 |
| horizon / mini_epochs | 32 / 6 |
| e_clip / critic_coef / bounds_loss_coef | 0.2 / 5 / 10 |
| llc_steps | 1 (HLP 1 step = LLC 1 step = 1/30 s) |
| reward 가중치 | **task 1.0**, disc 0.0, style 0.0 → reward = 환경 task reward만 사용 |

> ⚠ 현재 파일에는 실제 학습 값으로 보기 어려운 항목이 있다.
> - `max_epochs: 10` → 10 epoch에서 학습이 끝난다.
> - `minibatch_size: 128` → 주석에 원래 값으로 보이는 `16384`가 남아 있다.
> - `clip_value: Falsehrl` → 문자열이라 truthy로 처리되어 value clipping이 **켜진다**.
>
> 또 `numEnvs`는 커밋 기준 4096인데, 로컬에서 move는 128, dribble은 8로 수정되어 있다(미커밋). 실제 학습 때의 값은 [미확인].

---

## 3. LLC — `HumanoidAMPGetup`

### 3.1 에피소드
- `episodeLength: 300` (10 s)

### 3.2 초기 세팅 [코드]

reset이 필요한 env마다 아래 순서로 분기한다 ([humanoid_amp_getup.py:127](calm/env/tasks/humanoid_amp_getup.py#L127)).

1. **Recovery episode**: 넘어져 terminate된 env 중 확률 0.2(`recoveryEpisodeProb`)로 선택된다. 상태를 리셋하지 **않고** 쓰러진 자세 그대로 이어간다. 이후 60 step(`recoverySteps`, 2 s) 동안은 termination이 막혀 있어 일어나는 법을 학습한다.
2. 나머지 env 중 확률 0.1(`fallInitProb`)로 **fall state** 하나를 무작위로 가져와 시작한다. recovery 카운터는 60으로 설정된다.
   - fall state는 학습 시작 시 한 번만 만든다 ([humanoid_amp_getup.py:91](calm/env/tasks/humanoid_amp_getup.py#L91)). 만드는 방법은 다음과 같다.
     - 루트 회전을 무작위 쿼터니언(가우시안 샘플 후 정규화)으로 둔다.
     - PD 목표를 U(−0.5, 0.5) 무작위 액션으로 고정한다.
     - 물리를 150 step 돌린다.
     - 이때의 root/dof 상태를 저장하고 속도는 0으로 둔다. env 수만큼 만들어진다.
3. 남은 env는 `stateInit: Hybrid`를 따른다.
   - 확률 0.5(`hybridInitProb`): **reference state init**. 모션 클립을 가중치에 따라 샘플하고, 클립 안의 시간도 무작위로 골라 그 순간의 자세와 속도로 시작한다.
   - 나머지 0.5: **default**. 루트 z=0.89, 회전은 identity(+x를 향함), 모든 DOF 0(T-pose), 속도 0.

### 3.3 target / latent [코드]
- 별도의 task target은 없다. task obs도 없고, heading marker는 시각화 전용이다.
- latent z는 데모 모션의 60프레임 창(`numAMPEncObsSteps`)을 encoder에 넣어 얻는다. 각 env마다 10~150 step마다 새로 샘플한다.

### 3.4 termination [코드] ([humanoid.py:667](calm/env/tasks/humanoid.py#L667))
- **넘어짐**: 아래 두 조건이 모두 참이고 `progress > 1`일 때.
  - contactBodies(양 발)를 제외한 body 중 하나라도 접촉력이 0.1을 넘는다.
  - contactBodies를 제외한 body 중 하나라도 높이가 기준보다 낮다. 기준은 0.15 m이고, head만 0.3 m다.
- recovery 카운터가 0보다 크면 reset과 terminate가 모두 무시된다.
- 시간 초과: `progress ≥ 300`.

### 3.5 reward [코드]
- 환경 task reward는 항상 1이지만 가중치가 0이라 쓰이지 않는다.
- 실제 reward는 conditional discriminator reward 하나다.
  - `r = −log(max(1 − D(s, z), 1e-4)) × 2`
  - 여기서 D는 (AMP obs 10프레임, latent z)를 입력으로 받는 conditional discriminator다.

---

## 4. Move — `HumanoidMove`

### 4.1 에피소드
- `episodeLength: 50000` (≈27.8분). early termination도 OFF라서 **사실상 reset되지 않는다**. 캐릭터가 넘어져도 그대로 계속 진행된다.

### 4.2 초기 세팅
- `stateInit: Default`: 루트 z=0.89, identity 회전, T-pose, 속도 0.

### 4.3 target 설정 [코드] ([humanoid_move.py](calm/env/tasks/soccer_hlc/humanoid_move.py))
- reset 시점과 그 후 100~200 step마다(`headingChangeSteps`, 정수 균등 분포) 아래 값을 다시 뽑는다.
  - 이동 방향 θ ~ U(−π, π) → `tar_dir = (cos θ, sin θ)`
  - 바라볼 방향 θ_f ~ U(−π, π) → `tar_facing_dir`. 이동 방향과 독립이다.
  - 목표 속력 `tar_speed` ~ U(0, 7) m/s
- task obs(5): 캐릭터 heading 기준 `tar_dir`(2), `tar_speed`(1), 캐릭터 heading 기준 `tar_facing_dir`(2)

### 4.4 termination
- 시간 초과만 있다. 넘어짐 판정은 OFF다.

### 4.5 reward [코드]
`v` = 루트 xy 속도(이전 step과의 위치 차이 / dt), `s*` = tar_speed, `d` = tar_dir라고 하면:

```
v_par  = d · v                                  # 목표 방향 성분
v_tan  = sum(v − v_par·d)                       # ⚠ 벡터 크기가 아니라 x+y 성분 합 (부호 있음)
e_par  = (s* − v_par) / s*
e_tan  = v_tan / s*
r_dir  = exp(−0.25 · (e_par² + 0.1 · e_tan²))   # v_par ≤ 0 이면 0
r_face = max(0, facing_dir · tar_facing_dir)
r      = 0.7 · r_dir + 0.3 · r_face
```
- 계수: `VEL_ERR_SCALE`=0.25, `TANGENT_ERR_W`=0.1, `DIR_REWARD_W`=0.7, `FACING_REWARD_W`=0.3
- ⚠ `tarSpeedMin: 0`이라서 목표 속력이 0에 가까우면 e_par와 e_tan가 매우 커진다. 그러면 r_dir은 거의 0이 된다.

### 4.6 변형 — `HumanoidMoveStyle` [코드] ([humanoid_move_style.py](calm/env/tasks/soccer_hlc/humanoid_move_style.py))
원본 `PhysicsFC_project/calm`에서 그대로 옮겼다 (2026-11-19). cfg_env는 Move와 같은 `humanoid_move.yaml`을 쓴다. 학습 설정은 다음과 같다.
- cfg_train: `hrl_humanoid_style_control.yaml`
- algo: `hrl_soccer_conditioned`, [hrl_soccer_conditioned_agent.py](calm/learning/hrl_soccer_conditioned_agent.py)
- 모션: `soccer_move_style.yaml` (16개)
- LLC: `poseidon_90_100000.pth`

**목표 설정**
- obs, reward 식, termination은 Move와 같다. task obs는 5이다.
- 한 번에 리셋되는 env가 160개(16 × 10)를 넘으면 `make_situation_state`로 스타일 상황을 만든다.
  - 모션 16종마다 env의 5 %씩 `_tar_locomotion_index`를 지정한다.
  - 지정된 env는 face 방향을 이동 방향 + 모션별 각도(±10°)로 둔다. 모션별 각도는 Walk/Run/JogF 0°, WalkB/JogB 180°, CMove ±90°/±55°/±125°이다.
  - 지정된 env의 속력은 모션별 기준 속력의 0.8~1.2배이다. 기준 속력은 Walk 1, WalkB 0.5, Run 5, 나머지 2 m/s이다.
  - 지정되지 않은 env는 index −1이고, 속력을 절반으로 줄이고 task reward를 2배로 준다.
- 리셋 env가 160개 이하이면 Move와 같은 랜덤 목표를 쓴다.

**reward**
- `reward = 0.5 · task + 0.5 · style`이다 (`disc_reward_w` 0).
- style은 HLP latent와 지정 모션의 LLC encoder 출력(학습 시작 때 모션별 32개를 계산) 사이의 cosine 유사도를 (cos+1)/2로 바꾼 값이다. index −1이면 0이다.

**설정값**
- `llc_steps` 5 (원본 학습 명령도 지정하지 않았다. 원본 test 명령은 `--llc_steps 1`)
- `max_epochs` 10000
- `save_frequency` 50
- `minibatch_size` 16384
- test(`--test`)는 원본 전용 player(조이스틱, `/workspace` 경로 하드코딩) 대신 기본 `HRLPlayer`로 등록했다. 이를 위해 yaml에 player용 키를 `hrl_humanoid.yaml` 값으로 추가했다.

⚠ 원본 코드의 동작이라 그대로 두었다.
- **episode length와 리셋**: episode 50000에 early termination OFF라서, 처음 리셋(전체 env) 이후에는 스타일 상황이 거의 다시 만들어지지 않는다. 512 env smoke test 44 epoch 동안 2회였다.
- **`_tar_locomotion_index` 크기**: `make_situation_state`가 `_tar_locomotion_index`를 리셋된 env 수 크기의 새 텐서로 **교체**한다. 전체가 동시에 리셋될 때만 올바르다.
- **지정 안 된 env의 face 각도**: 새로 뽑지 않고 `2π·θ_f − π`로 변환된다.
- **출력**: `print('reward:')`가 매 step 출력되고, 상황을 만들 때마다 `print(n)`이 출력된다.

---

## 5. Dribble — `HumanoidDribble`

### 5.1 에피소드
- `episodeLength: 300` (10 s), early termination OFF.
- 공과 바닥 물리: `ballProfile: real_soccer` (2026-10-29부터, 이전 설정은 `dribble_legacy`). 1.4절 참고.

### 5.2 초기 세팅 [코드] ([humanoid_dribble.py:529](calm/env/tasks/soccer_hlc/humanoid_dribble.py#L529))
- 캐릭터: `Default` (T-pose, z=0.89). `IS_INIT_BY_FILE: False`.
- 공:
  ```
  ball_xyz = root_xyz + U[0,1)³ − 0.5        # INIT_BALL_DIST=0.5 → 월드축 기준 x, y 각각 ±0.5 m
  ball_z   = 0.12                            # INIT_BALL_Z (공 반지름 0.11 → 1 cm 띄움)
  ball_lin_vel = ball_ang_vel = 0
  ```
  - 오프셋은 캐릭터 heading이 아니라 **월드축** 기준이다. 범위가 ±0.5 m라서 공이 캐릭터 몸과 겹친 위치에 생길 수 있다.

### 5.3 target 설정 [코드]
- Move와 같은 방식이다. 100~200 step마다 이동 방향 θ ~ U(−π, π), 바라볼 방향 θ_f ~ U(−π, π), 목표 속력 ~ U(0.5, 5) m/s를 다시 뽑는다.
- task obs(14): `tar_dir`(2), `tar_speed`(1), `tar_facing_dir`(2), 공 상대위치(3, z는 절대 높이), 공 속도(3), 공 각속도(3). 모두 캐릭터 heading 기준이다.
- `tar_facing_dir`는 obs에는 들어가지만 **reward에는 쓰이지 않는다**.
- `relocate_dist`, `_ball_target`도 reward와 termination 어디에도 쓰이지 않는다.

### 5.4 termination
- 시간 초과(300)만 있다. 공이 멀어졌을 때 재배치하는 코드는 주석 처리되어 있다.

### 5.5 reward [코드] ([humanoid_dribble.py:639](calm/env/tasks/soccer_hlc/humanoid_dribble.py#L639))
`v_b` = 공 xy 속도(위치 차이 / dt), `v_c` = 캐릭터 루트 xy 속도라고 하면:

```
# (1) ball reward — 공이 tar_dir 방향으로 tar_speed로 움직이는지
b_par = d · v_b ;  b_tan = sum(v_b − b_par·d)
e_par = (s* − b_par)/s* ;  e_tan = b_tan/s*
r_ball = exp(−10 · (0.1·e_par² + e_tan²))        # b_par ≤ 0 이면 0   ⚠ Move와 가중치가 반대(평행 0.1, 수직 1)

# (2) humanoid reward — 캐릭터가 공 쪽으로 tar_speed로 움직이는지
u     = normalize(ball_xy − root_xy)
c_par = u · v_c ;  c_tan = sum(v_c − c_par·u)
r_hum = exp(−10 · (((s* − c_par)/s*)² + 0.1·(c_tan/s*)²))   # c_par ≤ 0 이어도 0 처리 없음

# (3) dist reward — 공과 캐릭터의 거리
r_dist = exp(−10 · ‖ball_xy − root_xy‖)

r = 0.6·r_ball + 0.2·r_hum + 0.2·r_dist          # BALL/HUMANOID/DIST_REWARD_W
```
- 세 성분은 `extras["reward_terms"]`로 넘겨 TensorBoard/wandb에 `reward_terms/*`(에피소드 합), `reward_terms_step/*`(epoch별 step당 평균)로 기록된다. 2026-10-29 이전에는 매 step 세 성분의 합계를 `rewards_<pid>.json`에 기록했다(`add_reward`, 삭제됨).

### 5.6 변형 — `HumanoidDribble_Fix` [코드] ([humanoid_dribble_fix.py](calm/env/tasks/soccer_hlc/humanoid_dribble_fix.py))
2026-10-30 추가. `HumanoidDribble`을 상속해 아래 세 가지만 바꾼 변형이다(#1은 2026-11-01 되돌림). 원본 `HumanoidDribble`은 그대로 남아 있다. 오버라이드하는 것은 `get_task_obs_size`, `_compute_task_obs`, `_compute_reward`, `_compute_reset` 넷뿐이며, 초기 세팅·target 샘플링·공 물리는 5.1~5.3절과 동일하다.

cfg_env는 전용 파일 [`humanoid_dribble_fix.yaml`](calm/data/cfg/soccer_hlc/humanoid_dribble_fix.yaml)을 쓴다. `humanoid_dribble.yaml`의 복사본이고 차이는 `relocate_dist` 한 줄(2.0 → 3.0)뿐이다.

| # | 변경 | 원본 | Fix |
|---|---|---|---|
| 1 | ~~수직 속도 크기~~ | `sum(v_tan)` (성분 합) | ~~`norm(v_tan)`~~ → 2026-11-01 원본과 같은 `sum(v_tan)`으로 되돌림 |
| 2 | task obs | 14 (`tar_facing_dir` 포함) | **12** (`tar_facing_dir` 제외) |
| 3 | `pos_err_scale` | 10.0 | **2.0** |
| 4 | termination | 시간 초과(300)만 | 시간 초과 **+ 공 거리 > `relocate_dist`(3.0 m)** |

- **(1)** 2026-11-01 되돌림: Fix도 원본과 같이 성분 합을 쓴다. 아래 방향 의존 문제는 그대로 남아 있다. 성분 합은 `√2·|cos(θ+45°)|`배라 목표 방향 θ에 따라 0~1.41배로 달라졌고, θ=45°/225°에서는 수직 오차 패널티가 0이었다. norm으로 바꾸면 방향 의존이 사라진다(ball/humanoid reward 양쪽). 9절 #8 참고.
- **(2)** `tar_facing_dir`는 reward 항이 없어 학습 신호가 없었다(5.3절). 전체 obs는 223 + 12 = **235**.
- **(3)** `r_dist = exp(−2·d)`. 같은 보상값 0.135가 나오는 거리가 0.2 m → **1.0 m**로 5배 완화된다. 초기 공 거리가 ±0.5 m(5.2절)라 원본에서는 에피소드 시작 시 `r_dist ≈ 0`이었다.
- **(4)** `_compute_reset`이 원본(`super()`)을 먼저 부른 뒤, `‖ball_xy − root_xy‖ > relocate_dist`인 env의 `reset_buf`/`_terminate_buf`를 1로 올린다. `_terminate_buf`를 함께 세우므로 rl_games가 해당 전이를 bootstrap하지 않는다(`extras["terminate"]`, [humanoid.py:461](calm/env/tasks/humanoid.py#L461)).
  - `enableEarlyTermination`은 **넘어짐 판정만** 제어하며 Fix에서도 `False`다. 공 거리 종료는 이 플래그와 무관하게 항상 켜져 있다. 넘어짐 종료도 쓰려면 `enableEarlyTermination: True`로 바꾸면 된다.
  - 에피소드 길이가 env마다 달라진다. 원본은 모든 env가 300 step에서 동시에 끝나 `reward_terms/*`(에피소드 합)가 약 9.4 epoch마다만 갱신됐는데, Fix에서는 수시로 갱신되고 에피소드 길이도 달라지므로 **에피소드 합끼리의 비교는 길이 차이를 포함한다.** step당 평균인 `reward_terms_step/*`을 보는 편이 낫다.
  - 공 초기 오프셋이 월드축 기준 ±0.5 m(5.2절)라 시작 거리는 최대 0.707 m다. 3.0 m는 그보다 충분히 크다.
- 정규화 분모는 원본과 같이 `s*` 그대로다(clamp 없음). `tarSpeedMin: .5`라 학습 중에는 `s* = 0`이 샘플링되지 않지만, play 모드에서 스틱을 놓으면 `tar_speed = 0`이 되어(`humanoid_dribble.py:366`) `0/0 → nan`이 될 수 있다. `--test`에서는 보상을 쓰지 않아 드러나지 않는다.
- `tar_speed = 0`(정지 명령)은 다루지 않는다. 의미 있는 명령으로 만들려면 `speed_mask`가 `tar_dir_speed ≤ 0`에서 보상을 0으로 만드는 것(5.5절)과 분모 스케일, 샘플링 비중을 함께 바꿔야 한다.
- `tangent_err_w`가 ball reward에서 평행 항에 곱해지는 문제(5.5절 ⚠)는 **그대로 두었다.**
- 보상 지형이 바뀌므로 기존 Dribble 체크포인트(`dribble-legacy`, `dribble-damping`, `dribble-realsoccer`)와 리턴 값을 직접 비교할 수 없다.

### 5.7 변형 — `HumanoidDribbleFace` [코드] ([humanoid_dribble_face.py](calm/env/tasks/soccer_hlc/humanoid_dribble_face.py))
2026-11-02 추가. 목표 방향을 바라보면서 공을 목표 방향으로 모는 task. `HumanoidDribble_Fix`를 상속하고 `get_task_obs_size`, `_compute_task_obs`, `_update_task`, `_reset_task`, `_compute_reward`를 오버라이드한다. 초기 세팅(캐릭터 기본 자세, 공 환형 스폰 0.3~0.5 m)은 5.2절과 같다.

cfg_env는 전용 파일 [`humanoid_dribble_face.yaml`](calm/data/cfg/soccer_hlc/humanoid_dribble_face.yaml) (`humanoid_dribble_fix.yaml` 복사본). 차이: `relocate_dist` 3.0 → **5.0**, `tarSpeedMax` 5 → **3**, 가중치 아래 표, `FACE_ERR_SCALE: 4.0`, `DIST_ERR_SCALE: 4.0` 추가. `headingChangeSteps*`, `enableRandHeading`은 쓰지 않는다.

- **obs**: 원본 `HumanoidDribble`과 같은 14 (`tar_dir` 2, `tar_speed` 1, `tar_facing_dir` 2, 공 위치/속도/각속도 9, 모두 heading 기준). 전체 obs 223 + 14 = 237. 원본 Dribble 체크포인트와 obs 배치가 같다.
- **target** (임시): 리셋 때 `face_dir = normalize(ball_xy − root_xy)`, `tar_dir = −face_dir`, `tar_speed ~ U[0.5, 3]`. 에피소드 중에는 다시 뽑지 않는다(`_update_task`가 아무것도 하지 않음). 캐릭터가 공을 바라보며 공을 자기 쪽(뒤로 물러나는 방향)으로 끌고 가는 형태다. 조이스틱 play는 지원하지 않는다.
- **termination**: 시간 초과(300) + 공 거리 > `relocate_dist`(5.0 m) (5.6절 #4와 같은 코드). 넘어짐 종료는 `enableEarlyTermination: False`로 꺼져 있다.
- **reward** (θ = 캐릭터 heading과 `face_dir`의 각도 차이, φ = heading 기준 공의 방위각, d = 공-루트 xy 거리):

| 항 | 식 | 가중치 |
|---|---|---|
| ball_face | `r_ball·(α + (1−α)·r_face)`, `r_ball` = 5.6절 ball 식(가중치 전), `r_face = exp(−4·|θ|/π)`, α = `FACE_FLOOR` **0.0** | `BALL_FACE_REWARD_W` 0.7 |
| humanoid | 5.6절과 같음 | `HUMANOID_REWARD_W` 0.2 |
| dist | `|φ| ≤ π/2`: `exp(−4·d)`, `|φ| > π/2`: `−exp(−4·d)·(|φ| − π/2)/(π/2)` | `DIST_REWARD_W` 0.1 |

  - ball_face: 공을 목표 방향으로 몰면서 목표 방향을 볼 때만 보상. 따로 더하면 "공을 보고 서 있기"만으로 face 보상이 나오므로 곱으로 묶었다. α > 0이면 방향이 틀려도 ball의 α배는 받는다. ball 보상이 0이면(공이 목표 방향으로 안 움직이면) face 신호도 0이 되는데, 처음 face dir = 공 방향이라 dist 항(공을 앞쪽 절반에)이 방향 신호를 일부 대신한다. `BALL_REWARD_W`는 1.0 (곱하기 전 값을 받기 위함).
  - face: θ = 0°/90°/180°에서 1/0.14/0.02. 오차가 클수록 기울기가 작아진다(180° 근처는 10°당 0.004). 시작 시 θ는 공 스폰 각도와 같다(캐릭터는 +x를 보고 시작). 2026-11-02 `0.5·[(1 − |θ|/π) + exp(−4θ²)]`에서 변경.
  - dist: 앞쪽 절반은 각도와 무관하게 `exp(−4d)`, 옆(90°)을 넘는 순간 0으로 떨어지고 정후방에서 `−exp(−4d)`. 옆에서 `exp(−4d)`만큼의 계단이 있다.
  - 공 스폰 각도가 무작위라 절반은 공이 등 뒤에서 시작해 첫 step부터 dist가 음수다. 에피소드 기대 보상이 음수가 되면 공을 5 m 밖으로 차서 끝내는 것이 이득이 될 수 있으므로 `reward_terms/dist`와 에피소드 길이를 확인할 것.
  - 로깅: `reward_terms/ball_face`, `humanoid`, `dist` (`total`이 항목 합이라 face 단독 값은 기록하지 않는다).

---

## 6. Kick — `HumanoidBallPos_1`

### 6.1 에피소드
- `episodeLength: 120` (4 s). `tools/run_test_hlc.sh`의 테스트 명령은 `--episode_length 80`으로 덮어쓴다.

### 6.2 초기 세팅 [코드] ([humanoid_ball_pos_task_1.py:248](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L248))
- 캐릭터: `Default` (T-pose, +x를 향함). `IS_INIT=False`.
- **공 위치**:
  ```
  near    ~ Bernoulli(0.5)
  dist    = near ? U(0.5, 1.0] : U[1.0, 3.0)        # (dist_max−1.0)·U+1.0, near일 때 dist_max=0.5
  θ_ball  ~ U[−π/2, π/2)                             # 월드 +x 기준 반원 = 초기 캐릭터 정면 쪽
  ball    = root_xy + dist·(cos θ_ball, sin θ_ball),  z = 0.11 (지면 접촉)
  ball_lin_vel = ball_ang_vel = 0
  ```
- **목표 지점(target)**:
  ```
  θ_tar   = θ_ball + U(−π/4, π/4)
  dist    = 50 m                                     # tarDistMin = tarDistMax = 50, 캐릭터 루트 기준
  height  ~ U(0, 50) m                               # tarHeightMin/Max
  target  = root_xy + 50·(cos θ_tar, sin θ_tar),  z = height
  ```
- **목표 공 속력**: `tar_speed` ~ U(15, 30) m/s
- **목표 방향**: `ori_direc = normalize(target − ball)`. 3D 단위벡터라서 높이 성분이 들어간다. 올려차는 각도는 약 0°~45°다.
- ⚠ `_ori_dist`, `_ori_z`, `_ori_direc`, `_fastest`는 reset되는 env만이 아니라 **모든 env**에 대해 다시 계산된다. 그래서 다른 env가 reset될 때 진행 중인 env의 `ori_direc`도 현재 공 위치 기준으로 바뀐다.

### 6.3 target / obs
- task obs(10): 공 상대위치(3, z 절대), 공 속도(3), `tar_speed`(1), `ori_direc`(3). 모두 캐릭터 heading 기준이다. 목표 지점 좌표 자체는 obs에 없다.

### 6.4 termination [코드] ([humanoid_ball_pos_task_1.py:654](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L654))
- 다음 중 하나라도 참이고 `progress > 1`이면 terminate된다.
  1. **넘어짐**: LLC와 같은 기준. contactBodies는 양 발이다.
  2. **tar_fail**: `tar_contact_forces`의 xy가 1을 넘고, 발이 아닌 body의 접촉력이 1을 넘을 때.
     - ⚠ `tar_contact_forces`는 body index `num_actors−2 = 1`로 잘라 쓴다. 이 index는 **캐릭터 torso**다. 공의 body index는 15다. 즉 실제로는 "torso 접촉 + 다른 body 접촉" 조건으로 동작한다 [코드 해석].
  3. **킥 후 10 step 경과**: 공 속력이 처음 0.5 m/s를 넘으면 `touched=1`이 되고, 그 뒤 카운터가 10이 되면 `touched=2`가 되어 terminate된다.
- 시간 초과: `progress ≥ 120`.
- `is_wrong_foot`(왼발로 찼는지 판정)는 계산만 하고 termination 조건에서는 주석 처리되어 있다.

### 6.5 reward [코드] ([humanoid_ball_pos_task_1.py:564](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L564))
```
r = 1.0 · exp(−10 · ‖ v_ball / tar_speed − ori_direc ‖²)
```
- `distance_reward`와 `tar_speed_reward`도 계산은 하지만 가중치가 0이다.
- 공이 정지해 있으면 ‖−ori_direc‖²=1이라 r=e⁻¹⁰≈4.5e-5이다. 즉 차기 전에는 reward가 사실상 0이다.
- 매 step `print('reward:', ...)`를 출력한다.

---

## 7. Trap — `HumanoidTrapping_3`

### 7.1 에피소드
- `episodeLength: 60` (2 s)

### 7.2 모듈 상단 플래그 (학습 경로 결정)

| 플래그 | 값 | 의미 |
|---|---|---|
| `IS_INIT` | True | 캐릭터 초기 상태를 `blackhole/init_states/move_ball_state.pkl`에서 샘플 |
| `IS_SIMPLE` | True | 단순화된 obs 사용 |
| `IS_TARGET_PART` | True | 목표 body part one-hot(6)을 obs에 포함 |
| `IS_TARGET_FALL` / `IS_CONTACT_OBS` / `IS_GROUND_BALL` / `IS_WALL` | False | 사용 안 함 |

### 7.3 초기 세팅 — 캐릭터 [코드]
- `move_ball_state.pkl`에서 상태를 무작위로 골라 root state, dof pos, dof vel을 설정한다. 이 pkl은 Move 정책을 실행해 수집한 상태로 추정된다.
- 그 다음 **루트 xy를 (0, 0)으로 덮어쓴다**. 루트 높이, 회전, 속도는 pkl 값을 그대로 쓴다.

### 7.4 초기 세팅 — 공 (`_reset_ball_new_2`) [코드] ([humanoid_trapping_3.py:753](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L753))

reset 비율은 `[1.0, 0.0]`이라서 **모든 env가 공중 공(air ball)** 이다. 바닥 공(`_reset_ball_new_ground`)과 먼 공(far ball)은 0%다.

포물선의 착지점을 먼저 정하고, 거꾸로 발사 위치와 발사 속도를 계산한다.

```
# 0) 캐릭터 이동 방향
root_dir = atan2(root_vel_y, root_vel_x)          # heading이 아니라 루트 '속도' 방향

# 1) 발사 속력
speed ~ U(10, 15) m/s
ratio = (speed − 10) / 5                          # 0~1

# 2) 착지점 (fall point)
max_fall_dist = 2 + 2·ratio                       # 2~4 m (빠른 공일수록 멀리 떨어질 수 있음)
fall_dist     ~ U(0, max_fall_dist)
fall_θ        = root_dir + U(−180°, 180°)          # 사실상 전방위 균등
fall_xy       = fall_dist·(cos fall_θ, sin fall_θ) + 0.5·root_vel_xy   # 0.5초 앞을 내다봄(리드)

# 3) 발사 방위와 발사각
θ  ~ U(−180°, 180°)                               # 착지점에서 본 발사 지점의 방위
φ  ~ U(15°, 60°)                                  # 발사 고각

# 4) 진공 포물선으로 비행시간과 수평거리 계산
t     = 2·speed·sin φ / g                          # g = 9.81
range = speed·cos φ · t

# 5) 발사 위치
launch_xy = fall_xy + range·(cos θ, sin θ),  z = 0.115
launch_dir = normalize(fall_xy − launch_xy)        # _launch_direc (공이 날아가는 방향)

# 6) 감쇠 보정 후 초기 속도
speed' = speed · (1 + 0.025·t)                     # linear_damping 0.1을 보상하기 위해 속력을 키움
v = speed' · (cos φ·cos(launch_dir), cos φ·sin(launch_dir), sin φ)
```

- **목표 body part**: `{torso, head, right_shin, right_foot, left_shin, left_foot}`(rigid body index 1, 2, 10, 11, 13, 14) 중 하나를 균등 확률로 고른다.
- **계산 범위**:
  - 수평거리: 최소 약 5.1 m(10 m/s, 15°)부터 최대 약 22.9 m(15 m/s, 45°)까지.
  - 비행시간: 약 0.53 s(10 m/s, 15°)부터 약 2.65 s(15 m/s, 60°)까지.
  - 에피소드가 2 s라서 **고각이 크고 빠른 공은 에피소드 안에 착지점에 도착하지 못할 수 있다**.
- 공 각속도는 reset 때 초기화하지 **않는다**. 이전 에피소드의 값이 남는다.
- 좌표는 원점 기준으로 계산된다. 캐릭터 루트 xy를 0으로 두었기 때문에 캐릭터 위치와 맞아떨어진다.

### 7.5 target / obs [코드]
- task obs(15): 공 상대위치(3, z 절대), 공 속도(3), 공 각속도(3), 목표 body part one-hot(6). 모두 캐릭터 heading 기준이다.
- `_reset_task`는 heading과 속도 목표도 뽑지만 obs와 reward에 쓰이지 않는다. 또 100~200 step 주기는 60 step 에피소드보다 길어서 reset 때만 실행된다.
- `_target_fall_pos`(접촉 후 ±90° 방향, 1 m 안쪽의 무작위 착지 목표)는 계산만 하고 obs(`IS_TARGET_FALL=False`)와 reward 어디에도 쓰이지 않는다.

### 7.6 termination [코드] ([humanoid_trapping_3.py:1274](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L1274))
contactBodies = `[right_foot, left_foot, right_thigh, right_shin, left_thigh, left_shin]`

다음 중 하나라도 참이면 terminate된다. 다른 태스크와 달리 `progress > 1` 조건이 **없다**.
1. **넘어짐**: contactBodies 외 body의 접촉력이 0.1을 넘고, contactBodies 외 body의 높이가 0.15 m(head는 0.3 m)보다 낮을 때.
2. **tar_fail**: 공 접촉력 xy가 1을 넘고, contactBodies 외 body의 접촉력이 1을 넘을 때. 공 접촉력 index는 body 17로 올바르다.
3. **ball_fall**: 공 높이가 0.13 m 미만일 때(즉 지면에 닿음). 공중 공에만 적용된다.
4. **contact**: contactBodies 외 body(pelvis, torso, head, 팔, 손) 중 **어디든 접촉력이 0.1을 넘을 때**. 바닥이든 공이든 상관없다.
   - ⚠ 그래서 목표 body part가 torso나 head인 경우, 공이 그 부위에 닿는 순간 바로 terminate된다 [코드 해석].
5. **트래핑 후 10 step 경과**: 공 접촉력이 0.1을 넘고 공 높이가 0.2 m를 넘으면 `touched=1`이 되고, 카운터가 10이 되면 terminate된다.
6. 시간 초과: `progress ≥ 60`.

### 7.7 reward [코드] ([humanoid_trapping_3.py:1210](calm/env/tasks/soccer_hlc/humanoid_trapping_3.py#L1210))
```
d      = ‖ ball_pos − rb_pos[target_part] ‖          # 3D 거리, one-hot으로 고른 body
r      = exp(−5 · d)
facing = facing_dir_xy · (−launch_dir)              # 캐릭터 정면과 '공이 날아오는 쪽'의 내적
r      = 0   if facing < 0.5                         # 날아오는 방향에서 ±60° 밖을 보고 있으면 0
```
- 접촉 전후로 reward 식이 같다. 접촉 후용 식(`fall_pos_reward`)은 주석 처리되어 있다.
- `fall_pos_reward`, `fall_vel_reward`, `_ball_contact_flag`, `_nearest`는 계산만 하고 reward에 쓰이지 않는다.
- 매 step `print('reward:', ...)`를 출력한다.

---

## 8. Ball Mastery — `HumanoidBallMastery`

2026-10-30 추가. 무작위 방향에서 오는 공을 **원터치**로 목표 방향·속력으로 보내는 task다.
`HumanoidDribble`을 상속해 공 actor, 공 물리 프로파일, target 샘플링을 그대로 쓰고
obs / reward / termination / 초기 상태만 바꾼다.

### 8.1 에피소드
- `episodeLength: 45` (1.5 s). 2026-10-31에 90 → 45. 터치까지 빨리 가도록 압박하고, epoch당 에피소드 수를 2배로 늘린다. `enableEarlyTermination: False` — 이 플래그는 **넘어짐 판정만** 제어한다. 터치 종료와 공 이탈 종료는 플래그와 무관하게 항상 켜져 있다.
- 공과 바닥 물리: `ballProfile: real_soccer` (1.4절).
- 목표는 에피소드 내내 고정이다. `headingChangeStepsMin/Max: 1000/1001`로 두어 상속한 `_update_task`의 재샘플링이 에피소드 안에 일어나지 않게 했다.

### 8.2 초기 세팅 — 두 가지 소스 [코드]
매 리셋마다 env별로 둘 중 하나를 쓴다 (`initBufferDribbleRatio: 0.2`).

1. **드리블식 스폰 (20 %, 버퍼가 비어 있으면 100 %)** — 5.2절과 동일. T-pose, 공은 캐릭터 기준 **반경 0.3~0.5 m 환형** 안의 랜덤 위치, 정지 상태. (2026-10-31 이전에는 ±0.5 m 정사각형이라 공이 몸 안에 생겼다. 8.8절)
2. **상태 버퍼 (80 %)** — "좋은 터치로 끝난 에피소드의 마지막 상태"를 캐릭터·공 **둘 다** 복원한다.

버퍼 사양:

| 항목 | 값 |
|---|---|
| 크기 | `initBufferSize: 65536` (FIFO), 균등 랜덤 샘플링 |
| 저장 조건 | 터치 품질(가중치 적용 전 `q`)이 `initBufferTouchRewardMin: 0.5` 초과 |
| 저장 항목 | root state(13 = 위치3+쿼터니언4+**선속도3+각속도3**), `dof_pos`, `dof_vel`, ball state(13, 동일 구성) — 약 82 float = 21 MB |
| 좌표계 | 캐릭터 루트 xy를 원점으로 옮겨 저장하고, 복원할 때 그 env의 기본 스폰 xy를 더한다 |

- `_rigid_body_*`는 저장하지 않는다. 물리에 반영되는 건 root state와 dof state뿐이고, 나머지는 AMP obs 초기화에만 쓰이는데 HLP는 `disc_reward_w = style_reward_w = 0`이라 AMP obs가 학습에 쓰이지 않는다 (2.2절).
- 월드 좌표를 그대로 저장하면 다른 env로 순간이동하므로 상대 좌표가 필요하다 (`envSpacing: 5`).

### 8.3 target / obs [코드]
- 목표 방향 θ ~ U(−π, π), 목표 속력 ~ U(0.1, 2) m/s. 에피소드당 1회.
- **목표 facing 방향**: `tar_dir` 기준 ±`tarFacingRangeDeg: 90`도 안에서 균등. 상속한 `_reset_task`는 `tar_dir`과 완전히 독립으로 뽑는데, 그러면 각도차가 U(0°, 180°)가 되어 **1/3이 백힐 구간**(120° 이상)이다. 원터치로는 사실상 불가능해 보상이 노이즈가 되므로 ±90°(정면 패스 ~ 인사이드 사이드 패스)로 제한했다. 넓히면 그대로 커리큘럼 2단계가 된다.
- task obs(**15**): `tar_dir`(2), `tar_speed`(1), `tar_facing_dir`(2), 공 상대위치(3, z는 절대 높이), 공 속도(3), 공 각속도(3), **접촉 플래그**(1). 방향·공 관련 값은 모두 캐릭터 heading 기준이다. 전체 obs는 223 + 15 = **238**.
- 접촉 플래그는 "지금 접촉 중"인 순간값이다. `_compute_observations`를 오버라이드해 맨 앞에서 접촉을 갱신하므로, obs·reward·reset이 모두 같은 step의 값을 본다. 원본 이식 코드처럼 `post_physics_step` 뒤에서 갱신하면 obs에 한 step 늦은 값이 들어간다.

### 8.4 접촉 판정 [코드]
공이 바닥이 아닌 휴머노이드 body에 닿았는지를 아래 세 조건의 AND로 본다.

| 신호 | 조건 | 값 |
|---|---|---|
| force | 공 rigid body의 net contact force 크기 `>` | `ball_contact_force_threshold: 4.412` |
| 캡슐 거리 | 공 표면과 가장 가까운 body 충돌 캡슐 표면 사이 거리 `<=` | `ball_contact_capsule_distance_threshold: 0.1` |
| 속도 변화 | 서브스텝 간 공 xy 속도 변화량 `>=` | `ball_contact_velocity_change_threshold: 0.0` (= 항상 통과) |

- **서브스텝 피크 추적**: `controlFrequencyInv: 2`라 한 컨트롤 스텝에 `simulate()`가 두 번 돈다. 끝에 한 번만 읽으면 서브스텝 안에서 생겼다 사라지는 터치를 놓치므로, `_physics_step`에서 매 서브스텝마다 텐서를 refresh하고 force 최대값 / 캡슐 거리 최소값 / 속도 변화 최대값을 모은다. 판정은 이 피크값으로 한다.
- `BallPhysicsMixin._physics_step`(MuJoCo 마찰 토크)에 합쳐 넣었다. 첫 `simulate` 전에는 root state를 refresh하지 않는다는 기존 제약을 그대로 지킨다 (모든 refresh가 `simulate` 뒤에 있다).
- `progress_buf == 1`(리셋 직후 첫 물리 스텝)은 무조건 False로 둔다. 솔버가 침투를 푸는 동안 힘·속도가 튀는 것은 터치가 아니다.
- 캡슐 치수는 `mjcf/amp_humanoid_shoe.xml`에서 옮겨 적은 하드코딩 테이블이다 (Isaac Gym에 런타임 조회 API가 없다). 발만 충돌체가 mesh라 MJCF의 touch box site로 근사했다.
- 감지 대상 body는 `ball_contact_capsule_body_names`로 따로 받는다. `contactBodies`는 넘어짐 종료에 쓰이므로 건드리지 않는다.
- 뷰어 모드에서는 접촉한 step에만 공이 빨간색이 된다.

### 8.5 termination [코드]
| 조건 | `reset` | `terminate` |
|---|---|---|
| 터치 종료 (직전 step 접촉 → 이번 step 비접촉) | 1 | 1 |
| 공 거리 > `relocate_dist: 3.0` | 1 | 1 |
| 90 step 시간 초과 | 1 | 0 (value bootstrap) |

터치 종료는 성공이지만 보상이 그 step에 모두 들어가므로 `terminate = 1`이 맞다.

**test 모드(`--test`)에서는 터치로 끝내지 않는다.** 대신 그 자리에서 새 목표(방향·속력)를 다시 뽑고 에피소드를 이어간다. "터치 → 새 목표 → 다시 터치"가 끊기지 않고 이어지는 것을 눈으로 보려는 용도다. 공 거리와 시간 초과 종료는 그대로다.
- 상속한 `_update_task`/`_reset_task`의 play 모드 분기는 조이스틱을 읽으므로 오버라이드해서 비활성화했다. test에서도 학습과 같은 랜덤 목표를 쓴다.
- 더 오래 이어서 보려면 `--episode_length`로 90을 늘리거나 `relocate_dist`를 키우면 된다.

### 8.6 reward [코드]
`q(v)`를 "공 xy 속도 `v`가 목표 속도와 얼마나 맞는가"라고 하면:

```
b_par = d · v ;  e_par = s* − b_par           # 목표 방향 성분의 절대 오차
e_tan = ‖v − b_par·d‖                         # 수직 성분 (성분 합이 아니라 norm)
q(v)  = exp(−1.5 · (1.0·e_par² + 0.1·e_tan²)) # velErrScale, velErrParW, velErrTanW

p(t)  = exp(−2·‖ball_xy − root_xy‖)           # DIST_ERR_SCALE

f(t)  = max(0, tar_facing_dir · 캐릭터 정면)   # 0~1

터치 전   r = 0.50 · (p(t) − p(t−1)) + 0.50 · (f(t) − f(t−1))   # distRewardW, faceShapeW
접촉 중   r = 0.50 · (q(v_t) − q(v_{t−1}))                      # contactRewardW
터치 끝   r = 1.00 · q(v_eval)      + 0.50 · f(t)               # touchRewardW, faceTouchW
```

dist와 접촉 중은 둘 다 **개선량(potential-based shaping)**이다. 에피소드 첫 step은 `prev`가 직전 에피소드의 값이라 제외한다 (접촉 플래그를 `progress_buf == 1`에서 끄는 것과 같은 이유). 빼면 에피소드마다 −0.27 정도의 가짜 보상이 들어간다.

- **오차를 `s*`로 나누지 않는다.** 드리블은 상대 오차 `(s*−b_par)/s*`를 쓰는데(5.5절), 그러면 같은 절대 오차 0.3 m/s가 `s*=2`에서 보상 0.98, `s*=0.1`에서 0.0001로 8000배 갈린다. 목표 속력 범위가 [0.1, 2]라 저속 명령에 학습 신호가 남지 않는다.
- **터치 전에는 공 속도 보상을 주지 않는다.** 그 시점의 공 속도는 에이전트가 만든 값이 아니다. 특히 `s*`가 낮으면 정지한 공이 그대로 만점에 가까워서, 터치하면 에피소드가 끝나는 구조와 맞물려 "건드리지 않기"가 최적 전략이 된다.
- **dist도 개선량이어야 한다.** 값을 그대로 주던 때(`distRewardW: 0.05`) 실제 학습에서 `reward_terms/touch`가 우하향하고 `dist`가 우상향했다. 원인은 dist만 누적형이라는 것 — 공이 0.3 m 앞이면 step당 `0.05·exp(−0.6) = 0.0274`, 90 step이면 **2.47**로 touch 상한 1.0을 넘는다. 터치하면 에피소드가 끝나 그 수입이 끊기므로, 한 step 미루는 이득(+0.0274)이 터치 보상의 할인 손실(`0.01·q ≤ 0.01`)보다 **2.7배** 커서 **"공 옆에서 안 건드리고 맴돌기"가 최적해**였다. 명목 가중치는 `touch(1.0) > contact(0.5) > dist(0.05)`인데 에피소드 합은 `dist(2.47) > touch(1.0) > contact(0.5)`로 순서가 뒤집혀 있었다.
  - 개선량으로 바꾸면 합이 `p(접촉 직전) − p(시작)`로 telescoping되어 **머무르는 이득이 정확히 0**이다. 접근할 때만 보상이 나오므로 유도 신호는 그대로다. 0.5 m → 0.2 m 접근이면 합 0.15.
  - 공을 놓치면(3 m 이탈) 합이 음수가 되어 자연히 벌점이 된다. 0.3 m → 3.0 m 이탈이 −0.27.
  - `normalize_value` / `normalize_advantage`가 켜져 있어 절대 스케일은 정규화된다. 항들 사이의 **상대** 크기만 의미가 있다.
- `velErrTanW: 0.1`은 5.5절 ⚠의 뒤바뀜을 고친 방향이다. 드리블 원본 가중치는 반대(평행 0.1 / 수직 1.0)이고, 그 조합도 yaml에서 바로 넣어볼 수 있다.
- **접촉은 길게 유지해도 된다** (순간 원터치만을 고집하지 않는다). 접촉 중에 공 속도를 목표에 가깝게 만들어 가다가 끝내는 것이 학습 대상이다.
  - 그래서 접촉 중 보상은 `q`의 **값이 아니라 개선량**이다. 접촉이 step a~b 동안 이어지면 합이 `q(v_b) − q(v_{a−1})`로 telescoping되어, **접촉 길이와 무관하게 1 이하로 묶인다.** 올렸다 내렸다 반복해도 순이득이 0이라 농사가 불가능하다.
  - 값 그대로 줬다면 `gamma: 0.99`, 90 step 기준 누적 상한이 `(1−0.99⁹⁰)/0.01 = 59.5`배라, `contactRewardW ≥ touchRewardW / 59.5 = 0.0168`에서 **"접촉을 영원히 끝내지 않기"가 최적해**가 된다. 그러면 에피소드가 매번 시간 초과로 끝나고 상태 버퍼도 채워지지 않는다.
  - 개선량이라 보상이 음수가 될 수 있다. 접촉으로 공 속도를 목표에서 더 멀어지게 하면 벌점이다.
- **facing 보상** (2026-10-31 추가): dist와 같은 개선량 + touch와 같은 종료 보상을 함께 준다. 매 step `f`를 그대로 주면 에피소드 합이 `0.5 × 45`가 되어 **"목표 방향 보고 가만히 서 있기"가 최적해**가 되므로 절대 그렇게 하면 안 된다. `q`에 곱하지 않고 더하는 이유는, 곱하면 희소 조건 두 개가 AND로 묶여 초기 학습이 막히기 때문이다.
- `v_eval`은 `touchEvalMode`로 고른다.
  - `rolling` (기본): 미끄럼 → 구름 전이 후의 속도 `v_roll = (5·v − 2R(ẑ × ω)) / 7`. 접촉점 기준 각운동량 보존이라 마찰계수와 무관하다. 지면에 있는 공에만 유효해서, 떠 있으면 `v`를 그대로 쓴다. 전이 후의 굴림 마찰 감속은 무시한다.
  - `instant`: 터치가 끝난 step의 공 속도 그대로.
  - 어느 쪽이 "터치 후 실제 공 속도"를 더 잘 맞히는지 실측한 결과는 8.7절에 있다.
- 네 성분은 `extras["reward_terms"]`로 넘겨 `reward_terms/{dist,contact,touch,face}`, `reward_terms_step/*`로 기록된다. 터치가 없으면 `reward_terms/touch`가 0이므로 이 값이 사실상 터치 성공률 지표다.

### 8.7 `touchEvalMode` 실측 [실측]
2026-10-30. 터치 종료 시점의 두 예측값을 **H step 뒤의 실제 공 xy 속도**와 비교했다.
터치로 종료하지 않고 계속 시뮬레이션하는 임시 서브클래스로 512 env × 25 epoch을 돌려
4019 샘플을 모았다 (학습 전 랜덤 정책이라 터치 속도 중앙값 0.81 m/s, 각속도 중앙값 5.5 rad/s).

H = 6 step (0.2 s) 기준:

| 구간 | n | A(instant) 평균오차 | C(rolling) 평균오차 | C 승률 |
|---|---|---|---|---|
| 공중 (C가 `v` 그대로 사용) | 537 (50 %) | — 동일 — | — 동일 — | – |
| 지면, 터치 속도 0.3~1.0 m/s | 160 | 0.230 m/s | 0.204 m/s | 34 % |
| 지면, 터치 속도 1.0~2.0 m/s | 206 | 0.425 m/s | **0.244 m/s** | 66 % |
| 지면, 터치 속도 2.0 m/s~ | 125 | 0.891 m/s | **0.318 m/s** | 89 % |

- 터치 종료의 절반은 공이 떠 있어 두 방식이 같은 값을 낸다. **차이는 지면 터치에서만 생긴다.**
- 지면 터치에서 C가 더 정확하고, **터치가 빠를수록 격차가 커진다** (2 m/s 이상에서 오차 64 % 감소). 스핀이 실린 빠른 터치일수록 `instant`가 어긋난다는 예상과 맞는다.
- 보상 환산: 터치 속도 1 m/s 이상 구간의 평균 오차가 A 0.617 / C 0.379 m/s다. 이 오차가 평행 성분에 그대로 실린다고 보면 터치 보상 `q`가 각각 최대 43 % / 19 % 어긋난다.
- 목표 속력 범위가 [0.1, 2] m/s라 1~2 m/s 구간이 가장 중요하다. → **기본값을 `rolling`으로 정했다.**
- 한계: 학습 전 랜덤 정책의 터치라 실제 킥보다 약하고 지저분하다. 다만 "(v, ω)에서 이후 속도를 예측하는 정확도"는 정책과 무관한 물리 성질이다.
- 저속(0~0.3 m/s) 구간은 두 방식 모두 중앙값 오차가 0.133 m/s인데, 이는 목표 속력 하한 0.1 m/s보다 크다. 저속 명령이 측정 오차에 묻힌다는 뜻이다.

### 8.8 리셋 직후 공 속도 스파이크 [실측]
2026-10-31. 리셋으로 상태를 복원하면 PhysX 솔버가 접촉을 cold 로 다시 풀기 때문에, 공이 몸 안에 겹쳐 있으면 침투 해소 임펄스가 공을 차 버린다. **리셋 직후 1 step 째 공 xy 속도 변화**를 1024 env × 40 epoch 으로 측정했다 (에이전트가 만들지 않은 가짜 임펄스의 크기).

| 초기화 | 1 step \|Δv\| 중앙값 | 95 % | 최대 |
|---|---|---|---|
| 드리블 스폰 — **정사각형 (수정 전)** | 0.0000 | **3.26** | 11.5 m/s |
| 드리블 스폰 — **환형 0.3~0.5 m (수정 후)** | 0.0000 | **0.0000** | **0.0000** |
| 버퍼 (touch_end 저장, 수정 전) | 0.0056 | 0.336 | 8.55 |
| 버퍼 (touch_end 저장, 수정 후) | 0.0000 | **0.0327** | 7.78 |
| 버퍼 (접촉 중 저장, 참고) | 0.0298 | 1.433 | 13.4 |

- **원인**: `_reset_ball`이 `root_pos + rand(0,1) − 0.5`로 캐릭터 중심 1 m 정사각형 안에 공을 놓아서, **다리·발 캡슐 안쪽에도 스폰**됐다. 중앙값은 0이지만 5 %가 3 m/s 넘는 발길질을 받고 시작했다. 목표 속력 범위가 [0.1, 2] m/s라 이 크기는 치명적이다.
- **수정**: 최소 반경 `INIT_BALL_DIST_MIN: 0.3`을 둔 환형 샘플링. 공 반지름 0.11 + 발/정강이 캡슐 반지름 0.05 = 0.16 m 에 발이 루트보다 앞으로 나온 길이를 더해 잡았다. 스파이크가 **완전히 사라졌다**. 드리블·드리블 Fix·Ball Mastery 가 같은 `_reset_ball`을 공유하므로 셋 다 적용된다.
- 버퍼 쪽도 함께 좋아졌다 (95 % 0.336 → 0.0327). 스폰이 깨끗해지니 에피소드가 깨끗해지고, 저장되는 종료 상태도 깨끗해진다.
- **접촉 중 상태를 버퍼에 저장하는 것(향후 과제)은 그대로 하면 안 된다.** 저장 시점 캡슐 표면거리 중앙값 −0.0029 m, **침투 58.1 %**라 95 % 스파이크가 1.43 m/s로 목표 속력 범위 한가운데다. 저장 시 `_ball_capsule_distance_min > 0` 필터를 걸면 (접촉 상태의 41.9 % 가 통과) 침투 자체가 없어진다. 침투를 표면거리 0으로 보정해 저장하는 방법도 있으나 9절 #16 참고.
- 한계: 학습 전 랜덤 정책이라 터치가 지저분하다. 다만 "저장된 자세에 침투가 있나"는 기하학적 성질이라 정책 의존성이 낮다.

### 8.9 남은 과제
- **force 임계값의 근거**: 9절 #15대로 공의 net contact force가 m·g의 0.5배~28배로 읽힌다. 그렇다면 `force > 4.412`는 사실상 항상 참이고 판정이 캡슐 거리 단독으로 수렴한다. 임계값은 눈으로 검증된 값 그대로 두되, 세 조건 각각의 통과율을 재서 확인해야 한다.
- **버퍼 초기화의 샘플 효율**: 버퍼 상태는 공이 최대 2 m/s로 멀어지는 중이라, 3 m 종료까지 약 1.25 s 안에 따라잡아야 한다. 터치 없이 끝나는 비율이 높으면 롤아웃이 낭비된다. 초기화 소스별 터치 성공률을 봐야 한다.
- 목표 속력 하한 0.1 m/s는 8.7절의 저속 구간 측정 오차(0.133 m/s)보다 작다. 저속 명령이 학습되지 않으면 하한을 올리는 것을 검토한다.

---

## 9. 확인이 필요한 사항 모음

| # | 내용 | 위치 |
|---|---|---|
| 1 | `disc_coef: 5관` (문자열) → LLC 학습 시 TypeError 예상 | `calm_humanoid.yaml:98` |
| 2 | HLP 학습 설정이 테스트용 값으로 보임 (`max_epochs: 10`, `minibatch_size: 128`, `clip_value: Falsehrl`) | `hrl_humanoid.yaml` |
| 3 | LLC 학습 motion 데이터셋 (`soccer_90.yaml` vs `soccer_9.yaml`) | README / `run_train_soccer.sh` |
| 4 | LLC는 box 발, HLP는 shoe mesh 발 (발 질량 1.0 → 0.48 kg) | 1.3절 |
| 5 | ~~Dribble/Kick 공 5.575 kg vs Trap/데모 0.45 kg~~ → 2026-10-28 0.45 kg으로 통일. 기존 Dribble/Kick 체크포인트는 재학습 필요 | 1.4절 |
| 6 | rolling/torsion friction이 시뮬레이션에 적용되지 않음 (동작 실험으로 확인). Kick/Trap 공의 감속은 damping으로만 이루어짐. Dribble은 2026-10-29부터 MuJoCo식 외부 토크로 대체 | 1.3절, 1.4절 |
| 7 | Move: 50000 step, early termination OFF → 넘어져도 reset되지 않음 | 4절 |
| 8 | Move/Dribble 수직 오차가 벡터 크기가 아닌 성분 합 (`torch.sum`) → 목표 방향 θ에 따라 패널티가 0~1.41배로 달라짐 (θ=45°/225°는 패널티 0). `HumanoidDribble_Fix`에서 norm으로 수정했다가 2026-11-01 되돌림 (현재 수정된 task 없음) | 4.5절, 5.5절, 5.6절 |
| 9 | Kick `tar_contact_forces`가 공이 아니라 torso(body 1)를 가리킴 | 6.4절 |
| 10 | Kick: 다른 env의 reset이 진행 중인 env의 `ori_direc`를 다시 계산함 | 6.2절 |
| 11 | Trap: torso/head 목표일 때 공이 닿는 즉시 terminate | 7.6절 |
| 12 | Trap: 최대 비행시간(약 2.65 s)이 에피소드 길이(2 s)보다 김 | 7.4절 |
| 13 | 데모(`fifa.py`) 캐릭터 angular_damping 0.5 vs 학습 0.01 | 1.3절 |
| 14 | Dribble 공/바닥이 `real_soccer`로 바뀌어 기존 Dribble 체크포인트(예: `87.pth`)는 다른 공으로 학습된 것. 기존 체크포인트를 볼 때는 `ballProfile: dribble_legacy`로 바꿔야 함 | 1.4절, `tools/run_view_soccer.sh` |
| 15 | net contact force가 캐릭터가 넘어진 env에서 바닥에 가만히 있는 공에 대해 0.5배~28배 m·g로 읽힘. 공 contact force를 쓰는 코드(Trap의 `_ball_contact_forces` 등)도 영향을 받을 수 있음 | 1.4절 |
| 16 | Dribble 뷰어 모드: `_draw_task`가 simulate마다 공 root state를 다시 씀. MuJoCo식 마찰이 켜져 있으면 두 번째 simulate 전에 state를 refresh하므로 최신 값이 쓰이지만, 꺼져 있으면 첫 simulate 전 값으로 되돌아감 (기존 동작) | 5절 |
| 17 | 접촉 중 상태를 초기 상태 버퍼에 저장하면 58 %가 침투 상태라 리셋 때 공이 최대 13 m/s로 튄다. `_ball_capsule_distance_min > 0` 필터로 피할 수 있다. 침투를 표면거리 0으로 보정해 저장하는 방법은 (a) 가장 깊은 접촉 하나만 풀어서 다리 사이에 낀 공은 반대쪽으로 밀리고, (b) 공만 옮기고 속도는 그대로라 접촉의 물리적 일관성이 깨지며, (c) 캡슐 테이블은 실제 충돌 메시의 근사라 보정량 자체가 부정확하다. 필터 쪽이 낫다 | 8.8절 |

---

## 10. 변경 이력

| 날짜 | 변경 | 파일 |
|---|---|---|
| 2026-10-28 | Dribble 공 `density = 80.71` 추가 (5.575 → 0.45 kg) | [humanoid_dribble.py:222](calm/env/tasks/soccer_hlc/humanoid_dribble.py#L222) |
| 2026-10-28 | Kick 공 `density = 80.71` 추가 (5.575 → 0.45 kg) | [humanoid_ball_pos_task_1.py:132](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L132) |
| 2026-10-29 | 공 물리 프로파일 + MuJoCo식 torsion/rolling 마찰 추가 (`BallPhysicsMixin`, `MujocoFriction`) | [ball_physics.py](calm/env/tasks/soccer_hlc/ball_physics.py), [ball/](calm/data/cfg/soccer_hlc/ball/) |
| 2026-10-29 | Dribble 공/바닥을 `real_soccer` 프로파일로 변경 (0.43 kg 속이 빈 공, lin 0.1, ang 0, max ω 300, friction 0.2, restitution 0.88, MuJoCo torsion 0.005 / rolling 0.015 m; 바닥 1.0/1.0/0.2 유지). yaml의 `plane:`은 프로파일 `ground:`로 이동 | [humanoid_dribble.py](calm/env/tasks/soccer_hlc/humanoid_dribble.py), [humanoid_dribble.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble.yaml) |
| 2026-10-29 | `ball_fifa_test_vis.py`가 `MujocoFriction`을 공용 모듈에서 import. N을 net contact force 대신 m·g(바닥 접촉 시)로 계산 (real_soccer 굴림 6.31 → 6.26 m, 회전 정지 1.65 → 1.63 s) | [ball_fifa_test_vis.py](ball_fifa_test_vis.py) |
| 2026-10-29 | Dribble 학습 로깅 추가 (학습 동작 변화 없음): reward 항목별 에피소드 합 `reward_terms/ball`, `humanoid`, `dist`, `total` (가중치 0.6/0.2/0.2 적용 후 값, total = `rewards0`). `add_reward`(`rewards_<pid>.json` 기록) 삭제. 매 epoch 갱신되는 step당 평균 `reward_terms_step/*` 추가 (에피소드 값은 모든 env가 300 step에서 동시에 끝나 약 9.4 epoch마다만 바뀜). 에피소드 길이는 기존 `episode_lengths/*`. wandb 프로젝트 `hlp-dribble`. wandb는 TensorBoard 동기화 대신 epoch당 `wandb.log` 한 번 (step = epoch, `frame` 함께 기록; `WandbEpochWriter`) | [run.py](calm/run.py), [humanoid_dribble.py](calm/env/tasks/soccer_hlc/humanoid_dribble.py), [run_train_hlc.sh](tools/run_train_hlc.sh) |
| 2026-10-30 | `HumanoidDribble_Fix` 추가 (`HumanoidDribble` 상속, 원본은 그대로). 수직 속도를 성분 합 → norm, task obs에서 `tar_facing_dir` 제외 (14 → 12), `pos_err_scale` 10.0 → 2.0. 정규화 분모 clamp와 정지 명령(`tar_speed = 0`)은 제외했다. 5.6절 참고 | [humanoid_dribble_fix.py](calm/env/tasks/soccer_hlc/humanoid_dribble_fix.py), [parse_task.py](calm/utils/parse_task.py) |
| 2026-10-30 | `HumanoidDribble_Fix`에 early termination 추가: 공이 캐릭터에서 `relocate_dist`(3.0 m)보다 멀어지면 종료. 전용 cfg_env `humanoid_dribble_fix.yaml` 추가(`humanoid_dribble.yaml` 복사본, `relocate_dist` 2.0 → 3.0만 다름). 원본 `HumanoidDribble`과 그 cfg는 변경 없음. 5.6절 #4 참고 | [humanoid_dribble_fix.py](calm/env/tasks/soccer_hlc/humanoid_dribble_fix.py), [humanoid_dribble_fix.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble_fix.yaml) |
| 2026-10-30 | `HumanoidBallMastery` 추가 (`HumanoidDribble` 상속). 원터치로 공을 목표 방향·속력으로 보내는 task. 공-휴머노이드 접촉 감지(서브스텝 피크, 캡슐 거리), 터치 종료/공 3 m early termination, 좋은 터치의 마지막 상태를 모으는 초기 상태 버퍼, 절대 오차 기반 ball reward. 8절 참고 | [humanoid_ball_mastery.py](calm/env/tasks/soccer_hlc/humanoid_ball_mastery.py), [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml), [parse_task.py](calm/utils/parse_task.py) |
| 2026-10-30 | Ball Mastery `touchEvalMode` 기본값을 `instant` → `rolling` 로 (지면 터치 4019 샘플 실측, 터치 속도 1 m/s 이상에서 예측 오차 0.43 → 0.24 m/s). `contactRewardW` 0.1 → 0.01 (접촉을 끝내지 않는 것이 최적해가 되지 않도록). 8.6 / 8.7절 참고 | [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml) |
| 2026-10-30 | Ball Mastery 접촉 중 보상을 `q` 값 → **개선량** `q(v_t) − q(v_{t−1})` 으로 변경. 접촉 길이와 무관하게 합이 1 이하로 묶여 "접촉을 끝내지 않기"가 최적해가 되는 문제가 사라진다. `contactRewardW` 0.01 → 0.5. 8.6절 참고 | [humanoid_ball_mastery.py](calm/env/tasks/soccer_hlc/humanoid_ball_mastery.py), [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml) |
| 2026-10-30 | Ball Mastery `default_buffer_size_multiplier` 10.0 → 2.0. 8 GB RTX 2070에서 4096 env 학습 중 PhysX GPU 할당 실패(`PxgCudaDeviceMemoryAllocator ... Result = 2`)로 죽는 문제. 피크 7729 → 7364 MiB (여유 463 → 828 MiB), 처리량 변화 없음. 접촉 감지기의 추가 메모리는 Dribble 대비 70 MiB뿐이라 원인이 아니다 | [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml) |
| 2026-10-30 | Ball Mastery **dist 보상도 개선량** `p(t) − p(t−1)` 으로 변경 (`p = exp(−2·거리)`), `distRewardW` 0.05 → 0.5. 값 그대로 주던 때는 dist만 누적형이라 에피소드 합 2.47 로 touch(1.0)를 넘어, 터치를 미루고 공 옆에 머무는 것이 최적해였다 (`reward_terms/touch` 우하향의 원인). 에피소드 첫 step은 `prev` 오염 때문에 제외. 8.6절 참고 | [humanoid_ball_mastery.py](calm/env/tasks/soccer_hlc/humanoid_ball_mastery.py), [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml) |
| 2026-10-31 | **공 스폰을 환형으로** (`INIT_BALL_DIST_MIN: 0.3` 추가). 기존 `root_pos + rand(0,1) − 0.5` 정사각형 샘플링은 공이 캐릭터 다리·발 캡슐 안에 생겨, 리셋 직후 PhysX 침투 해소 임펄스가 공을 찼다 (1 step 공 xy Δv 95 % **3.26 m/s**, 최대 11.5). 수정 후 스파이크가 **완전히 사라졌다**(95 % 0.0000). Dribble / Dribble Fix / Ball Mastery 공용. 8.8절 참고 | [humanoid_dribble.py](calm/env/tasks/soccer_hlc/humanoid_dribble.py), [humanoid_dribble.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble.yaml), [humanoid_dribble_fix.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble_fix.yaml), [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml) |
| 2026-10-31 | Ball Mastery **target facing 추가**. task obs 13 → 15 (`tar_facing_dir` 2, heading 기준), 보상에 `faceShapeW: 0.5` 개선량 + `faceTouchW: 0.5` 종료 항 추가. facing 목표는 `tar_dir` 기준 ±`tarFacingRangeDeg: 90`도 안에서 뽑는다 (상속한 독립 샘플링은 1/3이 백힐이라 원터치 불가). 매 step 값을 주면 "목표 방향 보고 서 있기"가 최적해가 되므로 개선량 + 종료 형태로 했다. 8.3 / 8.6절 참고 | [humanoid_ball_mastery.py](calm/env/tasks/soccer_hlc/humanoid_ball_mastery.py), [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml) |
| 2026-10-31 | Ball Mastery `episodeLength` 90 → 45 (3 s → 1.5 s). 터치까지 빨리 가도록 압박하고 epoch당 에피소드 수를 2배로 늘린다. 8.1절 참고 | [humanoid_ball_mastery.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml) |
| 2026-11-01 | `HumanoidDribble_Fix` 수직 속도를 norm → **성분 합(`torch.sum`)으로 되돌림** (ball/humanoid reward 양쪽, 원본 `HumanoidDribble`과 동일). obs 제외, `pos_err_scale` 2.0, 공 거리 종료는 그대로. 5.6절 #1 참고 | [humanoid_dribble_fix.py](calm/env/tasks/soccer_hlc/humanoid_dribble_fix.py) |
| 2026-11-02 | `HumanoidDribbleFace` 추가 (`HumanoidDribble_Fix` 상속). obs 14 (`tar_facing_dir` 포함), target은 리셋 때 캐릭터→공 방향 = face, 반대 = ball dir, 속도 U[0.5, 3], 에피소드 중 고정. reward = ball/humanoid(Fix와 같음) + face `0.5·[(1−|θ|/π)+exp(−4θ²)]` + dist `exp(−4d)` (공이 뒤쪽 절반이면 각도에 비례한 음수). 공 5 m 종료. 5.7절 참고 | [humanoid_dribble_face.py](calm/env/tasks/soccer_hlc/humanoid_dribble_face.py), [humanoid_dribble_face.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble_face.yaml), [parse_task.py](calm/utils/parse_task.py), [run_train_hlc.sh](tools/run_train_hlc.sh) |
| 2026-11-02 | `HumanoidDribbleFace` face reward를 `0.5·[(1−|θ|/π)+exp(−4θ²)]` → **`exp(−4·|θ|/π)`** 단일 커널로 (`FACE_ERR_SCALE` 4.0 그대로). 5.7절 참고 | [humanoid_dribble_face.py](calm/env/tasks/soccer_hlc/humanoid_dribble_face.py) |
| 2026-11-02 | `HumanoidDribbleFace` ball과 face를 **곱으로 묶음**: `BALL_FACE_REWARD_W(0.7)·r_ball·(α + (1−α)·r_face)`, α = `FACE_FLOOR` 0.0 (순수 곱). `FACE_REWARD_W` 삭제, `BALL_REWARD_W` 0.5 → 1.0 (가중치 전 값). 로깅 `ball`/`face` → `ball_face`. 5.7절 참고 | [humanoid_dribble_face.py](calm/env/tasks/soccer_hlc/humanoid_dribble_face.py), [humanoid_dribble_face.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble_face.yaml) |
| 2026-11-03 | Dribble / Dribble Fix / Dribble Face `default_buffer_size_multiplier` 10.0 → 2.0. Ball Mastery(같은 캐릭터 1 + 공 1, 4096 env)의 2026-10-30 실측(처리량 36.2k → 36.5k fps, GPU 피크 7729 → 7364 MiB)을 근거로 했고, Dribble에서 따로 재측정하지는 않았다 | [humanoid_dribble.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble.yaml), [humanoid_dribble_fix.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble_fix.yaml), [humanoid_dribble_face.yaml](calm/data/cfg/soccer_hlc/humanoid_dribble_face.yaml) |
| 2026-11-04 | Dribble / Dribble Fix / Dribble Face dist reward 기준점을 `root` → **`root + DIST_FRONT_OFFSET · heading 방향`** (`compute_dist_anchor`). yaml `DIST_FRONT_OFFSET` 기본 0.0 = 기존과 같음 (수치 확인). Face의 앞/뒤 판정은 root 기준 그대로 | [humanoid_dribble.py](calm/env/tasks/soccer_hlc/humanoid_dribble.py), [humanoid_dribble_fix.py](calm/env/tasks/soccer_hlc/humanoid_dribble_fix.py), [humanoid_dribble_face.py](calm/env/tasks/soccer_hlc/humanoid_dribble_face.py), dribble yaml 3개 |
| 2026-11-04 | `HumanoidTrap_Fix` 추가 (`HumanoidTrapping_3` 상속, 설계는 [trap_fix_plan.md](trap_fix_plan.md)). 공 `linear_damping_and_mjfriction`, 공중 / 땅볼 반반, 공중볼 감쇠 해석해 발사 (착지 오차 최대 1 cm 실측), 땅볼 순수 구름 닫힌 식 (도착 속도 2.6–15.2 m/s 실측), 트래핑 판정 공 접촉력 / 속도 급변, 실패 넘어짐·핸드볼·착지·통과, episodeLength 계산 (99 step), obs 17. 종료 사유를 `episode_stats/` 로 기록하도록 RLGPUAlgoObserver 확장 | [humanoid_trap_fix.py](calm/env/tasks/soccer_hlc/humanoid_trap_fix.py), [humanoid_trap_fix.yaml](calm/data/cfg/soccer_hlc/humanoid_trap_fix.yaml), [parse_task.py](calm/utils/parse_task.py), [run.py](calm/run.py), [run_train_hlc.sh](tools/run_train_hlc.sh) |
| 2026-11-06 | `HumanoidKick_Fix` 추가 (`HumanoidBallPos_1` 상속, 설계는 [kick_fix_plan.md](kick_fix_plan.md)). 드리블 상태 파일 초기화 `INIT_FILE_PROB` 0.8, 목표 방향을 공 기준으로 직접 샘플해 에피소드 동안 고정 (방위 ±45°, 고각 U(0°, 45°), 속력 8–30), obs 13 (+ 공 각속도), 킥 판정 = 바닥 마찰 원뿔로 설명되지 않는 공 충격량 > 0.5 m/s & 속력 ≥ 5 (반사실 유령 공 대비 접촉 없는 오검출 0, 킥 recall 99 %), 발 / 정강이 외 접촉 > 1 N 종료, 공 `linear_damping_and_mjfriction` | [humanoid_kick_fix.py](calm/env/tasks/soccer_hlc/humanoid_kick_fix.py), [humanoid_kick_fix.yaml](calm/data/cfg/soccer_hlc/humanoid_kick_fix.yaml), [parse_task.py](calm/utils/parse_task.py), [run_train_hlc.sh](tools/run_train_hlc.sh) |
| 2026-11-19 | `HumanoidMoveStyle` 이식 (원본 `PhysicsFC_project/calm`에서 task와 agent를 그대로 복사). algo `hrl_soccer_conditioned` 등록 (player는 기본 `HRLPlayer`), `hrl_humanoid_style_control.yaml`에 player용 키 추가. task 0.5 + style 0.5, llc_steps 5. 4.6절 참고. 512 env 학습 44 epoch, `move_style_27.pth` test 실행 확인 | [humanoid_move_style.py](calm/env/tasks/soccer_hlc/humanoid_move_style.py), [hrl_soccer_conditioned_agent.py](calm/learning/hrl_soccer_conditioned_agent.py), [hrl_humanoid_style_control.yaml](calm/data/cfg/train/rlg/hrl_humanoid_style_control.yaml), [parse_task.py](calm/utils/parse_task.py), [run.py](calm/run.py), [run_train_hlc.sh](tools/run_train_hlc.sh) |
