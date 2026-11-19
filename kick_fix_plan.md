# Kick_Fix — HumanoidBallPos_1 을 고친 킥 태스크

- 원본: [humanoid_ball_pos_task_1.py](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py), [humanoid_ball_pos_task_1.yaml](calm/data/cfg/soccer_hlc/humanoid_ball_pos_task_1.yaml) (건드리지 않음)
- 구현: [humanoid_kick_fix.py](calm/env/tasks/soccer_hlc/humanoid_kick_fix.py) — `HumanoidKick_Fix(BallPhysicsMixin, HumanoidBallPos_1)`, [humanoid_kick_fix.yaml](calm/data/cfg/soccer_hlc/humanoid_kick_fix.yaml)

## 1. 원본의 문제

| # | 문제 | 위치 |
|---|---|---|
| 1 | `_reset_target` 이 `_ori_direc`, `_ori_dist`, `_ori_z`, `_fastest` 를 **전체 env** 에 대해 다시 계산. 학습 중 다른 env 가 리셋될 때마다 진행 중인 env 의 목표 방향이 "지금 공 위치 → 타겟" 으로 바뀐다 (50 m 라 변화는 몇 도) | [:334-339](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L334-L339) |
| 2 | 목표 방위각이 root 기준 50 m 지점이라 공 기준 방향 범위가 정확히 ±45° 가 아님. 고각은 높이 U(0, 50) 이라 atan 분포 (45° 쪽이 0° 쪽의 2 배 밀도) | [:307-322](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L307-L322) |
| 3 | `IS_INIT` 분기의 타겟 거리가 `_ball_dist_min`(1) ~ 50 m, 방위 지터 ±22.5°. 기본 분기와 다른 분포 | [:262-281](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L262-L281) |
| 4 | `touched` = 공 속력 > 0.5 m/s. 드리블 상태에서 시작하면 81 % 가 첫 step 에 touched → 10 step 뒤 무조건 종료 | [:439-442](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L439-L442) |
| 5 | `_tar_contact_forces` 가 공이 아니라 humanoid body 1 의 접촉력 (`[..., num_actors - 2, :]` 은 actor 가 아니라 body index 1). `tar_fail` 의 "공 접촉" 조건이 실제로는 상체 접촉 | [:217](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L217) |
| 6 | 공 각속도가 관측에 없음 (계산만 하고 안 씀) | [:535](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L535) |
| 7 | 매 step `print('reward:')` | [:455](calm/env/tasks/soccer_hlc/humanoid_ball_pos_task_1.py#L455) |

## 2. 설계

### 2.1 초기 상태
- env 마다 `INIT_FILE_PROB` (0.8) 확률로 `INIT_FILES` 중 하나 (균등) 에서 캐릭터 + 공 상태를 그대로 가져온다. 기본은 `dribble_init_state_v3.pkl` (드리블, 공 바닥) + `trapping_ball_state.pkl` (트래핑 접촉 직후, 공 대부분 공중). 나머지는 원본 기본 배치 (캐릭터 앞쪽 반원 0.5–3 m, 공 정지).
- trapping 파일 시작 중 약 7 % 는 공이 발 / 정강이 외 body (허벅지, 몸통 등) 에 닿아 있어 3 step 안에 body_contact 로 끝난다 (0 동작 측정).
- 파일 샘플은 거르지 않는다 (공을 놓친 샘플, 공이 떠 있는 샘플 포함).

### 2.2 목표 (리셋 때 뽑아 에피소드 동안 고정, 리셋하는 env 만)
```
az    = atan2(공 - root) + U(-TAR_AZ_RANGE, TAR_AZ_RANGE)      # 45°
elev  ~ U(TAR_ELEV_MIN, TAR_ELEV_MAX)                          # 0–45°, 각도 균등
dir   = (cos elev cos az, cos elev sin az, sin elev)
speed ~ U(TAR_SPEED_MIN, TAR_SPEED_MAX)                        # 8–30 m/s
타겟 표시 = 공 + TAR_VIS_DIST · dir                              # 50 m, 보상 / 관측에는 안 씀
```
파일 / 기본 초기화 모두 같은 식. play mode 조이스틱은 원본처럼 왼쪽 스틱 = 방위각 (world), 크기 = 고각, 트리거 = 속력 (뜬 공 목표만).

땅볼 목표: `GROUND_PROB` (0.4) 확률로 고각 0, 속력 U(`GROUND_SPEED_MIN`, `GROUND_SPEED_MAX`) (8–20 m/s), 방위각은 같은 식. 뜬 공 고각 하한 `TAR_ELEV_MIN` 을 10° 로 두어 관측의 목표 방향만으로 땅볼과 구별된다. `GROUND_PROB` 0 이면 뜬 공 목표만.

### 2.3 관측 (13)
공 위치 3 (z 는 절대 높이) + 공 속도 3 + **공 각속도 3** + 목표 속력 1 + 목표 방향 3, 모두 heading 기준.

### 2.4 보상
원본과 같은 식: `exp(-10 |v_ball / speed - dir|^2)`. 단 킥 판정 step 부터만 주고 킥 전은 0 (원본은 처음부터). 로그 `reward_terms/vel`.
땅볼 목표는 수평 성분만 `exp(-10 |v_xy / speed - dir_xy|^2)` 에 높이 계수를 곱한다. 계수는 공 아래쪽 높이 `GROUND_EPS` (2 cm) 이하 1, `GROUND_HEIGHT_MAX` (0.3 m) 까지 선형으로 0. 처음에는 2 cm 넘게 뜨면 0 이었는데, 덜 띄울수록 나아지는 기울기가 없어 바꿨다. 0.3 m 면 2–4° 로 깔린 공은 거의 안 깎이고, 6–8° (14–20 m/s) 로 띄운 공은 창 평균 15–50 % 깎인다 (중력만 고려한 계산). 평가 구간은 뜬 공과 같이 킥 step + 창 10 step.
측정 (보상 변경 전 정책 `kickfix-initstate-rewardafterkick-max30`): 고각 0–5° 목표에도 공을 중앙값 6–8° 로 띄워 찼다. 뜬 공은 창 동안 마찰이 없어 속력을 유지하고, 5–7° 방향 오차는 보상을 7–14 % 만 깎아서다. 새 보상에서 같은 정책의 땅볼 목표 보상 합은 2.6 (뜬 공 목표 6.8–7.6).
킥 전 보상은 학습된 정책에서 전체의 0.1 % 였지만, 공을 5 m/s 미만으로 목표 방향에 계속 굴리면 킥보다 큰 보상을 받을 수 있는 구조라 막았다. `disc_reward_w` / `style_reward_w` 가 0 이라 킥을 미룰 유인은 없다.

### 2.5 킥 판정
매 control step, 공이 받은 접촉 충격량 (질량당) `J = Δv + g·dt·ẑ` 에서 **바닥으로 설명되지 않는 부분**을 본다.
- 이번 step 에 바닥에 닿았을 수 있으면 (직전 z, 지금 z, 자유낙하 외삽 z 중 하나라도 바닥 근처): 바닥은 `J_z >= 0`, `|J_xy| <= μ J_z` 인 원뿔 안의 충격량만 줄 수 있으므로 원뿔까지의 거리. μ = 공·바닥 마찰 평균 (PhysX 기본 결합).
- 아니면 `|J|` 전체.
- `touch = 위 값 > KICK_DV_THR (0.5)`, `kick = 킥 전 & touch & |v_ball| >= KICK_SPEED_MIN (5)`.
- 킥 판정 후 `KICK_WINDOW_STEPS` (10) step 뒤 종료 (terminate). 약한 터치는 킥이 아니며 에피소드가 계속된다 (결정 1.B).

처음 제안했던 "Δv > thr 이고 step 끝에 공이 다리 근처" (D2) 는 쓰지 않는다. 30 Hz control step 에서 세게 찬 공은 step 이 끝날 때 이미 발에서 0.3–1 m 떨어져 있어 킥을 절반 이상 놓친다 (§3).

### 2.6 종료
- 넘어짐 (원본과 같은 식, `contactBodies` = 양발).
- 발 / 정강이 외 body 의 접촉력 norm > `BODY_CONTACT_THR` (1 N). 공인지 바닥인지 구분하지 않는다.
- 킥 창 종료, timeout (`episodeLength` 120, truncation).
- 로그 `episode_stats/end_{kick, fall, body_contact, timeout}`.

### 2.7 공
`ballProfile` (기본 `linear_damping_and_mjfriction`, Trap_Fix / Dribble_Fix 와 같음). 원본 값은 `physicsfc_legacy`.

## 3. 킥 판정 측정

스크립트: `kick_detect_measure.py` (scratchpad). 정답은 반사실 비교로 만든다.
- env 마다 바닥하고만 충돌하는 유령 공 (같은 asset, collision filter 0b11, mj friction 도 적용) 을 두고, 매 control step 직전에 진짜 공 상태를 복사한다.
- step 뒤 `gt = |v_진짜 - v_유령|` = 캐릭터가 공에 준 속도 변화.
- PhysX 접촉 목록 API (`get_rigid_contacts`) 는 이 Isaac Gym 빌드에서 GPU / CPU PhysX 모두 쓰레기 값을 돌려줘서 쓸 수 없었다.
- 조건: CPU pipeline + GPU PhysX, substeps 2, controlFrequencyInv 2 (학습과 같음), 1024 env × 450 step, 리셋 / 재발사 직후 2 step 제외.

| 시나리오 | 내용 |
|---|---|
| shoot | 기본 자세, 정강이 / 발의 한 점 (+ 0–0.4 m 빗나감) 을 향해 1.2–2.5 m 에서 2–20 m/s 로 발사, 스핀 ±30 rad/s |
| drop | 캐릭터 0.15–1 m 근처, 높이 0.3–2 m, 수평 0–4 m/s, 스핀 ±30 — 다리 옆 바닥 튐 |
| dribble | 드리블 상태 파일에서 시작, 랜덤 동작 (0.3·N(0,1)) |

결과 (D1 = 구현, D2 = 처음 제안):

| | shoot | drop | dribble |
|---|---|---|---|
| 접촉 없는 step 의 D1 값 최대 | 0.070 | 0.057 | 0.046 |
| D1 (0.5) 접촉 없는데 검출 | 0 | 0 | 0 |
| D1 (0.5) recall, gt > 1 m/s | 0.974 | 0.938 | 0.971 |
| D1 (0.5) 킥 recall (gt > 1 & 속력 ≥ 5) | 0.992 | 1.000 (3 개) | 0.991 |
| D1 (1.0) recall, gt > 1 m/s | 0.964 | 0.919 | 0.601 |
| D2 (1.0) precision / recall | 0.89 / 0.47 | 0.62 / 0.20 | 0.96 / 0.33 |

- 임계값 1.0 은 드리블의 약한 밀기 (gt 1–2 m/s) 를 많이 놓친다. 원뿔 거리는 실제 충격량의 하한이라 (바닥이 줄 수 있는 만큼은 빼므로) 바닥에 붙은 공을 수평으로 미는 경우 실제보다 작게 나온다. 0.5 로 내려도 접촉 없는 step 의 최대값 (0.07) 과 7 배 차이라 0.5 로 정했다.
- 남는 미검출은 바닥 근처에서 캐릭터 충격량이 원뿔 안에 들어가는 경우 (공을 위로 떠받치거나 발이 바닥처럼 공을 받치는 경우). 킥 기준으로는 약 1 %. 놓친 킥은 timeout 까지 에피소드가 이어진다.
- 고속 바운스 (fast 시나리오: 바닥 도달 속력 10–35 m/s, 입사각 3–85°, 스핀 최대 ±100 rad/s, 캐릭터 0.3–1.5 m 근처, 0 동작) 에서도 접촉 없는 바운스 오검출은 0. 판정값 최대는 착지 직전 속력 0–10 / 10–20 / 20–30 / 30+ m/s 에서 0.024 / 0.049 / 0.074 / 0.084.
  이 시나리오의 터치 recall 은 0.76 으로 낮은데, 놓친 경우는 전부 빠르게 떨어지는 공 (직전 v_z 중앙값 -18 m/s) 이 위를 향한 몸 표면에 떨어져 튄 것 (J_xy/J_z 중앙값 0.51). 절반 이상은 넘어져 누운 캐릭터 (root z 0.23) 위다. 몸이 바닥처럼 공을 받친 경우라 바운스와 구별되지 않는다. 학습에서는 넘어지면 종료되고, 서 있을 때는 발등 / 정강이 위로 공이 떨어지는 경우라 킥 동작과는 거리가 있다.
- B 판정은 이미 빠른 공 (≥ 5 m/s) 을 살짝 건드려도 킥으로 본다 (shoot 에서 gt < 0.5 인 킥 1 건). 드리블 파일의 공은 90 % 가 4.1 m/s 이하.

## 4. 원본 대비 바뀌는 것

| 항목 | BallPos_1 | Kick_Fix |
|---|---|---|
| 초기 상태 | 기본 배치 (IS_INIT 상수, 경로 하드코딩) | 파일 0.8 / 기본 0.2, yaml |
| 목표 방향 | root 기준 50 m 점, 다른 env 리셋 때 재계산 | 공 기준 방향을 직접 샘플, 에피소드 동안 고정 |
| 고각 | atan(U(0,50)/~50) | U(0°, 45°) |
| 목표 속력 | 15–30 | 8–30 |
| 관측 | 10 | 13 (+ 공 각속도) |
| 킥 판정 | 공 속력 > 0.5 | 바닥 외 충격량 > 0.5 & 속력 ≥ 5 |
| 보상 시점 | 처음부터 매 step | 킥 판정 step 부터 |
| 땅볼 목표 | 없음 (고각 0 근처는 atan 분포로 드묾) | 40 %, 수평 보상 x 높이 계수 |
| 실패 | 넘어짐, (공 접촉 & 발 외 접촉) — 공 접촉은 사실 body 1 | 넘어짐, 발 / 정강이 외 접촉 > 1 N |
| 공 | 하드코딩 | `ballProfile` |
