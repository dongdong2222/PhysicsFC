# soccer_hlc HLC 학습
# 각 스킬은 --llc_checkpoint로 frozen LLC를 로드하고 --cfg_train hrl_humanoid.yaml(base HRL)로
# 학습됨. plan.md §1.1/§8 참고 (매핑 근거 및 검증 기록).
#
# --motion_file 경로(blackhole/motions/...)는 원본 개발 컨테이너에만 있던 외부 모션
# 데이터라 이 저장소에는 없다. 실행하려면 해당 모션 데이터를 준비하거나 보유한 다른
# .npy/.yaml 모션 파일 경로로 바꿔야 한다.

# move
python calm/run.py \
--task HumanoidMove \
--cfg_env calm/data/cfg/soccer_hlc/humanoid_move.yaml \
--cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
--motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
--llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
--llc_steps 1 \
--experiment move \
--headless --track

# dribble
# python calm/run.py \
# --task HumanoidDribble \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_dribble.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --experiment dribble \
# --headless --track

# kick
# python calm/run.py \
# --task HumanoidBallPos_1 \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_ball_pos_task_1.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FreeKick01_RF_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --episode_length 80 \
# --experiment kick \
# --headless --track

# trap
# python calm/run.py \
# --task HumanoidTrapping_3 \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_trapping_3.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --episode_length 60 \
# --experiment trap \
# --headless --track
