# soccer_hlc HLC 테스트 — 실제 배포된 4개 스킬(move/dribble/kick/trap) 체크포인트를
# calm/calm_exp/checkpoints/<skill>/default.pth에서 직접 로드해서 검증. plan.md §1.1/§8
# 참고 (매핑 근거 및 검증 기록 — 4개 스킬 전부 이 커맨드들로 실행 검증 완료, 2026-08-05).

# move
# python calm/run.py \
# --test \
# --num_envs 2 \
# --task HumanoidMove \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_move.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --checkpoint calm/calm_exp/checkpoints/move/default.pth \
# --llc_steps 1

# dribble
python calm/run.py \
--test \
--num_envs 2 \
--task HumanoidDribble \
--cfg_env calm/data/cfg/soccer_hlc/humanoid_dribble.yaml \
--cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
--motion_file blackhole/motions/AA_Soccer_Player_FootStop_MR_calm.npy \
--llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
--checkpoint calm/calm_exp/checkpoints/dribble/default.pth \
--llc_steps 1

# kick
# python calm/run.py \
# --test \
# --num_envs 2 \
# --task HumanoidBallPos_1 \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_ball_pos_task_1.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FreeKick01_RF_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --checkpoint calm/calm_exp/checkpoints/kick/default.pth \
# --llc_steps 1 \
# --episode_length 80

# trap
python calm/run.py \
--test \
--num_envs 2 \
--task HumanoidTrapping_3 \
--cfg_env calm/data/cfg/soccer_hlc/humanoid_trapping_3.yaml \
--cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
--motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
--llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
--checkpoint calm/calm_exp/checkpoints/trap/default.pth \
--llc_steps 1 \
--episode_length 60
