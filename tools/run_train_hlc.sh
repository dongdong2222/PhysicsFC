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

# move style (원본 PhysicsFC_project/calm에서 이식. 상황별 지정 모션과 latent의 cosine 유사도를 style reward로 사용)
# python calm/run.py \
# --task HumanoidMoveStyle \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_move.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid_style_control.yaml \
# --motion_file blackhole/motions/soccer_move_style.yaml \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/poseidon_90_100000.pth \
# --experiment move_style \
# --headless --track

# dribble
# python calm/run.py \
# --task HumanoidDribble \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_dribble.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --experiment dribble \
# --headless --track --wandb_project_name hlp-dribble

# dribble face (목표 방향을 보며 드리블)
# python calm/run.py \
# --task HumanoidDribbleFace \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_dribble_face.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --experiment dribble_face \
# --headless --track --wandb_project_name hlp-dribble-face

# ball mastery (원터치)
# python calm/run.py \
# --task HumanoidBallMastery \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --experiment ball_mastery \
# --headless --track --wandb_project_name hlp-ball-mastery

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

# kick fix (kick_fix_plan.md)
# python calm/run.py \
# --task HumanoidKick_Fix \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_kick_fix.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FreeKick01_RF_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --experiment kick_fix \
# --headless --track --wandb_project_name hlp-kick-fix

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

# trap fix (trap_fix_plan.md, episodeLength 는 태스크가 계산)
# python calm/run.py \
# --task HumanoidTrap_Fix \
# --cfg_env calm/data/cfg/soccer_hlc/humanoid_trap_fix.yaml \
# --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml \
# --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy \
# --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth \
# --llc_steps 1 \
# --experiment trap_fix \
# --headless --track --wandb_project_name hlp-trap-fix
