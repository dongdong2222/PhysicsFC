# soccer 6 motion inference - this is just practice training result
# not so good result, should be compared with original training quality

python calm/run.py --test \
--task HumanoidViewMotion \
--num_envs 1 \
--cfg_env calm/data/cfg/humanoid_calm.yaml \
--cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml \
--motion_file blackhole/motions/AA_Soccer_Player_DTurnJogR_L_calm.npy

#llc test
python calm/run.py --test   --task HumanoidAMPGetup   --num_envs 16   --cfg_env calm/data/cfg/humanoid_calm.yaml   --cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml   --motion_file blackhole/motions/soccer_9.yaml   --checkpoint calm/calm_exp/checkpoints/llc/default.pth
# move joystic test
python calm/run.py --test --task HumanoidMove --num_envs 16 --cfg_env calm/data/cfg/soccer_hlc/humanoid_move.yaml --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy  --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth --checkpoint calm/calm_exp/checkpoints/move/default.pth
# dribble joystick test
python calm/run.py --test --task HumanoidDribble --num_envs 16 --cfg_env calm/data/cfg/soccer_hlc/humanoid_dribble.yaml --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy  --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth --checkpoint calm/calm_exp/checkpoints/dribble/87.pth
# ball mastery test (조이스틱 없음, 학습과 같은 랜덤 목표)
python calm/run.py --test --task HumanoidBallMastery --num_envs 16 --cfg_env calm/data/cfg/soccer_hlc/humanoid_ball_mastery.yaml --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy  --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth --checkpoint output/ball_mastery/nn/ball_mastery.pth
# trap joystick test
python calm/run.py --test --task HumanoidTrapping_3 --num_envs 16 --cfg_env calm/data/cfg/soccer_hlc/humanoid_trapping_3.yaml --cfg_train calm/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file blackhole/motions/AA_Soccer_Player_FootStop_ML_calm.npy  --llc_checkpoint calm/calm_exp/checkpoints/llc/default.pth --checkpoint calm/calm_exp/checkpoints/trap/default.pth
# kick joystick test




