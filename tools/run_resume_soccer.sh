# soccer train

python calm/run.py \
--task HumanoidAMPGetup \
--num_envs 16 \
--cfg_env calm/data/cfg/humanoid_calm.yaml \
--cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml \
--motion_file /workspace/CALM/blackhole/motions/soccer_9.yaml \
--headless \
--resume 1 \
--checkpoint /workspace/CALM/output/unity_soccer_good_200000/nn/Humanoid.pth \
--max_iterations 400000 \
--experiment unity_soccer_good_400000