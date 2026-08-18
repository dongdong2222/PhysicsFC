# soccer train

python calm/run.py \
--task HumanoidAMPGetup \
--num_envs 16 \
--cfg_env calm/data/cfg/humanoid_calm.yaml \
--cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml \
--motion_file blackhole/motions/soccer_9.yaml \
--headless
