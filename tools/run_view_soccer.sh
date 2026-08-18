# soccer 6 motion inference - this is just practice training result
# not so good result, should be compared with original training quality

python calm/run.py --test \
--task HumanoidViewMotion \
--num_envs 1 \
--cfg_env calm/data/cfg/humanoid_calm.yaml \
--cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml \
--motion_file blackhole/motions/AA_Soccer_Player_DTurnJogR_L_calm.npy
