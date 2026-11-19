# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
sys.path.append('/workspace/CALM/fbxbinding/build/Distrib/site-packages/fbx/')

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder

from learning import hrl_agent
from learning import hrl_players
from learning import hrl_models
from learning import hrl_network_builder
from learning import hrl_soccer_conditioned_agent

from learning import calm_agent
from learning import calm_players
from learning import calm_models
from learning import calm_network_builder

from env.tasks import humanoid_amp_task

import datetime

try:
    import wandb
except:
    wandb = None

args = None
cfg = None
cfg_train = None
run_name = None


def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class WandbEpochWriter:
    """Wraps the agent's TensorBoard writer: everything still goes to TensorBoard, and each epoch's scalars go
    to wandb in one wandb.log (step = epoch). common_agent writes 'performance/total_fps' first in every epoch's
    logging block, so that tag starts a new epoch; the last epoch is sent on close()."""
    EPOCH_START_TAG = 'performance/total_fps'

    def __init__(self, writer):
        self._writer = writer
        self._scalars = {}

    def add_scalar(self, tag, value, global_step=None, *args, **kwargs):
        if tag == self.EPOCH_START_TAG:
            self.log_epoch()
            self._scalars['frame'] = global_step
        self._scalars[tag] = float(value)
        return self._writer.add_scalar(tag, value, global_step, *args, **kwargs)

    def log_epoch(self):
        if 'info/epochs' in self._scalars:
            wandb.log(self._scalars, step=int(self._scalars['info/epochs']))
        self._scalars = {}

    def close(self):
        self.log_epoch()
        return self._writer.close()

    def __getattr__(self, name):
        return getattr(self._writer, name)


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        if hasattr(self, 'algo'):
            # HRLAgent builds its frozen LLC agent with this same observer; keep the training agent (and its writer)
            return
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        if wandb is not None and wandb.run is not None:
            self.algo.writer = WandbEpochWriter(self.algo.writer)
        self.writer = self.algo.writer
        self.reward_terms = {}       # name -> AverageMeter of finished episodes' sums
        self.reward_term_sums = {}   # name -> per-env sum over the running episode
        self.reward_term_step_sums = {}  # name -> sum over all envs and steps of the current epoch
        self.reward_term_step_count = 0
        self.episode_stats = {}      # name -> AverageMeter of the value at the step each episode ended
        return

    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
            # per-step reward terms from the task (extras["reward_terms"]); with llc_steps > 1 only the
            # last LLC step's terms arrive here
            ids = done_indices.view(-1)
            reward_terms = infos.get('reward_terms', {})
            for name, r in reward_terms.items():
                if name not in self.reward_term_sums:
                    self.reward_term_sums[name] = r.new_zeros(r.shape)
                    self.reward_terms[name] = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
                    self.reward_term_step_sums[name] = r.new_zeros(())
                sums = self.reward_term_sums[name]
                sums += r
                self.reward_terms[name].update(sums[ids].unsqueeze(-1).to(self.algo.ppo_device))
                sums[ids] = 0
                self.reward_term_step_sums[name] += r.sum()
            if reward_terms:
                self.reward_term_step_count += next(iter(reward_terms.values())).numel()
            # per-episode values from the task (extras["episode_stats"]), read at the step the episode ended
            for name, v in infos.get('episode_stats', {}).items():
                if name not in self.episode_stats:
                    self.episode_stats[name] = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
                if len(ids) > 0:
                    self.episode_stats[name].update(v[ids].unsqueeze(-1).to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        if self.reward_terms and all(m.current_size > 0 for m in self.reward_terms.values()):
            total = 0.0
            for name, meter in self.reward_terms.items():
                mean = meter.get_mean().item()
                total += mean
                self.writer.add_scalar(f'reward_terms/{name}', mean, frame)
            self.writer.add_scalar('reward_terms/total', total, frame)
        # per-step means over this epoch's rollout: change every epoch, even when no episode ended
        if self.reward_term_step_count > 0:
            total = 0.0
            for name, step_sum in self.reward_term_step_sums.items():
                mean = step_sum.item() / self.reward_term_step_count
                total += mean
                self.writer.add_scalar(f'reward_terms_step/{name}', mean, frame)
                step_sum.zero_()
            self.writer.add_scalar('reward_terms_step/total', total, frame)
            self.reward_term_step_count = 0
        for name, meter in self.episode_stats.items():
            if meter.current_size > 0:
                self.writer.add_scalar(f'episode_stats/{name}', meter.get_mean().item(), frame)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)
        # temp = {}
        # print(info.keys())
        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space
        info['enc_amp_observation_space'] = self.env.enc_amp_observation_space

        #
        info['track_observation_space'] = self.env.track_observation_space
        info['enc_track_observation_space'] = self.env.enc_track_observation_space


        if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
            info['task_obs_size'] = self.env.task.get_task_obs_size()
        else:
            info['task_obs_size'] = 0

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.algo_factory.register_builder('amp', lambda **kwargs: amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp', lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs: amp_network_builder.AMPBuilder())

    runner.algo_factory.register_builder('hrl', lambda **kwargs: hrl_agent.HRLAgent(**kwargs))
    runner.player_factory.register_builder('hrl', lambda **kwargs: hrl_players.HRLPlayer(**kwargs))
    runner.model_builder.model_factory.register_builder('hrl', lambda network, **kwargs: hrl_models.ModelHRLContinuous(network))
    runner.model_builder.network_factory.register_builder('hrl', lambda **kwargs: hrl_network_builder.HRLBuilder())

    # HumanoidMoveStyle: HRL + style reward (cosine similarity to the requested motion's LLC encoding); test uses the plain HRL player
    runner.algo_factory.register_builder('hrl_soccer_conditioned', lambda **kwargs: hrl_soccer_conditioned_agent.HRLSoccerConditionedAgent(**kwargs))
    runner.player_factory.register_builder('hrl_soccer_conditioned', lambda **kwargs: hrl_players.HRLPlayer(**kwargs))

    runner.algo_factory.register_builder('calm', lambda **kwargs: calm_agent.CALMAgent(**kwargs))
    runner.player_factory.register_builder('calm', lambda **kwargs: calm_players.CALMPlayer(**kwargs))
    runner.model_builder.model_factory.register_builder('calm', lambda network, **kwargs: calm_models.ModelCALMContinuous(network))
    runner.model_builder.network_factory.register_builder('calm', lambda **kwargs: calm_network_builder.CALMBuilder())

    return runner



def main():
    global args
    global cfg
    global cfg_train
    global run_name

    set_np_formatting()
    #dong : args dict init
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.wandb_run_name}_{time_str}"
    if args.experiment != 'Base':
        # --experiment names the output folder and the wandb run as is (no timestamp: a rerun overwrites the folder)
        run_name = cfg_train['params']['config']['name']
        cfg_train['params']['config']['full_experiment_name'] = run_name
    assert args.track and wandb or not args.track, "Tracking requires wandb to be installed."

    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size

    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file

    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path

    if args.track:
        # scalars reach wandb once per epoch through WandbEpochWriter, not through TensorBoard sync
        wandb.init(
            project=args.wandb_project_name,
            config=args,
            save_code=True,
            name=run_name,
        )

    vargs = vars(args)

    algo_observer = RLGPUAlgoObserver()

    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)

    return


if __name__ == '__main__':
    main()
