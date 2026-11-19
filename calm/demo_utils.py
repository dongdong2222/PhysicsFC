
from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *
from gym import spaces

import torch
from utils import torch_utils
from torch import Tensor
from rl_games.algos_torch import torch_ext, players
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from typing import List, Tuple, Optional
import os
from learning import calm_models, calm_network_builder, hrl_models, hrl_network_builder
import yaml
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

def get_axis_params(value, axis_idx, dtype=np.float64, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    return list(params.astype(dtype))

#@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if not local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs

def compute_move_observations(root_states, target_dir, target_speed, target_face_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_rot = root_states[:, 3:7]

    target_dir3d = torch.cat([target_dir[..., 0:2], torch.zeros_like(target_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_target_dir = quat_rotate(heading_rot, target_dir3d)
    local_target_dir = local_target_dir[..., 0:2]
    target_speed = target_speed.unsqueeze(-1)
    
    target_face_dir3d = torch.cat([target_face_dir, torch.zeros_like(target_face_dir[..., 0:1])], dim=-1)
    local_target_face_dir = quat_rotate(heading_rot, target_face_dir3d)
    local_target_face_dir = local_target_face_dir[..., 0:2]

    obs = torch.cat([local_target_dir, target_speed, local_target_face_dir], dim=-1)
    obs = torch.nn.functional.pad(obs, (0, 18-5, 0, 0), mode='constant', value=0)
    return obs

def compute_dribble_observations(root_states, target_dir, target_speed, target_face_dir, ball_pos, ball_vel, ball_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    target_dir3d = torch.cat([target_dir[..., 0:2], torch.zeros_like(target_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_target_dir = quat_rotate(heading_rot, target_dir3d)
    local_target_dir = local_target_dir[..., 0:2]
    target_speed = target_speed.unsqueeze(-1)
    
    target_face_dir3d = torch.cat([target_face_dir, torch.zeros_like(target_face_dir[..., 0:1])], dim=-1)
    local_target_face_dir = quat_rotate(heading_rot, target_face_dir3d)
    local_target_face_dir = local_target_face_dir[..., 0:2]

    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    obs = torch.cat([local_target_dir, target_speed, local_target_face_dir, local_ball_pos, local_ball_vel, local_ball_ang_vel], dim=-1)
    obs = torch.nn.functional.pad(obs, (0, 18-14, 0, 0), mode='constant', value=0)
    return obs

def compute_kick_observations(root_states, ball_states, target_speed, target_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

    # target state
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    target_pos = ball_states[:, 0:3]
    target_vel = ball_states[:, 7:10]
    target_speed = target_speed.unsqueeze(-1)

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_target_pos = target_pos - root_pos
    local_target_pos[..., -1] = target_pos[..., -1]
    local_target_pos = quat_rotate(heading_rot, local_target_pos)
    local_target_vel = quat_rotate(heading_rot, target_vel)

    # target ball direc
    # target_dir3d = torch.cat([target_dir, torch.zeros_like(target_dir[..., 0:1])], dim=-1)
    target_dir3d = target_dir
    local_target_dir = quat_rotate(heading_rot, target_dir3d)
    local_target_dir = local_target_dir[..., 0:3]

    obs = torch.cat([local_target_pos, local_target_vel, target_speed, local_target_dir], dim=-1)
    obs = torch.nn.functional.pad(obs, (0, 18-10, 0, 0), mode='constant', value=0)
    return obs

def compute_heading_observations_simple_target_fall_onehot(root_states, ball_pos, ball_vel, ball_ang_vel, target_fall_pos, onehot_body_part):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    # fall pos
    target_fall_pos_tmp = torch.zeros_like(root_pos, device='cuda')
    target_fall_pos_tmp[..., :3] = target_fall_pos
    local_fall_pos = target_fall_pos_tmp - root_pos
    local_fall_pos = quat_rotate(heading_rot, local_fall_pos)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, local_fall_pos, onehot_body_part], dim=-1)
    return obs

def compute_heading_observations_simple_onehot(root_states, ball_pos, ball_vel, ball_ang_vel, onehot_body_part):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    # ball - character pos
    # ball vel (공의 속도만 고려? 아니면 공과 캐릭터 간 상대 속도도 고려?)
    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)

    # ball ang vel
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    obs = torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, onehot_body_part], dim=-1)
    obs = torch.nn.functional.pad(obs, (0, 18-15, 0, 0), mode='constant', value=0)
    return obs

# Fix 스킬 (HumanoidDribble_Fix / HumanoidKick_Fix / HumanoidTrap_Fix) 관측. 학습 태스크의 _compute_task_obs 와 같은 순서, 패딩 없음
def compute_dribble_fix_observations(root_states, target_dir, target_speed, ball_pos, ball_vel, ball_ang_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # 원본 dribble 에서 target_face_dir 제외 (14 -> 12)
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    target_dir3d = torch.cat([target_dir[..., 0:2], torch.zeros_like(target_dir[..., 0:1])], dim=-1)
    local_target_dir = quat_rotate(heading_rot, target_dir3d)[..., 0:2]

    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    return torch.cat([local_target_dir, target_speed.unsqueeze(-1), local_ball_pos, local_ball_vel, local_ball_ang_vel], dim=-1)

def compute_kick_fix_observations(root_states, ball_states, target_speed, target_dir):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    # 원본 kick + 공 각속도 (10 -> 13), 순서: 공 위치, 공 속도, 공 각속도, 목표 속력, 목표 방향
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_ball_pos = ball_states[:, 0:3] - root_pos
    local_ball_pos[..., -1] = ball_states[:, 2]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_states[:, 7:10])
    local_ball_ang_vel = quat_rotate(heading_rot, ball_states[:, 10:13])
    local_target_dir = quat_rotate(heading_rot, target_dir)
    return torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, target_speed.unsqueeze(-1), local_target_dir], dim=-1)

def compute_trap_fix_observations(root_states, ball_pos, ball_vel, ball_ang_vel, onehot_body_part):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # 원본 simple_onehot, one-hot 6 -> 8 (허벅지 2 개는 0)
    root_pos = root_states[:, :3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_ball_pos = ball_pos - root_pos
    local_ball_pos[..., -1] = ball_pos[..., -1]
    local_ball_pos = quat_rotate(heading_rot, local_ball_pos)
    local_ball_vel = quat_rotate(heading_rot, ball_vel)
    local_ball_ang_vel = quat_rotate(heading_rot, ball_ang_vel)

    onehot_body_part = torch.nn.functional.pad(onehot_body_part, (0, 8-onehot_body_part.shape[-1], 0, 0), mode='constant', value=0)
    return torch.cat([local_ball_pos, local_ball_vel, local_ball_ang_vel, onehot_body_part], dim=-1)

def compute_post_trap_target_observations(root_states, target_dir, target_speed):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # HumanoidTrap_Fix POST_TARGET_VEL: 트래핑 후 목표 공 방향 (heading 기준 xy) + 목표 속력
    heading_rot = torch_utils.calc_heading_quat_inv(root_states[:, 3:7])
    target_dir3d = torch.cat([target_dir[..., 0:2], torch.zeros_like(target_dir[..., 0:1])], dim=-1)
    local_target_dir = quat_rotate(heading_rot, target_dir3d)[..., 0:2]
    return torch.cat([local_target_dir, target_speed.unsqueeze(-1)], dim=-1)



class DoubleKeyDict(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            # idx 값으로 검색
            for k, v in self.items():
                if isinstance(k, tuple) and k[0] == key:
                    return v
        elif isinstance(key, str):
            # key 값으로 검색
            for k, v in self.items():
                if isinstance(k, tuple) and k[1] == key:
                    return v
        return super().__getitem__(key)

class FootballSkillMapper():
    skill_list = ["move", "dribble", "kick", "trap"]
    @staticmethod
    def skill_id(key:str):
        return FootballSkillMapper.skill_list.index(key)
    @property
    def enum_skill(self):
        return enumerate(self.skill_list)
    
class FootballFSM(FootballSkillMapper):
    @property
    def state(self):
        return self._state.clone()

    def __init__(self, num_character, device="cuda:0"):
        self.device = device
        self._num_character = num_character
        self._state = torch.zeros(self._num_character, dtype=torch.int32, device=self.device)
        self._init_state = self._state.clone()

    def override(self, state):
        self._state[:] = state.clone()

    def update(
        self, 
        start_commands              :List[bool], 
        end_commands                :List[bool], 
        ball_character_distances    :torch.Tensor, 
        ball_character_collisions   :torch.Tensor, 
        is_approaching              :torch.Tensor,
        is_closest                  :Optional[torch.Tensor] = None
        ):
        '''
        On Start Commands: 
            Move: convert state to "Trap"
            Dribble: convert state to "Kick"
        On End Commands:
            Trap: revert state to "Move"
            Kick: revert state to "Dribble"
        '''
        prev_state = self._state.clone()
        tensor_start = to_torch(start_commands, dtype=torch.bool, device=self.device)
        tensor_end = to_torch(end_commands, dtype=torch.bool, device=self.device)

        
        # Move -> Dribble: when ball_character_distance is less than 1m and is approaching
        if is_closest is not None:
            self._state = torch.where(
                torch.logical_and(
                    torch.logical_and(
                        torch.logical_and(
                            prev_state == self.skill_id("move"),
                            torch.less_equal(ball_character_distances, 2.0)
                        ),
                        is_approaching
                    ),
                    is_closest
                ),
                self.skill_id("dribble") * torch.ones_like(self._state),
                self._state
            )
        else:
            self._state = torch.where(
                torch.logical_and(
                    torch.logical_and(
                        prev_state == self.skill_id("move"),
                        torch.less_equal(ball_character_distances, 2.0)
                    ),
                    is_approaching
                ),
                self.skill_id("dribble") * torch.ones_like(self._state),
                self._state
            )
        # Move -> Trap: Start Command & ball is approaching
        self._state = torch.where(
            torch.logical_and(
                prev_state == self.skill_id("move"),
                torch.logical_and(
                    tensor_start == True,
                    is_approaching
                )
            ),
            self.skill_id("trap") * torch.ones_like(self._state),
            self._state
        )
        # Dribble -> Kick
        self._state = torch.where(
            torch.logical_and(
                prev_state == self.skill_id("dribble"),
                tensor_start == True
            ),
            self.skill_id("kick") * torch.ones_like(self._state),
            self._state
        )
        # Dribble -> Move: when ball_character_distance is greater than 2m
        if is_closest is not None:
            self._state = torch.where(
                torch.logical_and(
                    prev_state == self.skill_id("dribble"),
                    torch.logical_or(
                        torch.greater(ball_character_distances, 2.0),
                        torch.logical_not(is_closest)
                    )
                ),
                self.skill_id("move") * torch.ones_like(self._state),
                self._state
            )
        else:
            self._state = torch.where(
                torch.logical_and(
                    prev_state == self.skill_id("dribble"),
                    torch.greater(ball_character_distances, 2.0)
                ),
                self.skill_id("move") * torch.ones_like(self._state),
                self._state
            )
        # Trap -> Move: End Command | not approaching
        self._state = torch.where(
            torch.logical_and(
                prev_state == self.skill_id("trap"),
                torch.logical_or(
                    tensor_end == True,
                    torch.logical_not(is_approaching)
                )
            ),
            self.skill_id("move") * torch.ones_like(self._state),
            self._state
        )
        # Trap -> Dribble: when humanoid is in contact with the ball
        self._state = torch.where(
            torch.logical_and(
                prev_state == self.skill_id("trap"),
                ball_character_collisions
            ),
            self.skill_id("dribble") * torch.ones_like(self._state),
            self._state
        )
        # Kick -> Dribble
        self._state = torch.where(
            torch.logical_and(
                prev_state == self.skill_id("kick"),
                tensor_end == True
            ),
            self.skill_id("dribble") * torch.ones_like(self._state),
            self._state
        )
        # Kick -> Move: when humanoid is in contact with the ball or ball_character_distance is greater than 2m
        self._state = torch.where(
            torch.logical_and(
                prev_state == self.skill_id("kick"),
                torch.logical_or(
                    ball_character_collisions,
                    torch.greater(ball_character_distances, 2.0)
                )
            ),
            self.skill_id("move") * torch.ones_like(self._state),
            self._state
        )
        # from kick or dribble to move, if there was no colision, assume kick failed.
        kick_failed = torch.logical_and(
            torch.logical_and(
                torch.logical_or(
                    prev_state == self.skill_id("kick"),
                    prev_state == self.skill_id("dribble")
                ),
                self._state == self.skill_id("move"),
            ),
            torch.logical_not(ball_character_collisions)
        )
        
        kick_success = torch.logical_and(
            torch.logical_and(
                prev_state == self.skill_id("kick"),
                self._state == self.skill_id("move")
            ),
            ball_character_collisions
        )

        return kick_failed, kick_success

    def reset_state(self, my_idx=None, target_state:Optional[str]=None):
        self._state[:] = self._init_state.clone()
        if my_idx is not None:
            self._state[my_idx] = self.skill_id(target_state.lower())

class ControllerNetwork(FootballSkillMapper):
    def __init__(self, batch_size, device="cuda:0", checkpoint_path_dict={}):
        # Init Class Variables. NOTE: List all first!
        self.hlc_model_dict = DoubleKeyDict()
        self.hlc_running_mean_std_dict = DoubleKeyDict()
        self.llc_model = None
        self.llc_running_mean_std = None
        self.hlc_num_obs_dict = DoubleKeyDict()
        self.num_humanoid_obs = None
        self.llc_actions_low = None
        self.llc_actions_high = None
        self.latent_size = None
        self.hlc_rnn_state = None
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_path_dict:dict = {
            "llc" : "default.pth"
        }
        self.checkpoint_path_dict.update({
            skill_key: "87.pth" for _, skill_key in self.enum_skill
        })
        self.checkpoint_path_dict.update(checkpoint_path_dict)

        self._init_variables()
    def _init_variables(self):
        # Model Common Configs
        self.num_humanoid_obs = 223
        num_humanoid_act = 28

        # Latent Z Config
        self.latent_size = 64
        latent_z_shape = (self.latent_size,)

        # HLC Configs
        hlc_checkpoint_path_dict = DoubleKeyDict({
            (skill_idx, skill_key): os.path.join(PROJECT_PATH, "calm_exp", "checkpoints", skill_key, self.checkpoint_path_dict[skill_key])
                for skill_idx, skill_key in self.enum_skill
        })

        # LLC Configs
        llc_checkpoint_path = os.path.join("calm/calm_exp/checkpoints/llc", self.checkpoint_path_dict["llc"])
        llc_input_shape = (self.num_humanoid_obs,)
        llc_input_shape = torch_ext.shape_whc_to_cwh(llc_input_shape)
        llc_act_space = spaces.Box(np.ones(num_humanoid_act) * -1., np.ones(num_humanoid_act) * 1.)
        self.llc_actions_low = torch.from_numpy(llc_act_space.low.copy()).float().to(self.device)
        self.llc_actions_high = torch.from_numpy(llc_act_space.high.copy()).float().to(self.device)
        llc_model_config = {
            'actions_num': num_humanoid_act, 
            'input_shape': llc_input_shape, 
            'num_seqs': 1, 
            'amp_input_shape': (1250,), 
            'amp_obs_steps': 10, 
            'calm_latent_shape': latent_z_shape
            }

        # Build & load trained llc model
        with open(os.path.join(os.getcwd(), "calm/calm_exp/cfg/llc.yaml"), 'r') as f:
            llc_network_params = yaml.load(f, Loader=yaml.SafeLoader)
        llc_builder = calm_network_builder.CALMBuilder()
        llc_builder.load(llc_network_params)
        llc_model_builder = calm_models.ModelCALMContinuous(llc_builder)
        self.llc_model = llc_model_builder.build(llc_model_config)
        self.llc_model.to(self.device)
        self.llc_model.eval()
        llc_checkpoint = torch_ext.load_checkpoint(llc_checkpoint_path)
        self.llc_model.load_state_dict(llc_checkpoint['model'])
        self.llc_running_mean_std = RunningMeanStd(llc_input_shape).to(self.device)
        self.llc_running_mean_std.load_state_dict(llc_checkpoint['running_mean_std'])
        self.llc_running_mean_std.to(self.device)
        self.llc_running_mean_std.eval() 

        # Build & load trained hlc model
        hlc_checkpoint_dict = DoubleKeyDict({
            (skill_idx, skill_key): torch_ext.load_checkpoint(hlc_checkpoint_path_dict[skill_idx])
                for skill_idx, skill_key in self.enum_skill
        })
        self.hlc_num_obs_dict.update({
            (skill_idx, skill_key): hlc_checkpoint_dict[skill_idx]['model']['a2c_network.actor_mlp.0.weight'].shape[1]
                for skill_idx, skill_key in self.enum_skill
        })

        with open(os.path.join(os.getcwd(), "calm/calm_exp/cfg/hlc.yaml"), 'r') as f:
            hlc_network_params = yaml.load(f, Loader=yaml.SafeLoader)
        hlc_builder = hrl_network_builder.HRLBuilder()
        hlc_builder.load(hlc_network_params)
        hlc_model_builder = hrl_models.ModelHRLContinuous(hlc_builder)

        for skill_idx, skill_key in self.enum_skill:
            hlc_model_config = {
                'actions_num': self.latent_size,
                'input_shape': (self.hlc_num_obs_dict[skill_idx],),
                'num_seqs': 1,
            }
            hlc_model = hlc_model_builder.build(hlc_model_config)
            hlc_model.load_state_dict(hlc_checkpoint_dict[skill_idx]['model'])
            hlc_model.to(self.device)
            hlc_model.eval()
            self.hlc_model_dict.update({
                (skill_idx, skill_key): hlc_model
            })
            hlc_running_mean_std = RunningMeanStd((self.hlc_num_obs_dict[skill_idx],)).to(self.device)
            hlc_running_mean_std.load_state_dict(hlc_checkpoint_dict[skill_idx]['running_mean_std'])
            hlc_running_mean_std.to(self.device)
            hlc_running_mean_std.eval()
            self.hlc_running_mean_std_dict.update({
                (skill_idx, skill_key): hlc_running_mean_std
            })
    def _preproc_obs(self, obs_batch, running_mean_std=None):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        
        if running_mean_std is not None:
            obs_batch = running_mean_std(obs_batch)
        return obs_batch 
    def _get_hlc_actions(self, skill_key_or_id, obs, is_determenistic=False):
        hlc_obs = obs[..., :self.hlc_num_obs_dict[skill_key_or_id]]
        hlc_processed_obs = self._preproc_obs(hlc_obs, self.hlc_running_mean_std_dict[skill_key_or_id])
        hlc_input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : hlc_processed_obs,
            'rnn_states' : self.hlc_rnn_state,
        }
        with torch.no_grad():
            hlc_input_dict
            res_dict = self.hlc_model_dict[skill_key_or_id](hlc_input_dict)
        hlc_mu = res_dict['mus']
        hlc_action = res_dict['actions']
        self.hlc_rnn_state = res_dict['rnn_states']
        if is_determenistic:
            hlc_action = hlc_mu
        
        hlc_action = torch.squeeze(hlc_action.detach())
        hlc_clamped_action = torch.clamp(hlc_action, -1.0, 1.0).view(self.batch_size, -1)
        return hlc_clamped_action
    def _get_latent_z(self, obs, target_skill_ids, is_determenistic=False):
        latent_z = torch.zeros(size=(self.batch_size, self.latent_size), dtype=torch.float32, device=self.device)
        
        for skill_id, skill_key in self.enum_skill:
            
            
            latent_z[target_skill_ids == skill_id] = self._get_hlc_actions(skill_id, obs, is_determenistic)[target_skill_ids == skill_id]
            
        latent_z = torch.nn.functional.normalize(latent_z, dim=-1)
        return latent_z
    def _get_llc_actions(self, obs, latent_z):
        llc_obs = obs[..., :self.num_humanoid_obs]
        llc_processed_obs = self._preproc_obs(llc_obs, self.llc_running_mean_std)
        llc_mu, _ = self.llc_model.a2c_network.eval_actor(llc_processed_obs, latent_z)
        llc_action = players.rescale_actions(self.llc_actions_low, self.llc_actions_high, torch.clamp(llc_mu, -1.0, 1.0))
        return llc_action   
    def get_action(self, obs, target_skill_ids):
        obs = torch.clamp(obs, -np.inf, np.inf).to(dtype=torch.float32)
        latent_z = self._get_latent_z(obs, target_skill_ids)
        actions = torch.clamp(self._get_llc_actions(obs, latent_z), -1.0, 1.0)
        return actions

    def get_action_and_latent(self, obs, target_skill_ids):
        obs = torch.clamp(obs, -np.inf, np.inf).to(dtype=torch.float32)
        latent_z = self._get_latent_z(obs, target_skill_ids)
        actions = torch.clamp(self._get_llc_actions(obs, latent_z), -1.0, 1.0)
        return actions, latent_z