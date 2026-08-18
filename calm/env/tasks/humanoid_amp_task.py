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

import torch
import numpy as np

import env.tasks.humanoid_amp as humanoid_amp
from isaacgym.torch_utils import *
from utils import torch_utils

class HumanoidAMPTask(humanoid_amp.HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        ################EVALUATION#################################### by Eunho Jung   
        self._is_eval = cfg['args'].evaluate
        print(f"\n\n\n\t evaluation: {self._is_eval}\n\n\n")
        self._eval_testcase_dict = {}
        self._eval_current_result = {}
        self._eval_data = None
        self._eval_task = None
        
        self.cfg = cfg
        
        self.max_episode_length = None
        if self._is_eval:
            self._eval_init()
        ################EVALUATION################################### 
        
        super().__init__(cfg=self.cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if self._enable_task_obs:
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        if not self._is_eval:
            self._update_task()
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self._is_eval:
            self._eval_tgt_update()
        else:
            self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    ################EVALUATION Code Start##############
    def _eval_tgt_update(self):
        pass
        
    def _eval_init(self):
        from eval_util import get_test_cases, EvaluationData, get_default_summary_dir
        import os
        self._eval_task = self.cfg["evaluation"]["task"]
        self._eval_save_result = self.cfg["evaluation"].get("save_results", False)
        
        self._eval_batch_num = self.cfg["evaluation"].get("batch_num", 100)
        self._eval_batch_size = self.cfg["evaluation"].get("batch_size", 100)
        self._eval_change_steps = self.cfg["evaluation"].get("change_steps", 150) # as default, 5 sec per testcase
        
        self._eval_gain_criteria_frame = self.cfg["evaluation"].get("gain_criteria_frame", 30)
        self._eval_criteria_dict = {
            "angle": self.cfg["evaluation"].get("criteria_angle", 10),
            "speed": self.cfg["evaluation"].get("criteria_speed", 0.5),
            "distance": self.cfg["evaluation"].get("criteria_distance", 0.5)
        }
        
        # we only need batch size envs
        # do this before super().__init__ because num_envs is going to be used in it.
        self.cfg["env"]["numEnvs"] = self._eval_batch_num
        # after evaluation done, episode ends.
        self.cfg["env"]["episodeLength"] = self._eval_batch_size * self._eval_change_steps
        self.max_episode_length = self._eval_batch_size * self._eval_change_steps
        
        self._eval_testcase_dict = get_test_cases(
            batch_num=self._eval_batch_num, 
            batch_size=self._eval_batch_size, 
            save_path=os.path.join(get_default_summary_dir(), f"testcase_bn{self._eval_batch_num}_bs{self._eval_batch_size}.npz")
        )
        self._eval_data = EvaluationData()
        
        criteria_str = ""
        for key in self._eval_criteria_dict.keys():
            criteria_str += "_" + key + f"{self._eval_criteria_dict[key]}"
        self._eval_filename_only = f"eval_cs{self._eval_change_steps}_bn{self._eval_batch_num}_bs{self._eval_batch_size}{criteria_str}"
        self._eval_filename = os.path.join(get_default_summary_dir(), self._eval_filename_only)
        
        if os.path.exists(self._eval_filename+".npz") :
            print(f"\n\nloading other task results:{self._eval_filename+'.npz'}\n\n")
            self._eval_data.load_npz(path=self._eval_filename+".npz", override_task=self._eval_task)
        else:
            print(f"\n\nfile not exists:{self._eval_filename+'.npz'}\n\n")
        
        self._eval_reservation(self._eval_task)

    def _eval_reservation(self, task):
        # metric result tensor reservation
        for key in self._eval_data.task_metric_dict[task]:
            if key.startswith("Time to achieve") or key.endswith("_gain_time"):
                self._eval_current_result.update({
                    key: -1 * np.ones((self._eval_batch_num, self._eval_batch_size), dtype=float),
                })
            else:
                self._eval_current_result.update({
                    key: np.nan * np.ones((self._eval_batch_num, self._eval_batch_size, self._eval_change_steps), dtype=float),
                })
            print(f"test: {self._eval_current_result.keys}")
            print(f"key: {key}")
    
    def post_physics_step(self):
        if self._is_eval:
            if self._eval_save_result:
                self._eval_process()
            self._eval_tgt_update()
        
        
        super().post_physics_step()
        
        # print(f"step: {self.progress_buf[0].item()}")

        if self.progress_buf[0].item() >= self.max_episode_length and self._is_eval and self._eval_save_result:
            from eval_util import get_default_summary_dir
            import os
            self._eval_update_result_on_evaluationData()
            self._eval_data.refresh_summary()
            # print(self._eval_data.data)
            # print("\n\n\n\n")
            # print(self._eval_data.summary)
            print(f"saving...{self._eval_filename}")
            self._eval_data.save_csv(path=self._eval_filename+".csv")
            self._eval_data.save_npz(path=self._eval_filename+".npz")
            self._eval_data.save_graphs(path=os.path.join(get_default_summary_dir(), "graphs", self._eval_filename_only))
        
        
        
    def _eval_process(self): # process common evaluations. 
        current_frame = self.progress_buf[0].item() % self._eval_change_steps
        testcase_index = torch.div(self.progress_buf[0], self._eval_change_steps, rounding_mode='floor').item()
        
        root_state = self._humanoid_root_states.clone()
        root_pos = root_state[..., 0:3]
        root_rot = root_state[..., 3:7]
        root_linvel = root_state[..., 7:10]
        root_angvel = root_state[..., 10:13]
        
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        facing_dir = torch.zeros_like(root_pos)
        facing_dir[..., 0] = 1.0
        facing_dir = quat_rotate(heading_rot, facing_dir)
        
        char_move_dir_diff = torch.rad2deg(torch_utils.get_angle_between(
            normalize(root_linvel[..., 0:2]),
            normalize(self._tar_dir)
        )).cpu().numpy() # (num envs) angle(degree) differences
        
        self._eval_current_result.get("char_tgt_move_diff")[
            :, 
            testcase_index, 
            current_frame
        ] = char_move_dir_diff 

        # print(f'current testcase tar_dir: {torch.from_numpy(self._eval_testcase_dict["target_move_directions"]).to(device=self.device, dtype=torch.float)[:,testcase_index]}')
        print(f"current tgt_dir: {self._tar_dir[0]}")
        print(f"current tgt_diff: {char_move_dir_diff[0]}")
        print(f"current tgt_speed: {self._tar_speed[0]}")
              
        char_speed_diff = (self._tar_speed - torch.norm(root_linvel[..., 0:2], dim=-1)).cpu().numpy() #xy speed
        
        self._eval_current_result.get("char_tgt_speed_diff")[
            :, 
            testcase_index, 
            current_frame
        ] = char_speed_diff
        
        # print(current_frame, self._eval_gain_criteria_frame)
        
        if current_frame >= self._eval_gain_criteria_frame - 1:
            self._eval_check_and_update_gain_time(
                metric="char_tgt_move",
                testcase_index=testcase_index,
                current_frame=current_frame,
                criteria_type="angle"
            )
            self._eval_check_and_update_gain_time(
                metric="char_tgt_speed",
                testcase_index=testcase_index,
                current_frame=current_frame,
                criteria_type="speed"
            )
        
    
    def _eval_update_result_on_evaluationData(self):
        for key in self._eval_current_result.keys():
            # print(f"{self._eval_task}, {key}: {self._eval_current_result.get(key)}")
            self._eval_data.update_data(metric=key, new_data=self._eval_current_result.get(key).flatten(), task=self._eval_task)
    
    
    def _eval_check_and_update_gain_time(self, metric, testcase_index, current_frame, criteria_type="angle", by_success=True):
        gain_time_numpy : np.ndarray = self._eval_current_result[metric+"_gain_time"][:, testcase_index].copy() #(num envs)
        past_diff = self._eval_current_result[metric+"_diff"][
            :, 
            testcase_index, 
            current_frame - self._eval_gain_criteria_frame + 1:current_frame + 1
        ].copy() #기준값으로 봐야하는 프레임 수 만큼만 가져온다.
        
        criteria = self._eval_criteria_dict[criteria_type]
        
        self._eval_current_result[metric+"_gain_time"][:, testcase_index] = np.where(
            np.prod(
                np.where(
                    np.less_equal(
                        np.abs(past_diff), criteria * np.ones_like(past_diff)
                        ) if by_success else np.greater(
                            np.abs(past_diff), criteria * np.ones_like(past_diff)
                        ),
                    np.ones_like(past_diff, dtype=bool),
                    np.zeros_like(past_diff, dtype=bool)
                ),
                axis= -1
            ), # criteria frame동안 모두 기준 각도 이하(성공) 또는 이상(실패)이면 1, 이외에는 0인 (num_envs) 배열
            np.where(
                np.greater_equal(gain_time_numpy, np.zeros_like(gain_time_numpy)),
                gain_time_numpy,
                current_frame / 30.0
            ), #1이면 업데이트 - 초기값은 -1이므로 0보다 작으면 바로 업데이트, 0보다 크거나 같으면 기존 값 유지
            gain_time_numpy #이외에는 기존 값 그대로
        )
        
    # criteria들을 하나라도 만족하면 1, 아니면 0이 env 자리에 저장된 np.ndarry 반환
    def _eval_check_frame_conditions_or(
        self,
        testcase_index,
        current_frame,
        criteria_metrics,
        criteria_types,
        criteria_funcs
        ):
        results = []
        for idx, metric_key in enumerate(criteria_metrics):
            results.append(
                np.logical_not(
                    criteria_funcs[idx](
                        self._eval_current_result[metric_key][:,testcase_index,current_frame], 
                        self._eval_criteria_dict[criteria_types[idx]]
                    )
                )
            )
        
        return np.logical_not(np.prod(results, axis=0))
        
    ################EVALUATION Code End##############
    
    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if self._enable_task_obs:
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return
