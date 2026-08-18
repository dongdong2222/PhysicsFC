import numpy as np
import torch
import yaml
import pandas as pd
import plotly.graph_objects as go

from torch.nn.functional import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from addict import Dict
from torch_cluster import knn

import time

MULTIPLE = True
w_key_body = 1.
w_root_rot = 1.
w_root_vel = 1.
w_future_pos = 5
w_future_rot = 5
k = 5
envs = 512

class MotionDB:
    def __init__(self):
        self._n_motion = 90
        self._n_feature = 125
        self._max_motion_len = 120
        self._feature_path = '../motion_features.npy'
        self._motion_names_path = '../blackhole/motions/soccer_90.yaml'
        self._motion_path = '../blackhole/motions/'
        self._latent_path = '../latents_90.npy'
        self._features = None
        self._features_extended = np.zeros(shape=(self._n_motion, self._max_motion_len, 19))
        self._features_extended_train = torch.zeros(size=(envs, self._n_motion, self._max_motion_len, 19), device='cuda')
        self._motions = None
        self._motion_names = None
        self._latents = None

        self.prepare_datas()

    def prepare_datas(self):
        # features
        # root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos
        # (1, 6 (tan+norm), 3, 3, 72 (12 body * 6 tan norm), 28 (12 bodies' eash dof), 12 (hand foot pos))

        # root_pos_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos
        # (3, 4 (quat), 3, 3, 72 (12 body * 6 tan norm), 28 (12 bodies' eash dof), 12 (hand foot pos))

        self._features = np.load(self._feature_path)
        self._features = self._features.reshape([self._n_motion, -1, self._n_feature])
        

        # motion names
        with open(self._motion_names_path, 'r') as f:
            self._motion_names = yaml.load(f, Loader=yaml.FullLoader)
            self._motion_names = [m['file'][:-4] for m in self._motion_names['motions']]

        # motions arr
        motion_data_path = Path(self._motion_path)
        self._motions = dict()
        for p in motion_data_path.iterdir():
            if 'calm.npy' in str(p):
                arr = np.load(p, allow_pickle=True)
                arr = Dict(arr.item())
                motion_path = str(p).split('/')[-1].split('.')[0]
                self._motions[motion_path] = arr

        # latents
        with open(self._latent_path, 'rb') as f:
            self._latents = np.load(f)
        
        self._frames = self._features.shape[1]


    def query(self, q_feature):
        q = np.tile(q_feature, [self._n_motion, self._max_motion_len]).reshape(self._n_motion, self._max_motion_len, -1)
        dists = np.linalg.norm(self._features_extended[..., :]-q[..., :], axis=-1)
        nearest_frame = np.argmin(dists)
        nearest_motion, nearest_frame = divmod(nearest_frame, self._max_motion_len)
        return nearest_motion, nearest_frame
    
    def query_train(self, q_feature, num_envs):
        q = torch.tile(q_feature, [self._n_motion, self._max_motion_len]).reshape(num_envs, self._n_motion, self._max_motion_len, -1)
        dists = torch.norm(self._features_extended_train[..., :] -q[..., :], dim=-1).reshape(num_envs, -1)
        nearest_frame = torch.argmin(dists, dim=1)
        nearest_motion = torch.div(nearest_frame, 90, rounding_mode='trunc')
        return nearest_motion

    def query_knn(self, q_feature):
        feature_db = torch.tensor(self._features_extended, device='cuda').reshape((-1, 19))
        assign_index = knn(feature_db, q_feature, 1)
        return assign_index[1][0] // 120
    
    def arange_feature(self):
        for motion_id in range(self._n_motion):
            self._features_extended[motion_id, :self._max_motion_len-1, :12] = self._features[motion_id, :self._max_motion_len-1, -12:]
            self._features_extended[motion_id, :self._max_motion_len-1, 12:15] = self._features[motion_id, :self._max_motion_len-1, 7:10]


    # - 런타임 혹은 임의로 작성한 채워넣은 feature 를 넣고 완성되어 있는 motion DB 에 쿼리하여 desired motion 과 그 motion 의 frame 을 반환
    def append_feature(self, n):
        ...
    # - 원본 feature 에 n 프레임 미래 정보 (변동 가능) 를 채워넣어 db 를 완성
    def export_feature(self):
        ...
    # - 완성된 feature (motion) db 를 array 로 저장
    def get_position(motion_idx, body):
        ...
    # - 특정 모션의 특정 body 의 pos 를 FK 를 통해 계산하여 반환
    def fill_future_root_pos(self, n):
        # feature has no root position, need motion data that including root pos in global frame
        for motion_idx in range(self._n_motion):
            motion_len = int(self._motions[self._motion_names[motion_idx]]['rotation']['arr'].shape[0] / 2)
            for frame in range(min(self._max_motion_len-6, motion_len-6)):
                rotation = torch.tensor(self._features[motion_idx, frame, 3:7]).unsqueeze(0)
                heading_rot = self.calc_heading_quat_inv(rotation.clone().detach())
                futures = [self._features[motion_idx, frame+i*5, :3] - self._features[motion_idx, frame, :3] for i in range(1, 2)]
                futures_local = np.array([self.quat_rotate(heading_rot, torch.tensor(f).unsqueeze(0)).numpy()[:, :2] for f in futures]).squeeze().ravel()
                self._features_extended[motion_idx, frame, 15:17] = futures_local
    # - 특정 motion 의 각 프레임마다 n 프레임 미래의 pos 를 해당 프레임의 캐릭터 프레임 기준으로 구해서 반환

    def fill_future_root_rot(self, n):
        # - 특정 motion 의 각 프레임마다 n 프레임 미래의 rot 을 해당 프레임의 캐릭터 프레임 기준으로 구해서 반환
        # rot = self._motions[self._motion_names[motion_idx]]['rotation']['arr']
        for motion_idx in range(self._n_motion):
            motion_len = int(self._motions[self._motion_names[motion_idx]]['rotation']['arr'].shape[0] / 2)
            for frame in range(min(self._max_motion_len-6, motion_len-6)):
                rotation = torch.tensor(self._features[motion_idx, frame, 3:7]).unsqueeze(0)
                heading_rot = self.calc_heading_quat_inv(rotation.clone().detach())
                futures = [self._features[motion_idx, frame+i*5, 3:7] for i in range(1, 2)]
                futures_local = np.array([self.calc_heading_xy(self.quat_mul(heading_rot, torch.tensor(f).unsqueeze(0))) for f in futures]).squeeze().ravel()
                self._features_extended[motion_idx, frame, 17:19] = futures_local

    def get_feature_shape(self):
        return self._features.shape

    ######################################################################################

    def quat_mul(self, a, b):
        assert a.shape == b.shape
        shape = a.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 4)

        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)

        quat = torch.stack([x, y, z, w], dim=-1).view(shape)

        return quat

    def normalize(self, x, eps: float = 1e-9):
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

    def quat_unit(self, a):
        return normalize(a)

    def quat_from_angle_axis(self, angle, axis):
        theta = (angle / 2).unsqueeze(-1)
        xyz = normalize(axis) * theta.sin()
        w = theta.cos()
        return self.quat_unit(torch.cat([xyz, w], dim=-1))

    def quat_rotate(self, q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a + b + c

    def calc_heading(self, q):
        # calculate heading direction from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        ref_dir = torch.zeros_like(q[..., 0:3])
        ref_dir[..., 0] = 1
        rot_dir = self.quat_rotate(q, ref_dir)

        heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
        return heading
    
    def calc_heading_xy(self, q):
        # calculate heading direction from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        ref_dir = torch.zeros_like(q[..., 0:3])
        ref_dir[..., 0] = 1
        rot_dir = self.quat_rotate(q, ref_dir)

        return rot_dir[..., 0], rot_dir[..., 1]

    def calc_heading_quat(self, q):
        # calculate heading rotation from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        heading = self.calc_heading(q)
        axis = torch.zeros_like(q[..., 0:3])
        axis[..., 2] = 1

        heading_q = self.quat_from_angle_axis(heading, axis)
        return heading_q

    def calc_heading_quat_inv(self, q):
        # calculate heading rotation from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        heading = self.calc_heading(q)
        axis = torch.zeros_like(q[..., 0:3])
        axis[..., 2] = 1

        heading_q = self.quat_from_angle_axis(-heading, axis)
        return heading_q
    
    def copysign(self, a, b):
        a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
        return torch.abs(a) * torch.sign(b)

    def slerp(self, q0, q1, t):
        cos_half_theta = torch.sum(q0 * q1, dim=-1)

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta);
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta);

        ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta;
        ratioB = torch.sin(t * half_theta) / sin_half_theta; 
        
        new_q = ratioA * q0 + ratioB * q1

        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

        return new_q

    def quat_from_angle_axis(self, angle, axis):
        theta = (angle / 2).unsqueeze(-1)
        xyz = self.normalize(axis) * theta.sin()
        w = theta.cos()
        return self.quat_unit(torch.cat([xyz, w], dim=-1))

    def get_euler_xyz(self, q):
        qx, qy, qz, qw = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
        cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
            q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)control
        sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
        pitch = torch.where(torch.abs(sinp) >= 1, self.copysign(
            np.pi / 2.0, sinp), torch.asin(sinp))

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
        cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
            q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


    def quat_from_euler_xyz(self, roll, pitch, yaw):
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp

        return torch.stack([qx, qy, qz, qw], dim=-1)

   
class MotionDB_2:
    def __init__(self, device='cuda'):
        self._feature_path = '../motion_features.npy'
        self._motion_names_path = '../blackhole/motions/soccer_90.yaml'
        self._motion_path = '../blackhole/motions/'
        self._latent_path = '../latents_90.npy'

        self._n_motion = 90
        self._n_feature = 125
        self._max_motion_len = 120
        self._offset = 3
        self._offset_rot = 3
        # self._feature_path = '/workspace/CALM/motion_features_locomotion_only_jump.npy'
        # self._motion_names_path = '/workspace/CALM/blackhole/motions/soccer_90_locomotion_only_jump.yaml'
        # self._motion_path = '/workspace/CALM/blackhole/motions/'
        # self._latent_path = '/workspace/CALM/feature_move_test_z.npy'
        self._features = None
        # self._features_extended_train = torch.zeros(size=(self._n_motion, self._max_motion_len, 21), device=device)
        self._motions = None
        self._motion_names = None
        self._latents = None
        self._mode = 'ORIGINAL_MATCHING'
        if self._mode == 'ORIGINAL_MATCHING':
            self._features_extended = torch.zeros(size=(self._n_motion, self._max_motion_len, 34), device=device)
        elif self._mode == 'SKILL_MATCHING':
            self._features_extended = None #torch.zeros(size=(len(self._features), 29), device=device)

        self.prepare_datas()
        self.arange_feature()
        self.fill_future_root_pos(3)
        self.fill_future_root_rot(3)

    def prepare_datas(self):
        if self._mode == 'ORIGINAL_MATCHING':
            # features
            # root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos
            # (1, 6 (tan+norm), 3, 3, 72 (12 body * 6 tan norm), 28 (12 bodies' eash dof), 12 (hand foot pos))

            # root_pos_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos
            # (3, 4 (quat), 3, 3, 72 (12 body * 6 tan norm), 28 (12 bodies' eash dof), 12 (hand foot pos))

            self._features = np.load(self._feature_path)
            self._features = self._features.reshape([self._n_motion, -1, self._n_feature])

            # motion names
            with open(self._motion_names_path, 'r') as f:
                self._motion_names = yaml.load(f, Loader=yaml.FullLoader)
                self._motion_names = [m['file'][:-4] for m in self._motion_names['motions']]

            # motions arr
            motion_data_path = Path(self._motion_path)
            self._motions = dict()
            for p in motion_data_path.iterdir():
                if 'calm.npy' in str(p):
                    arr = np.load(p, allow_pickle=True)
                    arr = Dict(arr.item())
                    motion_path = str(p).split('/')[-1].split('.')[0]
                    self._motions[motion_path] = arr

            # latents
            with open(self._latent_path, 'rb') as f:
                self._latents = np.load(f)
            
            self._frames = self._features.shape[1]
        
        elif self._mode == 'SKILL_MATCHING':
            self._features = np.load(self._feature_path)
            self._features = self._features.reshape([-1, self._n_feature])
            self._latents = np.load(self._latent_path)
            self._latents = self._latents.reshape([-1, 64])
            self._features_extended = torch.zeros(size=(self._features.shape[0], 29), device='cuda')

    def query(self, q_feature):
        if self._mode == 'ORIGINAL_MATCHING':
            q = torch.tile(q_feature, [self._n_motion, self._max_motion_len]).reshape(self._n_motion, self._max_motion_len, -1)
            q[..., :12] *= w_key_body
            q[..., 12:16] *= w_root_rot
            q[..., 16:19] *= w_root_vel
            # 
            q[..., 19:28] *= w_future_pos
            q[..., 29:34] *= w_future_rot
            dists = torch.norm(self._features_extended[..., :] -q[..., :], dim=-1)
            
            if MULTIPLE:
                nearest_vals, nearest_idxs = torch.min(dists, axis=1)
                nearest_motions = torch.argsort(nearest_vals)[:k]
                
                return nearest_motions, nearest_vals[nearest_motions]
            else:
                nearest_frame = np.argmin(dists)
                sim = np.min(dists)
                # if sim < 1.:
                #     print('similarity:', sim)
                nearest_motion, nearest_frame = divmod(nearest_frame, self._max_motion_len)
                # print(self._motion_names[nearest_motion])
                return nearest_motion, nearest_frame
        elif self._mode == 'SKILL_MATCHING':
            q = np.tile(q_feature, [len(self._features)]).reshape(len(self._features), -1)
            q[..., :12] *= .5
            q[..., 12:16] *= .2
            q[..., 16:19] *= .05
            dists = np.linalg.norm(self._features_extended[..., :].cpu() -q[..., :], axis=-1)
            nearest_frame = np.argmin(dists)
            sim = np.min(dists)
            print(sim)
            return None, nearest_frame


    ## under-construction!
    def query_train(self, q_feature, num_envs):
        q = torch.tile(q_feature, [self._n_motion, self._max_motion_len]).reshape(num_envs, self._n_motion, self._max_motion_len, -1)
        q[..., :12] *= w_key_body
        q[..., 12:16] *= w_root_rot
        q[..., 16:19] *= w_root_vel
        # 
        q[..., 19:28] *= w_future_pos
        q[..., 29:34] *= w_future_rot
        dists = torch.norm(self._features_extended[..., :] -q[..., :], dim=-1).reshape(num_envs, self._n_motion, -1)
        
        if MULTIPLE:
            nearest_vals = torch.min(dists, axis=2)
            nearest_motions = torch.argsort(nearest_vals[0], dim=1)[:, :k]
            
            return nearest_motions, nearest_vals[1].gather(1, nearest_motions)
        else:
            nearest_frame = np.argmin(dists)
            sim = np.min(dists)
            # if sim < 1.:
            #     print('similarity:', sim)
            nearest_motion, nearest_frame = divmod(nearest_frame, self._max_motion_len)
            # print(self._motion_names[nearest_motion])
            return nearest_motion, nearest_frame
        
    def query_train_old(self, q_feature, num_envs):
        q = torch.tile(q_feature, [self._n_motion, self._max_motion_len]).reshape(num_envs, self._n_motion, self._max_motion_len, -1)
        q[..., :12] *= .5
        q[..., 12:15] *= .1
        dists = torch.norm(self._features_extended[..., :] -q[..., :], dim=-1).reshape(num_envs, -1)
        mins, nearest_frame = torch.min(dists, dim=1)
        nearest_motion = torch.div(nearest_frame, self._max_motion_len, rounding_mode='trunc')
        # print(self._motion_names[nearest_motion[0]])
        return nearest_motion

    def arange_feature(self):
        if self._mode == 'ORIGINAL_MATCHING':
            for motion_id in range(self._n_motion):
                weights_1 = w_key_body
                weights_2 = w_root_rot
                weights_3 = w_root_vel
                self._features_extended[motion_id, :self._max_motion_len-1, :12] = torch.tensor(self._features[motion_id, :self._max_motion_len-1, -12:]) * weights_1
                self._features_extended[motion_id, :self._max_motion_len-1, 12:16] = torch.tensor(self._features[motion_id, :self._max_motion_len-1, 3:7]) * weights_2
                self._features_extended[motion_id, :self._max_motion_len-1, 16:19] = torch.tensor(self._features[motion_id, :self._max_motion_len-1, 7:10]) * weights_3
        elif self._mode == 'SKILL_MATCHING':
            weights_1 = .2
            weights_2 = .2
            weights_3 = .05
            self._features_extended[:, :12] = torch.tensor(self._features[:, -12:]) * weights_1
            self._features_extended[:, 12:16] = torch.tensor(self._features[:, 3:7]) * weights_2
            self._features_extended[:, 16:19] = torch.tensor(self._features[:, 7:10]) * weights_3
            
            
    def fill_train_db(self):
        self._features_extended_train = torch.tensor(self._features_extended, device='cuda')

    # - 런타임 혹은 임의로 작성한 채워넣은 feature 를 넣고 완성되어 있는 motion DB 에 쿼리하여 desired motion 과 그 motion 의 frame 을 반환
    def append_feature(self, n):
        ...
    # - 원본 feature 에 n 프레임 미래 정보 (변동 가능) 를 채워넣어 db 를 완성
    def export_feature(self):
        ...
    # - 완성된 feature (motion) db 를 array 로 저장
    def get_position(motion_idx, body):
        ...
    # - 특정 모션의 특정 body 의 pos 를 FK 를 통해 계산하여 반환
    def fill_future_root_pos(self, n):
        # feature has no root position, need motion data that including root pos in global frame
        if self._mode == 'ORIGINAL_MATCHING':
            for motion_idx in range(self._n_motion):
                motion_len = int(self._motions[self._motion_names[motion_idx]]['rotation']['arr'].shape[0] / 2)
                for frame in range(min(self._max_motion_len-(self._offset+7), motion_len-(self._offset+7))):
                    rotation = torch.tensor(self._features[motion_idx, frame, 3:7]).unsqueeze(0)
                    heading_rot = self.calc_heading_quat_inv(rotation.clone().detach())
                    futures = [self._features[motion_idx, frame+i*self._offset, :3] - self._features[motion_idx, frame, :3] for i in range(1, 4)]
                    futures_local = torch.tensor([self.quat_rotate(heading_rot, torch.tensor(f).unsqueeze(0)).numpy()[:, :3] for f in futures]).squeeze().ravel()
                    self._features_extended[motion_idx, frame, 19:28] = futures_local * w_future_pos
        elif self._mode == 'SKILL_MATCHING':
            for frame in range(self._features.shape[0]-2*self._offset):
                rotation = torch.tensor(self._features[frame, 3:7]).unsqueeze(0)
                heading_rot = self.calc_heading_quat_inv(rotation.clone().detach())
                futures = [self._features[frame+i*self._offset, :3] - self._features[frame, :3] for i in range(1, 3)]
                futures_local = torch.tensor([self.quat_rotate(heading_rot, torch.tensor(f).unsqueeze(0)).numpy()[:, :3] for f in futures]).squeeze().ravel()
                self._features_extended[frame, 19:25] = futures_local


    # - 특정 motion 의 각 프레임마다 n 프레임 미래의 pos 를 해당 프레임의 캐릭터 프레임 기준으로 구해서 반환

    def fill_future_root_rot(self, n):
        # - 특정 motion 의 각 프레임마다 n 프레임 미래의 rot 을 해당 프레임의 캐릭터 프레임 기준으로 구해서 반환
        if self._mode == 'ORIGINAL_MATCHING':
            for motion_idx in range(self._n_motion):
                motion_len = int(self._motions[self._motion_names[motion_idx]]['rotation']['arr'].shape[0] / 2)
                for frame in range(min(self._max_motion_len-(self._offset_rot+7), motion_len-(self._offset_rot+7))):
                    rotation = torch.tensor(self._features[motion_idx, frame, 3:7]).unsqueeze(0)
                    heading_rot = self.calc_heading_quat_inv(rotation.clone().detach())
                    futures = [self._features[motion_idx, frame+i*self._offset_rot, 3:7] for i in range(1, 4)]
                    futures_local = torch.tensor([self.calc_heading_xy(self.quat_mul(heading_rot, torch.tensor(f).unsqueeze(0))) for f in futures]).squeeze().ravel()
                    self._features_extended[motion_idx, frame, 28:34] = futures_local * w_future_rot
        elif self._mode == 'SKILL_MATCHING':
            for frame in range(self._features.shape[0]-2*self._offset_rot):
                rotation = torch.tensor(self._features[frame, 3:7]).unsqueeze(0)
                heading_rot = self.calc_heading_quat_inv(rotation.clone().detach())
                futures = [self._features[frame+i*self._offset_rot, 3:7] for i in range(1, 3)]
                futures_local = torch.tensor([self.calc_heading_xy(self.quat_mul(heading_rot, torch.tensor(f).unsqueeze(0))) for f in futures]).squeeze().ravel()
                self._features_extended[frame, 25:31] = futures_local

    def fill_future_root_vel(self, n):
        # - 특정 motion 의 각 프레임마다 n 프레임 미래의 vel 을 해당 프레임의 캐릭터 프레임 기준으로 구해서 반환
        for motion_idx in range(self._n_motion):
            motion_len = int(self._motions[self._motion_names[motion_idx]]['rotation']['arr'].shape[0] / 2)
            for frame in range(min(self._max_motion_len-(self._offset+1), motion_len-(self._offset+1))):
                rotation = torch.tensor(self._features[motion_idx, frame, 3:7]).unsqueeze(0)
                heading_rot = self.calc_heading_quat_inv(rotation.clone().detach())
                futures = [self._features[motion_idx, frame+i*self._offset, 7:10] for i in range(1, 2)]
                futures_local = torch.tensor([self.quat_rotate(heading_rot, torch.tensor(f).unsqueeze(0)).numpy()[:, :2] for f in futures]).squeeze().ravel()
                self._features_extended[motion_idx, frame, 36:38] = futures_local

    def get_feature_shape(self):
        return self._features.shape
    
    def get_latent_code(self, frame):
        return torch.tensor(self._latents[frame], device='cuda')

    ######################################################################################

    def quat_mul(self, a, b):
        assert a.shape == b.shape
        shape = a.shape
        a = a.reshape(-1, 4)
        b = b.reshape(-1, 4)

        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)

        quat = torch.stack([x, y, z, w], dim=-1).view(shape)

        return quat

    def normalize(self, x, eps: float = 1e-9):
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

    def quat_unit(self, a):
        return normalize(a)

    def quat_from_angle_axis(self, angle, axis):
        theta = (angle / 2).unsqueeze(-1)
        xyz = normalize(axis) * theta.sin()
        w = theta.cos()
        return self.quat_unit(torch.cat([xyz, w], dim=-1))

    def quat_rotate(self, q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a + b + c

    def calc_heading(self, q):
        # calculate heading direction from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        ref_dir = torch.zeros_like(q[..., 0:3])
        ref_dir[..., 0] = 1
        rot_dir = self.quat_rotate(q, ref_dir)

        heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
        return heading
    
    def calc_heading_xy(self, q):
        # calculate heading direction from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        ref_dir = torch.zeros_like(q[..., 0:3])
        ref_dir[..., 0] = 1
        rot_dir = self.quat_rotate(q, ref_dir)

        return rot_dir[..., 0], rot_dir[..., 1]

    def calc_heading_quat(self, q):
        # calculate heading rotation from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        heading = self.calc_heading(q)
        axis = torch.zeros_like(q[..., 0:3])
        axis[..., 2] = 1

        heading_q = self.quat_from_angle_axis(heading, axis)
        return heading_q

    def calc_heading_quat_inv(self, q):
        # calculate heading rotation from quaternion
        # the heading is the direction on the xy plane
        # q must be normalized
        heading = self.calc_heading(q)
        axis = torch.zeros_like(q[..., 0:3])
        axis[..., 2] = 1

        heading_q = self.quat_from_angle_axis(-heading, axis)
        return heading_q
    
    def copysign(self, a, b):
        a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
        return torch.abs(a) * torch.sign(b)

    def slerp(self, q0, q1, t):
        cos_half_theta = torch.sum(q0 * q1, dim=-1)

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta);
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta);

        ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta;
        ratioB = torch.sin(t * half_theta) / sin_half_theta; 
        
        new_q = ratioA * q0 + ratioB * q1

        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

        return new_q

    def quat_from_angle_axis(self, angle, axis):
        theta = (angle / 2).unsqueeze(-1)
        xyz = self.normalize(axis) * theta.sin()
        w = theta.cos()
        return self.quat_unit(torch.cat([xyz, w], dim=-1))

    def get_euler_xyz(self, q):
        qx, qy, qz, qw = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
        cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
            q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
        pitch = torch.where(torch.abs(sinp) >= 1, self.copysign(
            np.pi / 2.0, sinp), torch.asin(sinp))

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
        cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
            q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


    def quat_from_euler_xyz(self, roll, pitch, yaw):
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)

        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp

        return torch.stack([qx, qy, qz, qw], dim=-1)


mdb = MotionDB()
mdb_2 = MotionDB_2()

mdb.arange_feature()
mdb.fill_future_root_pos(3)
mdb.fill_future_root_rot(3)

import timeit
from time import time

motion_idx = 47
frame_idx = 15
dt = 1 / 30
feature = torch.tensor(mdb._features_extended[motion_idx, frame_idx])
original_feature = mdb._features[motion_idx, frame_idx]

# feature manipulation
left_axis = torch.tensor((1, 0, 0), dtype=torch.float)
right_axis = torch.tensor((1, 0, 0), dtype=torch.float)

# only joystick control input affects future pos, rot
rotation = torch.tensor(original_feature[3:7]).unsqueeze(0)
heading_rot = mdb.calc_heading_quat_inv(rotation.clone())
vel_joystick = mdb.quat_rotate(heading_rot, left_axis.unsqueeze(0))
vel_sum = (torch.tensor(original_feature[7:10]) + vel_joystick[0]) / 2
print('speed', vel_sum)
future_pos = vel_sum * dt * 5
feature[15:17] = future_pos[:2]

rot = np.arctan2(right_axis[1], right_axis[0]) * 180 / np.pi
quat_joystick = mdb.quat_from_angle_axis((rot.clone() * np.pi / 180).unsqueeze(0), torch.tensor([0, 0, 1], dtype=torch.float).unsqueeze(0))
target_rot = mdb.quat_mul(heading_rot, quat_joystick)
slerp = mdb.slerp(heading_rot, target_rot, torch.tensor(1))
futures_local = mdb.calc_heading_xy(slerp)
feature[17] = futures_local[0]
feature[18] = futures_local[1]

# query
nearest_motion, nearest_frame = mdb.query(feature)

# query knn
# nearest_motion_knn = mdb.query_knn(feature)

# print('original', mdb._motion_names[motion_idx])
# print('query', mdb._motion_names[nearest_motion], nearest_frame)
# print('query knn', mdb._motion_names[nearest_motion_knn])

feature_stack = torch.tensor(feature.unsqueeze(0), device='cuda').reshape((-1, 19)).repeat([envs, 1])

print(timeit.timeit('mdb.query_knn(feature_stack)', globals=globals(), number=100))
print(timeit.timeit('mdb.query(feature)', globals=globals(), number=100))
print(timeit.timeit('mdb.query_train(feature_stack, envs)', globals=globals(), number=100))