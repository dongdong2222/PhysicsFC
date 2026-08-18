import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle


class SingleDistilModel(torch.nn.Module):
    def __init__(self, input_size, output_size, rnn=False):
        super().__init__()
        self.rnn = rnn

        hidden = 512
        self.mlp = torch.nn.Linear(input_size, hidden, device='cuda')
        self.mlp_2 = torch.nn.Linear(input_size, hidden, device='cuda')
        self.mlp_mu = torch.nn.Linear(hidden, output_size, device='cuda')
        self.mlp_sigma = torch.nn.Linear(hidden, output_size, device='cuda')
        self.mlp_critic = torch.nn.Linear(hidden, 1, device='cuda')
        self.fixed_sigma = torch.nn.Parameter(data=torch.ones(size=(output_size,), device='cuda'), requires_grad=False) * .1

    def forward(self, input_dict):
        is_train = False #input_dict.get('is_train', True)
        obs = input_dict['obs']
        # obs = obs.permute((0, 3, 1, 2))
        x = obs

        mu = self.mlp(x)
        mu = torch.nn.LeakyReLU()(mu)
        mu = self.mlp_mu(mu)
        mu = torch.tanh(mu)
        
        value = self.mlp_2(x)
        value = torch.nn.LeakyReLU()(value)
        value = self.mlp_critic(value)

        prev_actions = input_dict.get('prev_actions', None)
        distr = torch.distributions.Normal(mu, self.fixed_sigma)
        logstd = torch.log(self.fixed_sigma)

        if is_train:
            prev_neglogp = self.neglogp(prev_actions, mu, self.fixed_sigma, logstd)
            entropy = distr.entropy().sum(dim=-1)

            result = {'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'rnn_states': None,
            'mus': mu,
            'sigmas': self.fixed_sigma,
            }
            return result

        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, self.fixed_sigma, logstd)

            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : value,
                'actions' : selected_action,
                'rnn_states' : None,
                'mus' : mu,
                'sigmas' : self.fixed_sigma
            }
            return result

    def eval_critic(self, obs):
        value = self.mlp_2(obs)
        value = self.mlp_critic(value)
        return value
    
    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)


class MultiDistillModel(torch.nn.Module):
    def __init__(self, input_size, output_size, rnn=False):
        super().__init__()
        self.rnn = rnn

        hidden = 256
        self.mlp = torch.nn.Linear(input_size, hidden, device='cuda')
        self.mu_heads = torch.nn.ModuleList([torch.nn.Linear(hidden, output_size, device='cuda') for i in range(2)])
        self.fixed_sigma = torch.nn.Parameter(data=torch.ones(size=(output_size,), device='cuda'), requires_grad=False) * .1

    def forward(self, input_dict):
        is_train = False #input_dict.get('is_train', True)
        obs = input_dict['obs']
        skill_id = input_dict['skill_id']

        # obs = obs.permute((0, 3, 1, 2))
        x = obs

        mu = self.mlp(x)
        mu = torch.nn.LeakyReLU()(mu)
        mu = self.mu_heads[skill_id](mu)
        mu = torch.tanh(mu)
        
        prev_actions = input_dict.get('prev_actions', None)
        distr = torch.distributions.Normal(mu, self.fixed_sigma)
        logstd = torch.log(self.fixed_sigma)

        if is_train:
            prev_neglogp = self.neglogp(prev_actions, mu, self.fixed_sigma, logstd)
            entropy = distr.entropy().sum(dim=-1)

            result = {'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': None,
            'entropy': entropy,
            'rnn_states': None,
            'mus': mu,
            'sigmas': self.fixed_sigma,
            }
            return result

        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, self.fixed_sigma, logstd)

            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : None,
                'actions' : selected_action,
                'rnn_states' : None,
                'mus' : mu,
                'sigmas' : self.fixed_sigma
            }
            return result
    
    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)


class ExpDataset(Dataset):
    def __init__(self, data_path, input_size, rnn=False):
        
        # load data
        self.rnn = rnn
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        obs = torch.tensor([obs.cpu().numpy() for obs in data['obs_record']])
        latent_mu = torch.tensor([latent.cpu().numpy() for latent in data['latent_mu_record']])
        latent_sigma = torch.tensor([latent.cpu().numpy() for latent in data['latent_sigma_record']])
        
        obs = obs[:obs.shape[0]//2]
        latent_mu = latent_mu[:latent_mu.shape[0]//2]
        latent_sigma = latent_sigma[:latent_sigma.shape[0]//2]
        
        try:
            values = torch.tensor([value.cpu().numpy() for value in data['value_record']])
        except:
            ...
        self._obs = torch.reshape(obs, (-1, input_size))
        self._latent_mu = torch.reshape(latent_mu, (-1, 64))
        self._latent_sigma = torch.reshape(latent_sigma, (-1, 64))
        try:
            self._values = torch.reshape(values, (-1, 1))
        except:
            ...

        del obs, latent_mu, latent_sigma

        self._running_mean_std = data['running_mean_std']

    def __len__(self):
        return self._obs.shape[0] 
    
    def __getitem__(self, idx):
        return self._obs[idx], {'mus': self._latent_mu[idx],
                                'sigmas': self._latent_sigma[idx],
                                # 'values': self._values[idx],
                                }
    
    def get_running_mean_std(self):
        return self._running_mean_std

class MultiExpDataset(Dataset):
    def __init__(self, data_path, skill_id, input_size, rnn=False):
        
        self.rnn = rnn
        max_obs_size = 239
        
        # load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        obs = torch.tensor([obs.cpu().numpy() for obs in data['obs_record']])
        latent_mu = torch.tensor([latent.cpu().numpy() for latent in data['latent_mu_record']])
        latent_sigma = torch.tensor([latent.cpu().numpy() for latent in data['latent_sigma_record']])
        values = torch.tensor([value.cpu().numpy() for value in data['value_record']])

        self._obs = torch.reshape(obs, (-1, input_size)) # 237
        self._latent_mu = torch.reshape(latent_mu, (-1, 64))
        self._latent_sigma = torch.reshape(latent_sigma, (-1, 64))
        self._values = torch.reshape(values, (-1, 1))

        self._obs_padded = torch.empty(size=(self._obs.shape[0], max_obs_size))
        self._obs_padded[:, :input_size] = self._obs
        self._obs_padded[:, input_size:max_obs_size-2] = 0
        self._obs_padded[:, max_obs_size-2] = 1 if skill_id==0 else 0
        self._obs_padded[:, max_obs_size-1] = 1 if skill_id==1 else 0

        self._running_mean_std = data['running_mean_std']

    def __len__(self):
        return self._obs.shape[0] 
    
    def __getitem__(self, idx):
        return self._obs_padded[idx], {'mus': self._latent_mu[idx],
                                'sigmas': self._latent_sigma[idx],
                                'values': self._values[idx],
                                }

    def get_running_mean_std(self):
        return self._running_mean_std

class MultiMoEDataset(Dataset):
    def __init__(self, data_path, skill_id, input_size, rnn=False):
        
        self.rnn = rnn
        self._proprio_size = 223
        self._context_size = 14
        self._task_id_size = 2
        self._max_obs_size = 237 + self._task_id_size
        
        # load data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        obs = torch.tensor([obs.cpu().numpy() for obs in data['obs_record']])
        latent_mu = torch.tensor([latent.cpu().numpy() for latent in data['latent_mu_record']])
        latent_sigma = torch.tensor([latent.cpu().numpy() for latent in data['latent_sigma_record']])
        
        obs = obs[:obs.shape[0]//4]
        latent_mu = latent_mu[:latent_mu.shape[0]//4]
        latent_sigma = latent_sigma[:latent_sigma.shape[0]//4]
        
        # values = torch.tensor([value.cpu().numpy() for value in data['value_record']])

        self._obs = torch.reshape(obs, (-1, input_size)) # 237 or 228
        self._latent_mu = torch.reshape(latent_mu, (-1, 64))
        self._latent_sigma = torch.reshape(latent_sigma, (-1, 64))
        # self._values = torch.reshape(values, (-1, 1))

        # self._obs_padded = self._obs
        self._obs_padded = torch.empty(size=(self._obs.shape[0], self._max_obs_size))
        self._obs_padded[:, :input_size] = self._obs
        self._obs_padded[:, input_size:] = 0

        ones = torch.ones([self._task_id_size // 2])
        zeros = torch.zeros([self._task_id_size // 2])
        if skill_id == 0:
            task_ind = torch.cat([ones, zeros])
        elif skill_id == 1:
            task_ind = torch.cat([zeros, ones])

        self._obs_padded[:, -self._task_id_size:] = task_ind

        del obs, latent_mu, latent_sigma#, self._obs

        self._running_mean_std = data['running_mean_std']

    def __len__(self):
        return self._obs_padded.shape[0]
    
    def __getitem__(self, idx):
        return self._obs_padded[idx], {'mus': self._latent_mu[idx],
                                'sigmas': self._latent_sigma[idx],
                                # 'values': None,
                                }

    def get_running_mean_std(self):
        return self._running_mean_std


class MoELayer(torch.nn.Module):
    def __init__(self, input_size, output_size, num_experts, device):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList([torch.nn.Linear(input_size, output_size, device=device) for _ in range(num_experts)])
        self.gate = torch.nn.Linear(input_size, num_experts, device=device)

    def forward(self, x):
        # 게이트 출력을 계산 (각 전문가의 가중치)
        gate_outputs = F.softmax(self.gate(x), dim=1)

        # 각 전문가의 출력 계산
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # 게이트 출력과 전문가 출력의 가중합 계산
        output = torch.bmm(gate_outputs.unsqueeze(1), expert_outputs).squeeze(1)
        return output

class SingleMoEDistill(torch.nn.Module):
    def __init__(self, input_size, output_size, rnn=False):
        super().__init__()
        self.rnn = rnn

        hidden = 256
        # self.mlp = torch.nn.Linear(input_size, hidden, device='cuda')
        self.moe = MoELayer(input_size, hidden, 4, device='cuda')
        self.mlp_mu = torch.nn.Linear(hidden, output_size, device='cuda')
        self.mlp_sigma = torch.nn.Linear(hidden, output_size, device='cuda')
        self.fixed_sigma = torch.nn.Parameter(data=torch.ones(size=(output_size,), device='cuda'), requires_grad=False) * .1

    def forward(self, input_dict):
        is_train = False #input_dict.get('is_train', True)
        obs = input_dict['obs']
        # obs = obs.permute((0, 3, 1, 2))
        x = obs

        mu = self.moe(x)
        mu = torch.nn.LeakyReLU()(mu)
        mu = self.mlp_mu(mu)
        mu = torch.tanh(mu)
        
        prev_actions = input_dict.get('prev_actions', None)
        distr = torch.distributions.Normal(mu, self.fixed_sigma)
        logstd = torch.log(self.fixed_sigma)

        if is_train:
            prev_neglogp = self.neglogp(prev_actions, mu, self.fixed_sigma, logstd)
            entropy = distr.entropy().sum(dim=-1)

            result = {'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': None,
            'entropy': entropy,
            'rnn_states': None,
            'mus': mu,
            'sigmas': self.fixed_sigma,
            }
            return result

        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, self.fixed_sigma, logstd)

            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : None,
                'actions' : selected_action,
                'rnn_states' : None,
                'mus' : mu,
                'sigmas' : self.fixed_sigma
            }
            return result

    def eval_critic(self, obs):
        value = self.mlp_2(obs)
        value = self.mlp_critic(value)
        return value
    
    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)


class MoEDistillModel(torch.nn.Module):
    def __init__(self, output_size, rnn=False):
        super().__init__()
        self.rnn = rnn
        self._proprio_size = 223
        self._context_size = 237 - self._proprio_size # 14
        self._task_id_size = 2
        self._num_heads = 2
        hidden = 512

        self.proprio_moe = torch.nn.Linear(self._proprio_size+self._context_size, hidden, device='cuda')
        # self.task_ind_layer = torch.nn.Linear(self._task_id_size, hidden, device='cuda')

        self.mlp = torch.nn.Linear(hidden + self._task_id_size, hidden, device='cuda')
        # self.main_moe = MoELayer(input_size=hidden, output_size=hidden, num_experts=4, device='cuda')
        self.mu_heads = torch.nn.ModuleList([torch.nn.Linear(hidden, output_size, device='cuda') for i in range(self._num_heads)])
        self.fixed_sigma = torch.nn.Parameter(data=torch.ones(size=(output_size,), device='cuda'), requires_grad=False) * .1

    def forward(self, input_dict):
        is_train = False #input_dict.get('is_train', True)
        obs = input_dict['obs']
        skill_id = input_dict['skill_id']

        proprio = obs[:, :self._proprio_size+self._context_size]
        task_id = obs[:, -self._task_id_size:]

        proprio = torch.nn.LeakyReLU()(self.proprio_moe(proprio))
        # task_id = torch.nn.LeakyReLU()(self.task_ind_layer(task_id))

        cat = torch.cat([proprio, task_id], dim=-1)
        cat = torch.nn.LeakyReLU()(self.mlp(cat))

        # cat = torch.nn.LeakyReLU()(self.main_moe(cat))
        mu = self.mu_heads[skill_id](cat)
        mu = torch.tanh(mu)
        
        prev_actions = input_dict.get('prev_actions', None)
        distr = torch.distributions.Normal(mu, self.fixed_sigma)
        logstd = torch.log(self.fixed_sigma)

        if is_train:
            prev_neglogp = self.neglogp(prev_actions, mu, self.fixed_sigma, logstd)
            entropy = distr.entropy().sum(dim=-1)

            result = {'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': None,
            'entropy': entropy,
            'rnn_states': None,
            'mus': mu,
            'sigmas': self.fixed_sigma,
            }
            return result

        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, self.fixed_sigma, logstd)

            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : None,
                'actions' : selected_action,
                'rnn_states' : None,
                'mus' : mu,
                'sigmas' : self.fixed_sigma
            }
            return result
    
    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)

class MultiDistillModel_2(torch.nn.Module):
    # task id embedding 을 layer 마다 넣어줌
    def __init__(self, output_size, rnn=False):
        super().__init__()
        self.rnn = rnn
        self._proprio_size = 223
        self._context_size = 237 - self._proprio_size # 16
        self._task_id_size = 64
        self._num_heads = 2
        hidden = 64

        self.mlp_1 = torch.nn.Linear(input_size=self._proprio_size, output_size=hidden, device='cuda')
        self.task_ind_layer = torch.nn.Linear(self._task_id_size, hidden, device='cuda')

        self.mlp_2 = torch.nn.Linear(hidden * 2, hidden, device='cuda')
        self.mlp_3 = torch.nn.Linear(input_size=hidden * 2, output_size=hidden * 2, device='cuda')
        self.mlp_4 = torch.nn.Linear(input_size=hidden * 2, output_size=hidden * 2, device='cuda')
        self.mu_heads = torch.nn.ModuleList([torch.nn.Linear(hidden * 2, output_size, device='cuda') for i in range(self._num_heads)])
        self.fixed_sigma = torch.nn.Parameter(data=torch.ones(size=(output_size,), device='cuda'), requires_grad=False) * .1

    def forward(self, input_dict):
        is_train = False #input_dict.get('is_train', True)
        obs = input_dict['obs']
        skill_id = input_dict['skill_id']

        mu = obs[:, :self._proprio_size + self._context_size]
        task_id = obs[:, -self._task_id_size:]

        mu = torch.nn.LeakyReLU()(self.mlp_1(mu))
        task_id = torch.nn.LeakyReLU()(self.task_ind_layer(task_id))

        mu = torch.cat([mu, task_id], dim=-1)
        mu = torch.nn.LeakyReLU()(self.mlp_2(mu))

        mu = torch.cat([mu, task_id])
        mu = torch.nn.LeakyReLU()(self.mlp_3(mu))
        
        mu = torch.cat([mu, task_id])
        # mu = torch.nn.LeakyReLU()(self.mlp_4(mu))

        mu = self.mu_heads[skill_id](mu)
        mu = torch.tanh(mu)
        
        prev_actions = input_dict.get('prev_actions', None)
        distr = torch.distributions.Normal(mu, self.fixed_sigma)
        logstd = torch.log(self.fixed_sigma)

        if is_train:
            prev_neglogp = self.neglogp(prev_actions, mu, self.fixed_sigma, logstd)
            entropy = distr.entropy().sum(dim=-1)

            result = {'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': None,
            'entropy': entropy,
            'rnn_states': None,
            'mus': mu,
            'sigmas': self.fixed_sigma,
            }
            return result

        else:
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, self.fixed_sigma, logstd)

            result = {
                'neglogpacs' : torch.squeeze(neglogp),
                'values' : None,
                'actions' : selected_action,
                'rnn_states' : None,
                'mus' : mu,
                'sigmas' : self.fixed_sigma
            }
            return result
    
    def neglogp(self, x, mean, std, logstd):
        return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
            + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
            + logstd.sum(dim=-1)
