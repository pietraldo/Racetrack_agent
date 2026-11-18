import itertools

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class TanhDiagGaussian:
    @staticmethod
    def sample(mu, std):
        dist = TransformedDistribution(Normal(mu, std), [TanhTransform(cache_size=1)])
        return dist.rsample()
    
    @staticmethod
    def sample_log_prob(mu, std):
        dist = TransformedDistribution(Normal(mu, std), [TanhTransform(cache_size=1)])
        a = dist.rsample()
        return a, dist.log_prob(a).sum(-1, keepdim=True)

def create_ffn(input_dim: int, nn_arch: list[int]):
    # TODO: Normalizacja? W SB3 nie ma defaultowo ale może można dać
    # model = nn.Sequential(nn.Flatten())

    model = nn.Sequential()
    for in_neurons, out_neurons in itertools.pairwise([input_dim, *nn_arch]):
        model.append(nn.Linear(in_neurons, out_neurons))
        model.append(nn.ReLU())
    return model

class Actor(nn.Module):
    def __init__(
        self, 
        nn_arch: list[int],
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ):
        super().__init__()

        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(f"Only Box observation space supported, got {observation_space}")

        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"Only Box action space supported, got {action_space}")

        input_dim = np.prod(observation_space.shape)
        output_dim = np.prod(action_space.shape)
        
        self.model = create_ffn(input_dim, nn_arch)
        self.mu = nn.Linear(nn_arch[-1], output_dim)
        self.log_std = nn.Linear(nn_arch[-1], output_dim)

    def forward(self, x):
        model_out = self.model(x)
        mu = self.mu(model_out)
        log_std = self.log_std(model_out)

        return TanhDiagGaussian.sample(mu, th.exp(th.clamp(log_std, -20, 2)))
    
    def action_log_prob(self, x):
        model_out = self.model(x)
        mu = self.mu(model_out)
        log_std = self.log_std(model_out)

        return TanhDiagGaussian.sample_log_prob(mu, th.exp(th.clamp(log_std, -20, 2)))

    def action_deterministic(self, x):
        model_out = self.model(x)
        mu = self.mu(model_out)
        return th.tanh(mu)

class Critic(nn.Module):
    def __init__(
        self,
        nn_arch: list[int],
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,

    ):
        super().__init__()

        assert observation_space.shape is not None and action_space.shape is not None
        input_dim = np.prod(observation_space.shape) + np.prod(action_space.shape)
        
        self.q1 = create_ffn(input_dim, nn_arch)
        self.q1.append(nn.Linear(nn_arch[-1], 1))

        self.q2 = create_ffn(input_dim, nn_arch)
        self.q2.append(nn.Linear(nn_arch[-1], 1))

    def forward(self, x):
        return (self.q1(x), self.q2(x))