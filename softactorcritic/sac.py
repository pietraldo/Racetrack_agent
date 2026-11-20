import pickle
import random
from collections import deque
from typing import cast

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from softactorcritic.buffer import ReplayBuffer
from softactorcritic.models import Actor, Critic
from softactorcritic.scheduler import Scheduler


class SAC:
    def __init__(
            self,
            policy: str,
            env: VecEnv, 
            learning_rate=0.0003, 
            buffer_size=1000000, 
            learning_starts=100, 
            batch_size=256, 
            tau=0.005, 
            gamma=0.99, 
            train_freq=1, 
            gradient_steps=1, 
            # action_noise=None, 
            # replay_buffer_class=None, 
            # replay_buffer_kwargs=None, 
            # optimize_memory_usage=False, 
            # n_steps=1, 
            ent_coef='auto', 
            target_update_interval=1, 
            target_entropy='auto', 
            actor_arch: list[int] | None = None,
            critic_arch: list[int] | None = None,
            # use_sde=False, 
            # sde_sample_freq=-1, 
            # use_sde_at_warmup=False, 
            # stats_window_size=100, 
            # tensorboard_log=None, 
            # policy_kwargs=None, 
            verbose=0, 
            seed=None, 
            device='auto', 
            # _init_setup_model=True
            tensorboard_log='logs'
    ) -> None:
        if policy != 'MlpPolicy':
            raise NotImplementedError('Only MlpPolicy supported')

        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log

        # Fixed entropy coefficient; a float ent_coef overrides the default 0.2 ('auto' tuning is not implemented)
        self.alpha = float(ent_coef) if not isinstance(ent_coef, str) else 0.2
        actor_arch = actor_arch if actor_arch else [256, 256]
        critic_arch = critic_arch if critic_arch else [256, 256]

        self.actor = Actor(actor_arch, env.observation_space, env.action_space)
        self.critic = Critic(critic_arch, env.observation_space, env.action_space)
        self.critic_target = AveragedModel(
             self.critic,
             multi_avg_fn=get_ema_multi_avg_fn(1-tau)
        )
        self.critic_target.update_parameters(self.critic)

        self.actor_optim = th.optim.Adam(
            self.actor.parameters(), 
            lr=self.learning_rate,
            maximize=True,
        )
        self.critic_optim = th.optim.Adam(
            self.critic.parameters(), 
            lr=self.learning_rate
        )
        
        assert self.env.observation_space.shape is not None
        assert self.env.action_space.shape is not None
        self.replay_buffer = ReplayBuffer(
            self.buffer_size, 
            self.env.observation_space.shape,
            self.env.action_space.shape,
        )
        self.scheduler = Scheduler(
            learning_starts, 
            train_freq, 
            target_update_interval
        )

    def _scale_action(self, a):
        low_bound = self.env.action_space.low
        high_bound = self.env.action_space.high
        return low_bound + (a + 1.0) * 0.5 * (high_bound - low_bound)

    def train(self, gradient_steps, batch_size=64):
        for _ in range(gradient_steps):
            batch_size = min(batch_size, self.replay_buffer.size())
            obs, actions, rewards, new_obs, dones = self.replay_buffer.sample(batch_size)

            obs = th.as_tensor(obs, dtype=th.float32)
            actions = th.as_tensor(actions, dtype=th.float32)
            new_obs = th.as_tensor(new_obs, dtype=th.float32)
            rewards = th.as_tensor(rewards, dtype=th.float32).unsqueeze(-1)
            dones = th.as_tensor(dones, dtype=th.float32).unsqueeze(-1)
            
            # Crtitic gradient step
            with th.no_grad():
                next_action, next_log_prob = self.actor.action_log_prob(new_obs)
                tq1, tq2 = self.critic_target(th.cat((new_obs, next_action), dim=-1))
                q_target = rewards + self.gamma * (1-dones) * \
                    (th.min(tq1, tq2) - self.alpha * next_log_prob)
            q1, q2 = self.critic(th.cat((obs, actions), dim=-1))
            q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

            # Actor gradient step
            action, log_prob = self.actor.action_log_prob(obs)
            q1, q2 = self.critic(th.cat((obs, action), dim=-1))
            actor_loss = (th.min(q1, q2) - self.alpha * log_prob).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Target critic step
            if self.scheduler.should_train_target():
                self.critic_target.update_parameters(self.critic)

            self.scheduler.did_grad_step()        

    def learn(
        self, 
        total_timesteps: int, 
        log_interval: int | None = None, 
        callback: BaseCallback | list[BaseCallback] | None = None,
        reset_num_timesteps: bool = True,
    ):
        if reset_num_timesteps:
            self.scheduler.reset()
        
        self.scheduler.log_interval = log_interval

        if callback is None:
            callback = []
        elif isinstance(callback, BaseCallback):
            callback = [callback]

        [c._on_training_start() for c in callback]
        [c._on_rollout_start() for c in callback]

        obs: np.ndarray = self.env.reset() # type: ignore
        for _ in range(total_timesteps):
            with th.no_grad():
                actions = self.actor(th.as_tensor(obs)).detach().numpy()
            scaled_actions = self._scale_action(actions)
            new_obs, rewards, dones, _ = self.env.step(scaled_actions)
            new_obs = cast(np.ndarray, new_obs)
            [c._on_step() for c in callback]
            
            self.replay_buffer.extend(obs, actions, rewards, new_obs, dones)

            # if terminated or truncated:
            #     obs, _ = self.env.reset()
            # else:
            obs = new_obs # type: ignore

            if self.scheduler.should_train():
                [c._on_rollout_end() for c in callback]
                self.train(self.gradient_steps, self.batch_size)
                [c._on_rollout_start() for c in callback]

            if self.scheduler.should_log():
                print(f'Step: {self.scheduler.steps}')

            self.scheduler.did_step(new_obs.shape[0])

        [c._on_training_end() for c in callback]
    
    def predict(self, observation, deterministic=True):
        with th.no_grad():
            if deterministic:
                action = self.actor.action_deterministic(th.as_tensor(observation))
            else:
                action = self.actor.forward(th.as_tensor(observation))
            
            action = self._scale_action(action.detach().numpy())
            return action, None

    def __getstate__(self):
        # The env and the replay buffer are not saved; the target critic is stored as a state_dict,
        # because AveragedModel holds a local function that pickle cannot serialize
        state = self.__dict__.copy()
        state["env"] = None
        state["replay_buffer"] = ReplayBuffer(
            self.replay_buffer.capacity,
            self.replay_buffer.observation_shape,
            self.replay_buffer.action_shape,
        )
        state["critic_target"] = self.critic_target.state_dict()
        return state

    def __setstate__(self, state):
        target_state = state.pop("critic_target")
        self.__dict__.update(state)
        self.critic_target = AveragedModel(self.critic, multi_avg_fn=get_ema_multi_avg_fn(1 - self.tau))
        self.critic_target.load_state_dict(target_state)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path, **kwargs):
        with open(path, "rb") as f:
            sac = pickle.load(f)
            sac.__dict__.update(kwargs)
            return sac
        



