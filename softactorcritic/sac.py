import pickle
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from softactorcritic.models import Actor, Critic


class Scheduler:
    def __init__(
        self,
        learning_starts,
        train_freq,
        target_update_interval,
        log_interval = None
    ):
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.log_interval = log_interval

        self.learning_started = False
        self.steps = 0
        self.grad_steps = 0

    def did_step(self):
        self.steps += 1
        if not self.learning_started and self.steps >= self.learning_starts:
             self.learning_started = True

    def did_grad_step(self):
        self.grad_steps += 1

    def should_train(self):
        return (
            self.learning_started and
            self.steps % self.train_freq == 0
        ) 
    
    def should_train_target(self):
         return self.grad_steps % self.target_update_interval == 0
    
    def should_log(self):
        return self.log_interval and self.steps % self.log_interval == 0
        
    def reset(self):
        self.learning_started = False
        self.steps = 0
        self.grad_steps = 0

class SAC:
    def __init__(
            self,
            policy: str,
            env: gym.Env, 
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

        self.alpha = 0.2
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
        
        self.replay_buffer = deque([], maxlen=buffer_size)
        self.scheduler = Scheduler(
            learning_starts, 
            train_freq, 
            target_update_interval
        )

    def _scale_action(self, a):
        low_bound = th.Tensor(self.env.action_space.low)
        high_bound = th.Tensor(self.env.action_space.high)
        return th.lerp(low_bound, high_bound, (a + 1.0) * 0.5)

    def train(self, gradient_steps, batch_size=64):
        for _ in range(gradient_steps):
            batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
            # obs, actions, rewards, new_obs, dones = \
            #     [th.as_tensor(lst) for lst in zip(*batch)]
            
            obs, actions, rewards, new_obs, dones = zip(*batch)

            obs      = th.as_tensor(np.stack(obs),      dtype=th.float32)
            actions  = th.as_tensor(np.stack(actions),  dtype=th.float32)
            new_obs  = th.as_tensor(np.stack(new_obs),  dtype=th.float32)
            rewards  = th.as_tensor(rewards, dtype=th.float32).unsqueeze(-1)
            dones    = th.as_tensor(dones,   dtype=th.float32).unsqueeze(-1)
            
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
        log_interval: int = 0, 
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

        obs, _ = self.env.reset()
        for _ in range(total_timesteps):
            with th.no_grad():
                action = self.actor(th.as_tensor(obs)).detach()
                scaled_action = self._scale_action(action).numpy()
            new_obs, reward, terminated, truncated, _ = self.env.step(scaled_action)
            [c._on_step() for c in callback]
            
            self.replay_buffer.append((obs, action, reward, new_obs, terminated))

            if terminated or truncated:
                obs, _ = self.env.reset()
            else:
                obs = new_obs

            if self.scheduler.should_train():
                [c._on_rollout_end() for c in callback]
                self.train(self.gradient_steps, self.batch_size)
                [c._on_rollout_start() for c in callback]

            if self.scheduler.should_log():
                print(f'Step: {self.scheduler.steps}')

            self.scheduler.did_step()

        [c._on_training_end() for c in callback]
    
    def predict(self, observation, deterministic=True):
        with th.no_grad():
            if deterministic:
                action = self.actor.action_deterministic(th.as_tensor(observation))
            else:
                action = self.actor.forward(th.as_tensor(observation))
            
            action = self._scale_action(action).detach().numpy()
            return action, None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path, **kwargs):
        with open(path, "rb") as f:
            sac = pickle.load(f)
            sac.__dict__.update(kwargs)
            return sac
        



