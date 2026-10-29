import io

import gymnasium as gym
import highway_env  # registers racetrack envs
import numpy as np
import torch
from stable_baselines3 import SAC

# torch fails to read weights straight from the model's .zip ("internal miniz error"),
# so read each file into memory first
_torch_load = torch.load
torch.load = lambda f, *args, **kwargs: _torch_load(io.BytesIO(f.read()) if hasattr(f, "read") else f, *args, **kwargs)

# Random or trained SAC agent: shows a few episodes in a window, then computes statistics
# over 100 episodes and saves them to random_agent_stats.txt

AGENT = "sac"  # "random" or "sac"
MODEL_PATH = "sac_racetrack"  # used only when AGENT = "sac"
SHOW_EPISODES = 10
EPISODES = 100
CONFIG = {
    "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
    "other_vehicles": 3,
}

model = SAC.load(MODEL_PATH) if AGENT == "sac" else None


def choose_action(env, obs):
    if model is None:
        return env.action_space.sample()
    action, _ = model.predict(obs, deterministic=True)
    return action


# 1. Simulation in a window, in real time
env = gym.make("racetrack-v1", render_mode="human", config={**CONFIG, "real_time_rendering": True})
for episode in range(SHOW_EPISODES):
    obs, _ = env.reset(seed=episode)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(choose_action(env, obs))
        done = terminated or truncated
env.close()
