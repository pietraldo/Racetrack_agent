import gymnasium as gym
import highway_env  # registers racetrack envs
import numpy as np

# Random agent: shows a few episodes in a window, then computes statistics
# over 100 episodes and saves them to random_agent_stats.txt

SHOW_EPISODES = 10
EPISODES = 100
CONFIG = {
    "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
    "other_vehicles": 3,
}

# 1. Simulation in a window, in real time
env = gym.make("racetrack-v1", render_mode="human", config={**CONFIG, "real_time_rendering": True})
for episode in range(SHOW_EPISODES):
    env.reset(seed=episode)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
env.close()

