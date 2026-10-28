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

# 2. Statistics, without a window (fast)
env = gym.make("racetrack-v1", config=CONFIG)
env.action_space.seed(0)

lengths, returns, crashes, off_road = [], [], 0, 0
for episode in range(EPISODES):
    env.reset(seed=episode)
    steps, total, done = 0, 0.0, False
    while not done:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        total += reward
        done = terminated or truncated
    lengths.append(steps)
    returns.append(total)
    crashes += env.unwrapped.vehicle.crashed
    off_road += not env.unwrapped.vehicle.on_road
env.close()

stats = (
    f"Losowy agent, {EPISODES} epizodów\n"
    f"średnia długość epizodu: {np.mean(lengths):.1f} kroków (~{np.mean(lengths) / 5:.1f} s)\n"
    f"średni zwrot: {np.mean(returns):.2f} (± {np.std(returns):.2f})\n"
    f"wyjazd z drogi: {off_road / EPISODES:.0%}\n"
    f"kolizja: {crashes / EPISODES:.0%}\n"
)
print(stats)
with open("random_agent_stats.txt", "w", encoding="utf-8") as f:
    f.write(stats)
