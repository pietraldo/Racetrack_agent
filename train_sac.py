import gymnasium as gym
import highway_env  # registers racetrack envs
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# Trains SAC on racetrack-v1, saves the model and plots learning statistics

TIMESTEPS = 50_000
CONFIG = {
    "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
    "other_vehicles": 3,
}

# Monitor writes reward and length of every episode to sac_logs/monitor.csv
env = Monitor(gym.make("racetrack-v1", config=CONFIG), "sac_logs/")

# verbose=1 prints a table with stats (mean reward, episode length, losses) during training
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="sac_logs/")
model.learn(total_timesteps=TIMESTEPS)
model.save("sac_racetrack")
env.close()

# Learning curves from monitor.csv
log = pd.read_csv("sac_logs/monitor.csv", skiprows=1)  # first line is a comment
fig, (ax_reward, ax_length) = plt.subplots(2, 1, sharex=True)
ax_reward.plot(log["r"], alpha=0.3, label="episode")
ax_reward.plot(log["r"].rolling(20).mean(), label="mean of last 20")
ax_reward.set_ylabel("reward")
ax_reward.legend()
ax_length.plot(log["l"], alpha=0.3)
ax_length.plot(log["l"].rolling(20).mean())
ax_length.set_ylabel("episode length")
ax_length.set_xlabel("episode")
fig.suptitle("SAC on racetrack-v1")
fig.savefig("sac_learning_curve.png")
plt.show()
