import glob
import io
import sys
import gymnasium as gym
import highway_env  # registers racetrack envs
import matplotlib.pyplot as plt
import os
from collections import deque
from highway_env.vehicle.behavior import IDMVehicle

import numpy as np
import pandas as pd
import torch
# from stable_baselines3 import SAC
from softactorcritic import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

# Trains SAC on racetrack-v1 in stages (curriculum), saves the model and plots learning statistics

TIMESTEPS = 600_000
N_ENVS = 8  # number of envs simulated in parallel processes
SAVE_FREQ = 100 #10_000  # how often (in timesteps) a new checkpoint is saved (~6 MB each), watch_agent.py shows the newest
CHECKPOINT_DIR = "sac_checkpoints"
STAGE_FILE = "sac_checkpoints/stage.txt"  # current stage, read by watch_agent.py
# Path to a saved model to continue training, e.g. "sac_checkpoints/model_000158000.zip"; None = train from scratch.
# Models trained before the observation got speed and steering added cannot be loaded.
RESUME_FROM = None#"sac_checkpoints/latest.zip"
START_STAGE = 2  # stage to start from (useful together with RESUME_FROM)

CONFIG = {
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "speed_range": [0, 20],  # min speed 0 -> the car cannot drive backwards
        "steering_range": [-np.pi / 4, np.pi / 4],  # default is ±45°, which allows very sharp turns
    },
    "other_vehicles": 0,  # env spawns none, RacetrackWrapper places cars according to the current stage
    "controlled_vehicles": 1,
    "speed_limit": 30,
    "duration": 60,  # seconds, episode has duration * policy_frequency steps
    "policy_frequency": 5,
}
MAX_SPEED = CONFIG["action"]["speed_range"][1]
EPISODE_STEPS = CONFIG["duration"] * CONFIG["policy_frequency"]

# Reward, replaces the env's own reward (which paid ~0.5 per step just for surviving, so standing still was optimal)
SPEED_REWARD = 1.0  # per step at max speed along the track
CENTERING_REWARD = 0.5  # extra per step at max speed when exactly in the lane center
LANE_CENTERING_COST = 4  # how fast the centering bonus drops with distance from the center
STEERING_COST = 0.2  # penalty for |steering|, discourages turning when not needed
STEERING_CHANGE_COST = 0.3  # penalty for |steering change| between steps, discourages jerky turns
CRASH_PENALTY = -5  # crash, leaving the road or standing still
IDLE_SPEED = 1.0  # m/s, below this the car counts as standing
IDLE_STEPS = 25  # standing for this many steps (5 s) ends the episode with CRASH_PENALTY

# Curriculum: the next stage starts when the mean length of the last ADVANCE_EPISODES episodes
# is at least ADVANCE_LENGTH of the max length (the car stops crashing), or after max_timesteps in the stage
# "vehicles": (min, max) other cars, the number is drawn uniformly every episode
STAGES = [
    {"name": "empty track", "vehicles": (0, 0), "max_timesteps": 100_000},
    {"name": "one car", "vehicles": (1, 1), "max_timesteps": 100_000},
    {"name": "light traffic", "vehicles": (1, 5), "max_timesteps": 150_000},
    {"name": "heavy traffic", "vehicles": (10, 20), "max_timesteps": None},  # last stage runs until TIMESTEPS
]
VEHICLE_SPACING = 12  # m, min distance between spawned cars (env uses 20 m, which fits only ~12 cars)
EGO_SPACING = 20  # m, min distance between a spawned car and the agent
ADVANCE_LENGTH = 0.8
ADVANCE_EPISODES = 50
EASY_EPISODE_PROB = 0.2  # share of episodes played on the previous stage, so the agent does not forget it


class RacetrackWrapper(gym.Wrapper):
    """Own reward, speed and previous steering added to the observation, idle cutoff and curriculum stages."""

    def __init__(self, env, stage=0):
        super().__init__(env)
        size = int(np.prod(env.observation_space.shape)) + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)
        self.stage = stage
        self.easy_episode_prob = EASY_EPISODE_PROB
        self.prev_steering = 0.0
        self.idle_steps = 0

    def set_stage(self, stage):
        self.stage = stage  # applied on the next reset

    def _observation(self, obs):
        speed = self.env.unwrapped.vehicle.speed / MAX_SPEED
        return np.concatenate([obs.ravel(), [speed, self.prev_steering]]).astype(np.float32)

    def reset(self, **kwargs):
        stage = self.stage
        if stage > 0 and self.np_random.random() < self.easy_episode_prob:
            stage -= 1
        self.prev_steering = 0.0
        self.idle_steps = 0
        _, info = self.env.reset(**kwargs)
        low, high = STAGES[stage]["vehicles"]
        self._spawn_vehicles(self.np_random.integers(low, high + 1))
        obs = self.env.unwrapped.observation_type.observe()  # observe again, now with the spawned cars
        return self._observation(obs), info

    def _spawn_vehicles(self, count):
        """Places count cars at random free spots of the track (env's own spawning gives a random, smaller number)."""
        road, ego = self.env.unwrapped.road, self.env.unwrapped.vehicle
        for _ in range(count * 20):  # attempts; on a full track fewer cars are placed
            if len(road.vehicles) - 1 >= count:
                break
            lane_index = road.network.random_lane_index(self.np_random)
            lane = road.network.get_lane(lane_index)
            vehicle = IDMVehicle.make_on_lane(
                road,
                lane_index,
                longitudinal=self.np_random.uniform(0, lane.length),
                speed=6.0 + self.np_random.uniform(high=3.0),
            )
            if np.linalg.norm(vehicle.position - ego.position) < EGO_SPACING:
                continue
            if all(np.linalg.norm(vehicle.position - v.position) >= VEHICLE_SPACING for v in road.vehicles):
                road.vehicles.append(vehicle)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        vehicle = self.env.unwrapped.vehicle
        longitudinal, lateral = vehicle.lane.local_coordinates(vehicle.position)
        lane_heading = vehicle.lane.heading_at(longitudinal)
        forward = np.clip(vehicle.speed * np.cos(vehicle.heading - lane_heading) / MAX_SPEED, -1, 1)
        centering = 1 / (1 + LANE_CENTERING_COST * lateral**2)

        steering = float(np.clip(action[1], -1, 1))  # action = [acceleration, steering]
        steering_penalty = STEERING_COST * abs(steering) + STEERING_CHANGE_COST * abs(steering - self.prev_steering)
        self.prev_steering = steering

        # Bonuses are paid only for moving forward, so standing still gives nothing
        reward = forward * (SPEED_REWARD + CENTERING_REWARD * centering) - steering_penalty

        self.idle_steps = self.idle_steps + 1 if vehicle.speed < IDLE_SPEED else 0
        idle = self.idle_steps >= IDLE_STEPS
        if terminated or idle:  # env terminates on crash or leaving the road
            reward = CRASH_PENALTY
            terminated = True

        info["forward_speed"] = forward
        info["steering_penalty"] = steering_penalty
        info["idle"] = idle
        return self._observation(obs), reward, terminated, truncated, info


def make_env(render_mode=None, stage=0):
    env = gym.make("racetrack-v1", render_mode=render_mode, config=CONFIG)
    return RacetrackWrapper(env, stage)


# torch fails to read the weights directly from the file inside SB3's zip ("PytorchStreamReader failed"),
# so they are read into memory first. Applies to every SAC.load in this file and in files importing it.
_torch_load = torch.load
torch.load = lambda f, *args, **kwargs: _torch_load(io.BytesIO(f.read()) if hasattr(f, "read") else f, *args, **kwargs)


def checkpoint_path(timesteps):
    return f"{CHECKPOINT_DIR}/model_{timesteps:09d}.zip"  # zero padding keeps files sorted by timesteps


def newest_checkpoint():
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/model_*.zip"))
    return checkpoints[-1] if checkpoints else None


def read_stage():
    try:
        with open(STAGE_FILE) as f:
            return int(f.read())
    except (OSError, ValueError):
        return 0


def write_stage(stage):
    with open(STAGE_FILE, "w") as f:
        f.write(str(stage))


class SaveCheckpoint(BaseCallback):
    """Every save_freq timesteps saves the model to a new file, existing files are never overwritten."""

    def __init__(self, save_freq):
        super().__init__()
        self.save_freq = save_freq

    def _on_training_start(self):
        self.last_save = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self.last_save >= self.save_freq:
            self.last_save = self.num_timesteps
            self.model.save(checkpoint_path(self.num_timesteps))
        return True


class Curriculum(BaseCallback):
    """Moves all envs to the next stage when the agent masters the current one."""

    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.lengths = deque(maxlen=ADVANCE_EPISODES)

    def _on_training_start(self):
        self.stage_start = self.num_timesteps
        write_stage(self.stage)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:  # added by VecMonitor when an episode ends
                self.lengths.append(info["episode"]["l"])
        self.logger.record("curriculum/stage", self.stage)

        if self.stage == len(STAGES) - 1:
            return True
        mastered = (
            len(self.lengths) == ADVANCE_EPISODES and np.mean(self.lengths) >= ADVANCE_LENGTH * EPISODE_STEPS
        )
        max_timesteps = STAGES[self.stage]["max_timesteps"]
        if mastered or self.num_timesteps - self.stage_start >= max_timesteps:
            self.stage += 1
            self.stage_start = self.num_timesteps
            self.lengths.clear()
            self.training_env.env_method("set_stage", self.stage)
            write_stage(self.stage)
            reason = "mastered" if mastered else "time limit"
            print(f"\n=== Stage {self.stage}: {STAGES[self.stage]['name']} ({reason}) at {self.num_timesteps} ===\n")
        return True


# Subprocesses re-import this file on Windows, so training must be under the main guard
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # N_ENVS copies of the env, each in its own process
    # env = make_vec_env(make_env, n_envs=N_ENVS, env_kwargs={"stage": START_STAGE}, vec_env_cls=SubprocVecEnv)
    env = make_env(stage=START_STAGE)
    # VecMonitor writes reward and length of every episode to sac_logs/monitor.csv
    # env = VecMonitor(env, "sac_logs/")
    env = Monitor(env, "sac_logs/")

    # verbose=1 prints a table with stats (mean reward, episode length, losses) during training
    if RESUME_FROM:
        model = SAC.load(RESUME_FROM, env=env, verbose=1, tensorboard_log="sac_logs/")
        print(f"Resuming from {RESUME_FROM} ({model.num_timesteps} timesteps already trained)")
    else:
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="sac_logs/")
    # reset_num_timesteps=False keeps counting timesteps from the loaded model; TIMESTEPS more are trained
    # callbacks = [SaveCheckpoint(SAVE_FREQ), Curriculum(START_STAGE)]
    callbacks: list[BaseCallback] = [SaveCheckpoint(SAVE_FREQ)]
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks, reset_num_timesteps=False)
    model.save(checkpoint_path(model.num_timesteps))
    model.save("sac_racetrack")  # copy of the final model for random_agent_stats.py
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
