import argparse
import csv
import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import highway_env  # registers racetrack envs
import numpy as np
import torch
from highway_env.vehicle.behavior import IDMVehicle
from stable_baselines3.common.callbacks import BaseCallback

from softactorcritic import SAC

# Trains our own SAC (softactorcritic/) on racetrack-v1 in stages (curriculum).
# Every run goes to runs/<name>/: config.json, episodes.csv (every training episode), eval.csv (periodic
# deterministic evaluation), curriculum.csv, checkpoints/. Plots: plot_training.py, evaluation: evaluate.py.
#
# Examples:
#   python train_sac.py --name curriculum_a0.2                      # empty -> one car -> light traffic
#   python train_sac.py --name no_curriculum --start-stage 2        # straight to light traffic
#   python train_sac.py --name curriculum_a0.05 --alpha 0.05

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
# max_timesteps are for a single env (our SAC collects one transition per step)
STAGES = [
    {"name": "empty track", "vehicles": (0, 0), "max_timesteps": 50_000},
    {"name": "one car", "vehicles": (1, 1), "max_timesteps": 30_000},
    {"name": "light traffic", "vehicles": (1, 5), "max_timesteps": None},
    {"name": "heavy traffic", "vehicles": (10, 20), "max_timesteps": None},
]
VEHICLE_SPACING = 12  # m, min distance between spawned cars (env uses 20 m, which fits only ~12 cars)
EGO_SPACING = 20  # m, min distance between a spawned car and the agent
ADVANCE_LENGTH = 0.8
ADVANCE_EPISODES = 30
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


def episode_outcome(info, terminated):
    return "idle" if info.get("idle") else "crash" if terminated else "timeout"


class EpisodeLogger(gym.Wrapper):
    """Collects per-episode statistics of the training env (return, length, outcome, mean forward speed)."""

    def __init__(self, env):
        super().__init__(env)
        self.episodes = []  # finished episodes, read by TrainingCallback
        self._ret, self._len, self._speed = 0.0, 0, 0.0

    def reset(self, **kwargs):
        self._ret, self._len, self._speed = 0.0, 0, 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += reward
        self._len += 1
        self._speed += info["forward_speed"]
        if terminated or truncated:
            self.episodes.append({
                "return": self._ret,
                "length": self._len,
                "outcome": episode_outcome(info, terminated),
                "forward_speed": self._speed / self._len * MAX_SPEED,  # m/s along the track
            })
        return obs, reward, terminated, truncated, info


def evaluate(model, stage, episodes, seed):
    """Deterministic policy on a fresh env with fixed seeds; returns mean statistics."""
    env = make_env(stage=stage)
    env.easy_episode_prob = 0
    results = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        ret, length, speed, done = 0.0, 0, 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            length += 1
            speed += info["forward_speed"]
            done = terminated or truncated
        results.append({
            "return": ret,
            "length": length,
            "outcome": episode_outcome(info, terminated),
            "forward_speed": speed / length * MAX_SPEED,
        })
    env.close()
    return results


class TrainingCallback(BaseCallback):
    """Called by our SAC after every env step: logs episodes, runs the curriculum, evaluates and saves checkpoints."""

    def __init__(self, model, env, run_dir, stage, final_stage, eval_freq, eval_episodes, save_freq, total):
        super().__init__()
        self.sac, self.env, self.run_dir = model, env, run_dir
        self.stage, self.final_stage = stage, final_stage
        self.eval_freq, self.eval_episodes, self.save_freq, self.total = eval_freq, eval_episodes, save_freq, total
        self.steps = 0
        self.stage_start = 0
        self.logged_episodes = 0
        self.lengths = deque(maxlen=ADVANCE_EPISODES)
        self.start_time = time.time()
        self.episodes_file = open(f"{run_dir}/episodes.csv", "w", newline="")
        self.episodes_csv = csv.writer(self.episodes_file)
        self.episodes_csv.writerow(["episode", "timesteps", "stage", "return", "length", "outcome", "forward_speed"])
        self.eval_file = open(f"{run_dir}/eval.csv", "w", newline="")
        self.eval_csv = csv.writer(self.eval_file)
        self.eval_csv.writerow(["timesteps", "stage", "return_mean", "return_std", "length_mean", "success_rate",
                                "crash_rate", "forward_speed"])
        self.curriculum_file = open(f"{run_dir}/curriculum.csv", "w", newline="")
        self.curriculum_csv = csv.writer(self.curriculum_file)
        self.curriculum_csv.writerow(["timesteps", "stage", "name", "reason"])
        self.curriculum_csv.writerow([0, stage, STAGES[stage]["name"], "start"])
        self.curriculum_file.flush()
        self._write_stage()

    def _write_stage(self):
        with open(f"{self.run_dir}/stage.txt", "w") as f:
            f.write(str(self.stage))

    def _on_step(self):
        self.steps += 1
        while self.logged_episodes < len(self.env.episodes):
            ep = self.env.episodes[self.logged_episodes]
            self.logged_episodes += 1
            self.lengths.append(ep["length"])
            self.episodes_csv.writerow([self.logged_episodes, self.steps, self.stage, f"{ep['return']:.3f}",
                                        ep["length"], ep["outcome"], f"{ep['forward_speed']:.3f}"])
        self._curriculum()
        if self.steps % self.eval_freq == 0 or self.steps == self.total:
            self._evaluate()
        if self.steps % self.save_freq == 0:
            self.sac.save(f"{self.run_dir}/checkpoints/model_{self.steps:09d}.pkl")
        return True

    def _curriculum(self):
        if self.stage >= self.final_stage:
            return
        mastered = len(self.lengths) == ADVANCE_EPISODES and np.mean(self.lengths) >= ADVANCE_LENGTH * EPISODE_STEPS
        max_timesteps = STAGES[self.stage]["max_timesteps"]
        if mastered or (max_timesteps and self.steps - self.stage_start >= max_timesteps):
            self.stage += 1
            self.stage_start = self.steps
            self.lengths.clear()
            self.env.unwrapped_wrapper.set_stage(self.stage)
            self._write_stage()
            reason = "mastered" if mastered else "time limit"
            self.curriculum_csv.writerow([self.steps, self.stage, STAGES[self.stage]["name"], reason])
            self.curriculum_file.flush()
            print(f"=== Stage {self.stage}: {STAGES[self.stage]['name']} ({reason}) at {self.steps} ===", flush=True)

    def _evaluate(self):
        # Always evaluated on the final (target) stage, so curves of different runs are comparable
        results = evaluate(self.sac, self.final_stage, self.eval_episodes, seed=10_000)
        returns = [r["return"] for r in results]
        success = np.mean([r["outcome"] == "timeout" for r in results])
        crash = np.mean([r["outcome"] == "crash" for r in results])
        self.eval_csv.writerow([self.steps, self.stage, f"{np.mean(returns):.3f}", f"{np.std(returns):.3f}",
                                f"{np.mean([r['length'] for r in results]):.1f}", f"{success:.3f}", f"{crash:.3f}",
                                f"{np.mean([r['forward_speed'] for r in results]):.3f}"])
        self.eval_file.flush()
        self.episodes_file.flush()
        recent = [e["return"] for e in self.env.episodes[-20:]]
        elapsed = time.time() - self.start_time
        print(f"[{self.steps}/{self.total}] stage {self.stage} | train return (last 20) {np.mean(recent) if recent else 0:.1f}"
              f" | eval return {np.mean(returns):.1f}, success {success:.0%} | {self.steps / elapsed:.0f} steps/s",
              flush=True)

    def _on_training_end(self):
        for f in (self.episodes_file, self.eval_file, self.curriculum_file):
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name, output goes to runs/<name>/")
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--final-stage", type=int, default=2, help="curriculum stops here (2 = light traffic)")
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy coefficient")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    args = parser.parse_args()

    run_dir = f"runs/{args.name}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump({"args": vars(args), "env_config": CONFIG, "stages": STAGES, "advance_length": ADVANCE_LENGTH,
                   "advance_episodes": ADVANCE_EPISODES, "easy_episode_prob": EASY_EPISODE_PROB,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2, default=str)

    wrapper = make_env(stage=args.start_stage)
    env = EpisodeLogger(wrapper)
    env.unwrapped_wrapper = wrapper  # used by the curriculum to switch stages
    env.reset(seed=args.seed)

    model = SAC("MlpPolicy", env, learning_rate=args.lr, batch_size=args.batch_size,
                learning_starts=args.learning_starts, ent_coef=args.alpha, seed=args.seed)
    callback = TrainingCallback(model, env, run_dir, args.start_stage, args.final_stage, args.eval_freq,
                                args.eval_episodes, args.save_freq, args.timesteps)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(f"{run_dir}/final_model.pkl")
    env.close()
    print(f"Done, model saved to {run_dir}/final_model.pkl")
