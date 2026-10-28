import os

import gymnasium as gym
import highway_env  # registers racetrack envs
import numpy as np
from PIL import Image

# Random agent on racetrack-v1 with continuous throttle + steering.
# Records a GIF of a few episodes and prints baseline statistics.

CONFIG = {
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
    },
    "other_vehicles": 3,
}
VIDEO_EPISODES = 8
STATS_EPISODES = 100
OUT_DIR = "media"


def run_episode(env, seed, record=False):
    obs, info = env.reset(seed=seed)
    frames, ret, steps = [], 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ret += reward
        steps += 1
        if record:
            frames.append(env.render())
        if terminated or truncated:
            v = env.unwrapped.vehicle
            if v.crashed:
                outcome = "collision"
            elif not v.on_road:
                outcome = "off_road"
            else:
                outcome = "timeout"
            return frames, ret, steps, outcome


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = gym.make("racetrack-v1", render_mode="rgb_array", config=CONFIG)
    env.action_space.seed(0)

    frames = []
    for ep in range(VIDEO_EPISODES):
        ep_frames, ret, steps, outcome = run_episode(env, seed=ep, record=True)
        print(f"video episode {ep}: return={ret:.2f} steps={steps} outcome={outcome}")
        frames += ep_frames + [ep_frames[-1]] * 5  # short pause on episode end
    images = [Image.fromarray(f) for f in frames]
    images[0].save(os.path.join(OUT_DIR, "random_agent.gif"), save_all=True,
                   append_images=images[1:], duration=200, loop=0)

    returns, lengths, outcomes = [], [], []
    for ep in range(STATS_EPISODES):
        _, ret, steps, outcome = run_episode(env, seed=1000 + ep)
        returns.append(ret)
        lengths.append(steps)
        outcomes.append(outcome)
    env.close()

    policy_freq = env.unwrapped.config["policy_frequency"]
    print(f"\nRandom agent, {STATS_EPISODES} episodes:")
    print(f"  return        mean={np.mean(returns):.2f} std={np.std(returns):.2f}")
    print(f"  length [steps] mean={np.mean(lengths):.1f} std={np.std(lengths):.1f}"
          f"  (~{np.mean(lengths) / policy_freq:.1f} s)")
    for name in ("collision", "off_road", "timeout"):
        print(f"  {name:13s} {outcomes.count(name) / STATS_EPISODES:.0%}")


if __name__ == "__main__":
    main()
