import os
import time

import highway_env  # registers racetrack envs
import pygame
from stable_baselines3 import SAC

from train_sac import STAGES, make_env, newest_checkpoint, read_stage

# Shows the newest checkpoint saved by train_sac.py driving on the track.
# Run it in a second terminal while training; before every episode it switches to a newer checkpoint if one exists.

MIN_FILE_AGE = 3  # seconds, newer files may still be being written by the training

env = make_env(render_mode="human")
env.easy_episode_prob = 0  # always show the current stage, easier episodes are only for training

print("Waiting for a checkpoint...")
while newest_checkpoint() is None:
    time.sleep(1)

model, loaded_file = None, None
model_version = 0  # how many times a newer model was loaded
attempt = 0  # episodes driven by the current model

while True:
    newest = newest_checkpoint()
    if newest != loaded_file and (model is None or time.time() - os.path.getmtime(newest) >= MIN_FILE_AGE):
        try:
            model = SAC.load(newest, device="cpu")
        except (OSError, ValueError, RuntimeError):  # file is still being written
            time.sleep(0.5)
            continue
        loaded_file = newest
        model_version += 1
        attempt = 0
        print(f"\nLoaded model #{model_version}: {newest} (trained for {model.num_timesteps} timesteps)")

    attempt += 1
    stage = read_stage()  # drive on the same stage the training is on
    env.set_stage(stage)
    obs, info = env.reset()
    episode_reward, steps, done = 0.0, 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated
        pygame.display.set_caption(
            f"stage {stage}: {STAGES[stage]['name']} | "
            f"model #{model_version} ({model.num_timesteps} steps) | attempt {attempt} | "
            f"step {steps} | reward {episode_reward:.1f} | speed {env.unwrapped.vehicle.speed:.1f}"
        )
        # The viewer is set to None by env.close(), which happens when the window is closed
        if env.unwrapped.viewer is None:
            raise SystemExit

    reason = "standing still" if info["idle"] else "crash/off road" if terminated else "time limit"
    print(f"model #{model_version} attempt {attempt}: reward {episode_reward:.1f}, {steps} steps, {reason}")
