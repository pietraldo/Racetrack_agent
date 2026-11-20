import argparse
import glob
import os
import time

import highway_env  # registers racetrack envs
import pygame

from softactorcritic import SAC
from train_sac import STAGES, make_env

# Shows a trained agent driving on the track.
# Usage: python watch_agent.py [run_dir] [--stage N] [--model file.pkl]
#   run_dir   defaults to the most recently modified runs/*/
#   --stage   traffic: 0 empty track, 1 one car, 2 light traffic (1-5 cars), 3 heavy traffic (10-20 cars);
#             default = the stage the training is/was on (run_dir/stage.txt)
#   --model   exact model file; default = run_dir/final_model.pkl if the training finished, otherwise the newest
#             checkpoint (during training it switches to newer checkpoints before every episode)

MIN_FILE_AGE = 3  # seconds, newer files may still be being written by the training

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", nargs="?")
parser.add_argument("--stage", type=int, choices=range(len(STAGES)))
parser.add_argument("--model")
args = parser.parse_args()

runs = [d for d in glob.glob("runs/*/") if os.path.exists(f"{d}/stage.txt")]
run_dir = args.run_dir or max(runs, key=os.path.getmtime)
fixed_model = args.model or (f"{run_dir}/final_model.pkl" if os.path.exists(f"{run_dir}/final_model.pkl") else None)
STAGE = args.stage
print(f"Run: {run_dir}")


def newest_checkpoint():
    if fixed_model:
        return fixed_model
    files = sorted(glob.glob(f"{run_dir}/checkpoints/model_*.pkl"))
    return files[-1] if files else None


def read_stage():
    if STAGE is not None:
        return STAGE
    try:
        with open(f"{run_dir}/stage.txt") as f:
            return int(f.read())
    except (OSError, ValueError):
        return 0


env = make_env(render_mode="human")
env.easy_episode_prob = 0  # always show the current stage, easier episodes are only for training

print("Waiting for a checkpoint...")
while newest_checkpoint() is None:
    time.sleep(1)

model, loaded_file = None, None
attempt = 0

while True:
    newest = newest_checkpoint()
    if newest != loaded_file and (model is None or time.time() - os.path.getmtime(newest) >= MIN_FILE_AGE):
        try:
            model = SAC.load(newest, env=env)
        except (OSError, EOFError, ValueError, RuntimeError):  # file is still being written
            time.sleep(0.5)
            continue
        loaded_file = newest
        attempt = 0
        print(f"\nLoaded {newest}")

    attempt += 1
    stage = read_stage()
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
            f"stage {stage}: {STAGES[stage]['name']} | {os.path.basename(loaded_file)} | attempt {attempt} | "
            f"step {steps} | reward {episode_reward:.1f} | speed {env.unwrapped.vehicle.speed:.1f}"
        )
        # The viewer is set to None by env.close(), which happens when the window is closed
        if env.unwrapped.viewer is None:
            raise SystemExit

    reason = "standing still" if info["idle"] else "crash/off road" if terminated else "time limit"
    print(f"attempt {attempt}: reward {episode_reward:.1f}, {steps} steps, {reason}")
