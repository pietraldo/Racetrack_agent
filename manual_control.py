import gymnasium as gym
import highway_env  # registers racetrack envs
import numpy as np

# UP/DOWN = accelerate/brake, LEFT/RIGHT = steer (handled by highway-env's pygame viewer)
env = gym.make(
    "racetrack-large-v1",
    render_mode="human",
    config={
        "manual_control": True,
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
        },
    },
)
obs, info = env.reset()

while True:
    # With manual_control=True the passed action is ignored; keyboard events set it.
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    if terminated or truncated:
        obs, info = env.reset()
    # The viewer is set to None by env.close(), which happens when the window is closed
    if env.unwrapped.viewer is None:
        break

env.close()
