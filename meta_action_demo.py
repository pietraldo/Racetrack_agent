import gymnasium as gym
import highway_env  # registers racetrack envs
import numpy as np
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.vehicle.controller import MDPVehicle


# racetrack creates the ego vehicle with vehicle_class.make_on_lane, but DiscreteMetaAction
# returns a functools.partial without that method. A real subclass fixes it.
class RacetrackMDPVehicle(MDPVehicle):
    DEFAULT_TARGET_SPEEDS = np.array([0.0, 5.0, 10.0])


DiscreteMetaAction.vehicle_class = property(lambda self: RacetrackMDPVehicle)

# Actions: 0 = LANE_LEFT, 1 = IDLE, 2 = LANE_RIGHT, 3 = FASTER, 4 = SLOWER
env = gym.make("racetrack-v1", render_mode="human", config={
    "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10]},
    "other_vehicles": 3,
    "real_time_rendering": True,
})
print(env.action_space, env.unwrapped.action_type.actions)

for episode in range(3):
    env.reset(seed=episode)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
env.close()
