import numpy as np


class ReplayBuffer:
    def __init__(self, 
        capacity: int, 
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ) -> None:
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.obs = np.empty((capacity, *observation_shape))
        self.actions = np.empty((capacity, *action_shape))
        self.rewards = np.empty(capacity)
        self.next_obs = np.empty((capacity, *observation_shape))
        self.dones = np.empty(capacity)

        self.index = 0
        self.full = False

    def size(self):
        return self.capacity if self.full else self.index
    
    def append(
        self,
        obervation: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_obervation: np.ndarray,
        done: bool,
    ):
        self.obs[self.index] = obervation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obervation
        self.dones[self.index] = done

        if self.index + 1 >= self.capacity:
            self.full = True
        self.index = (self.index + 1) % self.capacity         

    def extend(
        self,
        obs: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        next_obs: np.ndarray,
        dones: np.ndarray,
    ):
        assert obs.shape[0] == actions.shape[0] == rewards.shape[0] == \
            next_obs.shape[0] == dones.shape[0]

        num_append = self.capacity - self.index
        num_overflow = obs.shape[0] - num_append

        self.obs[self.index:self.index+num_append] = obs[:num_append]
        self.obs[:num_overflow] = obs[num_append:]

        self.actions[self.index:self.index+num_append] = actions[:num_append]
        self.actions[:num_overflow] = actions[num_append:]

        self.rewards[self.index:self.index+num_append] = rewards[:num_append]
        self.rewards[:num_overflow] = rewards[num_append:]

        self.next_obs[self.index:self.index+num_append] = next_obs[:num_append]
        self.next_obs[:num_overflow] = next_obs[num_append:]

        self.dones[self.index:self.index+num_append] = dones[:num_append]
        self.dones[:num_overflow] = dones[num_append:]

        if self.index + obs.shape[0] >= self.capacity:
            self.full = True
        self.index = (self.index + obs.shape[0]) % self.capacity 

    def sample(self, number):
        indices = np.random.choice(self.size(), size=number)

        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )