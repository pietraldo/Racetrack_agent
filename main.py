import gymnasium
import highway_env
import numpy as np
import pygame

# Arrow keys: UP/DOWN = accelerate/brake, LEFT/RIGHT = steer
env = gymnasium.make('racetrack-v0', render_mode='human')
obs, info = env.reset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    acceleration = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
    steering = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
    action = np.array([acceleration, steering], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()
