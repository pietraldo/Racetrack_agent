import gymnasium
import highway_env

import pprint

from highway_env.vehicle.kinematics import Vehicle

# Wyzsza maksymalna predkosc (domyslnie 40 m/s)
Vehicle.MAX_SPEED = 150

# Sterowanie klawiszami (jedno nacisniecie = jedna akcja):
#   UP / DOWN    = zmiana pasa w lewo / w prawo (o jeden pas)
#   RIGHT / LEFT = przyspieszenie / hamowanie (+/- 5 m/s docelowej predkosci)
env = gymnasium.make('highway-v0', render_mode='human', config={
    'manual_control': True,
    'lanes_count': 5,
    'other_vehicles': 10,
    'vehicles_density': 1,
    'screen_height': 800,
    'screen_width': 1200,
    'action': {'type': 'DiscreteMetaAction'},
    # Bez limitu czasu (domyslnie 40 s) - epizod konczy sie tylko kolizja
    'duration': 100000,
})
pprint.pprint(env.unwrapped.config)
obs, info = env.reset()

running = True
while running:
    # Przy manual_control akcja z env.step jest ignorowana - klawisze obsluguje viewer
    obs, reward, terminated, truncated, info = env.step(1)  # 1 = IDLE
    # render() sam czyta zdarzenia pygame (klawisze); nie wolno ich pobierac wczesniej
    env.render()

    if env.unwrapped.viewer is None:  # okno zamkniete -> env.close()
        running = False
    elif terminated or truncated:
        obs, info = env.reset()
