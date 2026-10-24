import gymnasium
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.widgets import Button, Slider


def browse(fig, count, update):
    # Adds a slider, < > buttons and left/right keys that call update(i)
    fig.subplots_adjust(bottom=0.2)
    slider = Slider(fig.add_axes([0.15, 0.08, 0.7, 0.03]), "Step", 0, count - 1, valinit=0, valstep=1)
    prev_btn = Button(fig.add_axes([0.05, 0.08, 0.06, 0.04]), "<")
    next_btn = Button(fig.add_axes([0.89, 0.08, 0.06, 0.04]), ">")

    def on_slide(val):
        update(int(round(val)))
        fig.canvas.draw_idle()

    def step(delta):
        slider.set_val(int(np.clip(slider.val + delta, 0, count - 1)))

    def on_key(event):
        if event.key == "left":
            step(-1)
        elif event.key == "right":
            step(1)

    slider.on_changed(on_slide)
    prev_btn.on_clicked(lambda _: step(-1))
    next_btn.on_clicked(lambda _: step(1))
    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)
    return slider, prev_btn, next_btn


def show_frames(frames):
    if not frames:
        print("No frames recorded.")
        return []

    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    ax.axis("off")
    title = ax.set_title("")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"Frame {i + 1}/{len(frames)}")

    return browse(fig, len(frames), update)


def show_obs(observations):
    # obs has shape (2, 12, 12): channel 0 = presence (vehicles), channel 1 = on_road
    if not observations:
        print("No observations recorded.")
        return []

    fig, (ax_presence, ax_road) = plt.subplots(1, 2)
    names = ["presence", "on_road"]
    images = []
    for ax, name, channel in zip((ax_presence, ax_road), names, range(2)):
        im = ax.imshow(observations[0][channel], cmap="gray", vmin=0, vmax=1)
        n = observations[0].shape[1]
        ax.plot(n // 2, n // 2, "r+", markersize=12)  # ego vehicle sits in the centre
        ax.set_title(name)
        images.append(im)

    suptitle = fig.suptitle("")

    def update(i):
        for channel, im in enumerate(images):
            im.set_data(observations[i][channel])
        suptitle.set_text(f"Observation at step {i + 1}/{len(observations)}")

    return browse(fig, len(observations), update)


# Arrow keys: UP/DOWN = accelerate/brake, LEFT/RIGHT = steer
env = gymnasium.make('racetrack-v1', render_mode='rgb_array')
obs, info = env.reset()


plt_array=[]
obs_array=[]

running = True
while running:
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         running = False

    # keys = pygame.key.get_pressed()
    # acceleration = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
    # steering = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
    # action = np.array([acceleration, steering], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(np.array([0]))
    plt_array.append(env.render())
    obs_array.append(obs.copy())

    if terminated or truncated:
        obs, info = env.reset()
        break
env.close()

# Keep widget references alive so the sliders and buttons stay interactive
widgets = show_frames(plt_array) + show_obs(obs_array)
plt.show()
print(obs_array[0])

