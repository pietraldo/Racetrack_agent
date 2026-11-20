import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Learning curves of runs saved by train_sac.py.
# Usage: python plot_training.py [run_dir ...]   (default: every runs/*/ with episodes.csv)
# Output: figures/learning_<run>.png (one per run) and figures/compare_*.png (all runs together)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300"]
STAGE_COLORS = ["#f2f2f0", "#e6eef9", "#fbe9e1", "#e3f4ec"]  # background of each curriculum stage
STAGE_NAMES = ["empty track", "one car", "light traffic", "heavy traffic"]
SMOOTH = 30  # episodes in the rolling mean of training curves
OUT = "figures"

plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "lines.linewidth": 2,
})


def label(run_dir):
    return os.path.basename(os.path.normpath(run_dir))


def shade_stages(ax, run_dir, end):
    cur = pd.read_csv(f"{run_dir}/curriculum.csv")
    starts = list(cur["timesteps"]) + [end]
    for i, row in cur.iterrows():
        ax.axvspan(starts[i], starts[i + 1], color=STAGE_COLORS[row["stage"]], zorder=0, lw=0)
        ax.text(starts[i], 1.0, f" {row['name']}", transform=ax.get_xaxis_transform(), va="top", fontsize=8,
                color="#52514e")


def plot_run(run_dir):
    ep = pd.read_csv(f"{run_dir}/episodes.csv")
    ev = pd.read_csv(f"{run_dir}/eval.csv")
    end = max(ep["timesteps"].max(), ev["timesteps"].max())
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    ax = axes[0]
    shade_stages(ax, run_dir, end)
    ax.plot(ep["timesteps"], ep["return"], color=COLORS[0], alpha=0.15, lw=1)
    ax.plot(ep["timesteps"], ep["return"].rolling(SMOOTH, min_periods=1).mean(), color=COLORS[0],
            label=f"training episodes (mean of last {SMOOTH})")
    ax.plot(ev["timesteps"], ev["return_mean"], color=COLORS[1], marker="o", ms=4,
            label="evaluation on light traffic (deterministic, 10 ep.)")
    ax.set_ylabel("episode return")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    ax = axes[1]
    shade_stages(ax, run_dir, end)
    ax.plot(ep["timesteps"], ep["length"].rolling(SMOOTH, min_periods=1).mean(), color=COLORS[0], label="training")
    ax.plot(ev["timesteps"], ev["length_mean"], color=COLORS[1], marker="o", ms=4, label="evaluation")
    ax.axhline(300, color="#8a8984", lw=1, ls="--")
    ax.set_ylabel("episode length [steps]\n(300 = full 60 s)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    ax = axes[2]
    shade_stages(ax, run_dir, end)
    crash = (ep["outcome"] == "crash").astype(float).rolling(SMOOTH, min_periods=1).mean()
    ax.plot(ep["timesteps"], crash * 100, color=COLORS[0], label="training: crash / off-road rate")
    ax.plot(ev["timesteps"], ev["success_rate"] * 100, color=COLORS[1], marker="o", ms=4,
            label="evaluation: episodes finished without crash")
    ax.set_ylabel("%")
    ax.set_ylim(-3, 103)
    ax.set_xlabel("environment steps")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle(f"SAC (own implementation) - {label(run_dir)}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/learning_{label(run_dir)}.png")
    plt.close(fig)


def plot_compare(run_dirs):
    evs = {label(r): pd.read_csv(f"{r}/eval.csv") for r in run_dirs}
    eps = {label(r): pd.read_csv(f"{r}/episodes.csv") for r in run_dirs}
    panels = [
        ("eval_return", "evaluation return (light traffic)", lambda n: (evs[n]["timesteps"], evs[n]["return_mean"])),
        ("eval_success", "evaluation: % episodes without crash", lambda n: (evs[n]["timesteps"], evs[n]["success_rate"] * 100)),
        ("train_return", f"training return (mean of last {SMOOTH} ep.)",
         lambda n: (eps[n]["timesteps"], eps[n]["return"].rolling(SMOOTH, min_periods=1).mean())),
        ("eval_speed", "evaluation: mean speed along the track [m/s]", lambda n: (evs[n]["timesteps"], evs[n]["forward_speed"])),
    ]
    for key, title, data in panels:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, name in enumerate(evs):
            x, y = data(name)
            ax.plot(x, y, color=COLORS[i % len(COLORS)], label=name, marker="o" if key.startswith("eval") else None,
                    ms=3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/compare_{key}.png")
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    runs = sys.argv[1:] or sorted(os.path.dirname(p) for p in glob.glob("runs/*/episodes.csv"))
    for r in runs:
        plot_run(r)
    plot_compare(runs)
    print(f"Saved plots for {len(runs)} runs to {OUT}/")
