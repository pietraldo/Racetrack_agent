import argparse
import glob
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from softactorcritic import SAC
from train_sac import EPISODE_STEPS, MAX_SPEED, STAGES, episode_outcome, make_env

# Evaluation methodology:
#   - every agent drives EPISODES episodes in each environment setting (traffic density = curriculum stage),
#   - deterministic policy (mean action, no exploration noise),
#   - the same seeds for every agent, so all agents see the same starting positions and traffic,
#   - metrics: return, episode length, % episodes finished without crash (survived 60 s),
#     crash / off-road rate, mean speed along the track, distance driven.
# Output: results/eval_results.csv (every episode), results/eval_summary.csv/.md, figures/eval_*.png
#
# Usage: python evaluate.py                       # final models of all runs + random baseline
#        python evaluate.py --models runs/A_curriculum_a0.2/final_model.pkl

SEED = 123_456
COLORS = ["#8a8984", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]  # first = random baseline


def run_job(job):
    """One (agent, setting) pair; runs in a separate process."""
    name, model_path, stage, episodes = job
    torch.set_num_threads(1)
    if model_path is None:
        policy = lambda obs, env: env.action_space.sample()
    else:
        model = SAC.load(model_path, env=make_env())
        policy = lambda obs, env: model.predict(obs, deterministic=True)[0]
    rows = [{"agent": name, **r} for r in run_episodes(policy, stage, episodes)]
    print(f"{name:28s} {STAGES[stage]['name']:14s} return {np.mean([r['return'] for r in rows]):7.1f}"
          f"  success {np.mean([r['outcome'] == 'timeout' for r in rows]):4.0%}", flush=True)
    return rows


def run_episodes(policy, stage, episodes):
    env = make_env(stage=stage)
    env.easy_episode_prob = 0
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=SEED + i)
        ret, length, speed, done = 0.0, 0, 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs, env))
            ret += reward
            length += 1
            speed += info["forward_speed"] * MAX_SPEED
            done = terminated or truncated
        rows.append({"stage": stage, "setting": STAGES[stage]["name"], "episode": i, "return": ret, "length": length,
                     "outcome": episode_outcome(info, terminated), "forward_speed": speed / length,
                     "distance": speed / env.unwrapped.config["policy_frequency"]})
    env.close()
    return rows


def summarize(df):
    g = df.groupby(["agent", "stage", "setting"], sort=False)
    return pd.DataFrame({
        "return_mean": g["return"].mean(), "return_std": g["return"].std(),
        "length_mean": g["length"].mean(),
        "success_%": g["outcome"].apply(lambda o: (o == "timeout").mean() * 100),
        "crash_%": g["outcome"].apply(lambda o: (o == "crash").mean() * 100),
        "idle_%": g["outcome"].apply(lambda o: (o == "idle").mean() * 100),
        "speed_mps": g["forward_speed"].mean(), "distance_m": g["distance"].mean(),
    }).reset_index()


def bar_chart(summary, column, ylabel, fname):
    agents = list(dict.fromkeys(summary["agent"]))
    settings = list(dict.fromkeys(summary["setting"]))
    width = 0.8 / len(agents)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, agent in enumerate(agents):
        s = summary[summary["agent"] == agent].set_index("setting").reindex(settings)
        x = np.arange(len(settings)) + (i - (len(agents) - 1) / 2) * width
        ax.bar(x, s[column], width * 0.92, color=COLORS[i % len(COLORS)], label=agent)
    ax.set_xticks(np.arange(len(settings)), settings)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/{fname}", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="model .pkl files (default: runs/*/final_model.pkl)")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--stages", type=int, nargs="*", default=[0, 1, 2, 3])
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    agents = {"random": None}
    for path in args.models or sorted(glob.glob("runs/*/final_model.pkl")):
        agents[os.path.basename(os.path.dirname(path))] = path
    jobs = [(name, path, stage, args.episodes) for name, path in agents.items() for stage in args.stages]
    with Pool(min(len(jobs), os.cpu_count() - 1)) as pool:
        rows = [row for job_rows in pool.map(run_job, jobs) for row in job_rows]

    df = pd.DataFrame(rows)
    df.to_csv("results/eval_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv("results/eval_summary.csv", index=False)
    with open("results/eval_summary.md", "w") as f:
        f.write(f"{args.episodes} episodes per setting, deterministic policy, max episode length {EPISODE_STEPS}\n\n")
        cols = list(summary.columns)
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, row in summary.iterrows():
            f.write("| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in row) + " |\n")
    print(summary.to_string(index=False, float_format="%.1f"))

    bar_chart(summary, "return_mean", "mean episode return", "eval_return.png")
    bar_chart(summary, "success_%", "% episodes without crash (60 s)", "eval_success.png")
    bar_chart(summary, "distance_m", "mean distance driven [m]", "eval_distance.png")
