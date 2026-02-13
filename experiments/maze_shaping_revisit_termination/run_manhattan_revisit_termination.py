import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ACTIONS = np.array([
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
], dtype=np.int32)


@dataclass
class EnvConfig:
    step_reward: float = 0.0
    goal_reward: float = 1.0


class MazeEnv:
    def __init__(self, grid: np.ndarray, start=(1, 1), goal=None, cfg: EnvConfig = EnvConfig()):
        self.grid = grid
        self.h, self.w = grid.shape
        self.start = start
        self.goal = goal if goal is not None else (self.h - 2, self.w - 2)
        self.cfg = cfg

    def reset(self):
        return self.start

    def is_open(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w and self.grid[r, c] == 0

    def step(self, state, action):
        r, c = state
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc

        if not self.is_open(nr, nc):
            nr, nc = r, c

        next_state = (nr, nc)
        done_goal = next_state == self.goal
        reward = self.cfg.goal_reward if done_goal else self.cfg.step_reward
        return next_state, reward, done_goal


def manhattan_distance_map(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    rr, cc = np.indices((h, w))
    return np.abs(rr - goal[0]) + np.abs(cc - goal[1])


def epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, 4))
    m = np.max(q_row)
    ties = np.flatnonzero(np.isclose(q_row, m))
    return int(rng.choice(ties))


def greedy_action(q_row: np.ndarray) -> int:
    return int(np.argmax(q_row))


def run_train_episode(env: MazeEnv, q: np.ndarray, phi: np.ndarray, alpha: float, epsilon: float, gamma: float, rng):
    s = env.reset()
    a = epsilon_greedy(q[s[0], s[1]], epsilon, rng)
    visited = {s}
    steps = 0

    while True:
        steps += 1
        s2, base_r, done_goal = env.step(s, a)

        done_revisit = (s2 in visited) and (not done_goal)
        done = done_goal or done_revisit

        shaped_r = base_r + gamma * phi[s2] - phi[s]

        if done:
            td_target = shaped_r
            q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])
            return steps, done_goal

        visited.add(s2)
        a2 = epsilon_greedy(q[s2[0], s2[1]], epsilon, rng)
        td_target = shaped_r + gamma * q[s2[0], s2[1], a2]
        q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])
        s, a = s2, a2


def run_eval_episode(env: MazeEnv, q: np.ndarray):
    s = env.reset()
    visited = {s}
    steps = 0

    while True:
        steps += 1
        a = greedy_action(q[s[0], s[1]])
        s2, _, done_goal = env.step(s, a)

        done_revisit = (s2 in visited) and (not done_goal)
        done = done_goal or done_revisit

        if done:
            return steps, done_goal

        visited.add(s2)
        s = s2


def train_and_validate(
    env: MazeEnv,
    phi: np.ndarray,
    episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    validation_interval: int,
    validation_episodes: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.h, env.w, 4), dtype=np.float64)

    train_steps = np.zeros(episodes, dtype=np.int32)
    val_records = []

    for ep in range(episodes):
        steps, success = run_train_episode(env, q, phi, alpha, epsilon, gamma, rng)
        train_steps[ep] = steps

        if (ep + 1) % validation_interval == 0 or ep == 0:
            succ = 0
            total_steps = 0
            for _ in range(validation_episodes):
                s_steps, s_ok = run_eval_episode(env, q)
                succ += int(s_ok)
                total_steps += s_steps
            val_records.append(
                {
                    "episode": ep + 1,
                    "success_rate": succ / validation_episodes,
                    "avg_steps": total_steps / validation_episodes,
                }
            )

    return q, train_steps, pd.DataFrame(val_records)


def run_experiment(
    maze_path: Path,
    outdir: Path,
    episodes: int,
    runs: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    validation_interval: int,
    validation_episodes: int,
    step_reward: float,
    goal_reward: float,
    seed: int,
):
    grid = np.load(maze_path)
    env = MazeEnv(grid, cfg=EnvConfig(step_reward=step_reward, goal_reward=goal_reward))

    dist = manhattan_distance_map(grid, env.goal).astype(np.float64)
    phi0 = -dist

    scales = [0.0, 0.5, 1.0]
    labels = {0.0: "no_shaping", 0.5: "phi_half", 1.0: "phi_full"}

    learning_rows = []
    validation_rows = []

    for scale in scales:
        phi = scale * phi0
        all_steps = []
        all_val = []

        for run_idx in range(runs):
            q, steps, val_df = train_and_validate(
                env=env,
                phi=phi,
                episodes=episodes,
                alpha=alpha,
                epsilon=epsilon,
                gamma=gamma,
                validation_interval=validation_interval,
                validation_episodes=validation_episodes,
                seed=seed + run_idx + int(scale * 1000),
            )
            all_steps.append(steps)
            val_df["run"] = run_idx
            all_val.append(val_df)

        arr = np.stack(all_steps, axis=0)
        mean_steps = arr.mean(axis=0)
        std_steps = arr.std(axis=0)

        for ep in range(episodes):
            learning_rows.append(
                {
                    "condition": labels[scale],
                    "episode": ep + 1,
                    "mean_steps": float(mean_steps[ep]),
                    "std_steps": float(std_steps[ep]),
                }
            )

        val_cat = pd.concat(all_val, ignore_index=True)
        val_group = val_cat.groupby("episode", as_index=False).agg(
            mean_success_rate=("success_rate", "mean"),
            std_success_rate=("success_rate", "std"),
            mean_val_steps=("avg_steps", "mean"),
            std_val_steps=("avg_steps", "std"),
        )
        val_group["std_success_rate"] = val_group["std_success_rate"].fillna(0.0)
        val_group["std_val_steps"] = val_group["std_val_steps"].fillna(0.0)

        for _, r in val_group.iterrows():
            validation_rows.append(
                {
                    "condition": labels[scale],
                    "episode": int(r["episode"]),
                    "mean_success_rate": float(r["mean_success_rate"]),
                    "std_success_rate": float(r["std_success_rate"]),
                    "mean_val_steps": float(r["mean_val_steps"]),
                    "std_val_steps": float(r["std_val_steps"]),
                }
            )

    learn_df = pd.DataFrame(learning_rows)
    val_df = pd.DataFrame(validation_rows)
    learn_df.to_csv(outdir / "learning_curve.csv", index=False)
    val_df.to_csv(outdir / "validation_progress.csv", index=False)

    plot_learning_curve(learn_df, outdir / "learning_curve.png")
    plot_validation(val_df, outdir / "validation_progress.png")


def plot_learning_curve(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(9, 5))
    for cond, g in df.groupby("condition"):
        x = g["episode"].values
        y = g["mean_steps"].values
        s = g["std_steps"].values
        plt.plot(x, y, label=cond)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel("Steps to termination")
    plt.title("Manhattan PBRS with revisit-termination")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_validation(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for cond, g in df.groupby("condition"):
        x = g["episode"].values
        y1 = g["mean_success_rate"].values
        s1 = g["std_success_rate"].values
        y2 = g["mean_val_steps"].values
        s2 = g["std_val_steps"].values

        axes[0].plot(x, y1, label=cond)
        axes[0].fill_between(x, np.clip(y1 - s1, 0, 1), np.clip(y1 + s1, 0, 1), alpha=0.15)

        axes[1].plot(x, y2, label=cond)
        axes[1].fill_between(x, y2 - s2, y2 + s2, alpha=0.15)

    axes[0].set_title("Validation success rate")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Validation steps to termination")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg steps")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Manhattan PBRS experiment with revisit-termination.")
    parser.add_argument("--maze-path", type=Path, default=Path("신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy"))
    parser.add_argument("--outdir", type=Path, default=Path("신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v1"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--step-reward", type=float, default=0.0)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        maze_path=args.maze_path,
        outdir=outdir,
        episodes=args.episodes,
        runs=args.runs,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        step_reward=args.step_reward,
        goal_reward=args.goal_reward,
        seed=args.seed,
    )

    summary = {
        "maze_path": str(args.maze_path.as_posix()),
        "termination": "episode ends on goal or revisiting any previously visited state in the same episode",
        "potential_distance": "manhattan",
        "episodes": args.episodes,
        "runs": args.runs,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "step_reward": args.step_reward,
        "goal_reward": args.goal_reward,
        "validation_interval": args.validation_interval,
        "validation_episodes": args.validation_episodes,
        "outputs": {
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_png": "learning_curve.png",
            "validation_csv": "validation_progress.csv",
            "validation_png": "validation_progress.png",
        },
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] results saved to {outdir}")


if __name__ == "__main__":
    main()
