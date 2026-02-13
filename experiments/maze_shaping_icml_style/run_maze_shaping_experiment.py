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

    def step(self, state, action, rng):
        r, c = state
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc

        if not self.is_open(nr, nc):
            nr, nc = r, c

        next_state = (nr, nc)
        done = next_state == self.goal
        reward = self.cfg.goal_reward if done else self.cfg.step_reward
        return next_state, reward, done


def bfs_distance_map(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), np.inf, dtype=np.float64)
    if grid[goal] == 1:
        return dist

    q = [goal]
    dist[goal] = 0.0
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        d = dist[r, c]
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and dist[nr, nc] == np.inf:
                dist[nr, nc] = d + 1.0
                q.append((nr, nc))
    return dist


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


def train_sarsa(
    env: MazeEnv,
    phi: np.ndarray,
    episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    max_steps: int,
    seed: int,
    validation_interval: int,
    validation_episodes: int,
    exploration_bonus: float,
):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.h, env.w, 4), dtype=np.float64)

    train_steps = np.zeros(episodes, dtype=np.int32)
    val_records = []

    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(q[s[0], s[1]], epsilon, rng)
        visit_count = np.zeros((env.h, env.w), dtype=np.int32)
        visit_count[s] += 1

        for t in range(1, max_steps + 1):
            s2, base_r, done = env.step(s, a, rng)
            # Potential-based exploration shaping with visit-count potential:
            # Psi(s) = -N(s), where N(s) is in-episode visit count.
            psi_s = -float(visit_count[s])
            psi_s2 = -float(visit_count[s2])
            explore_shaping = exploration_bonus * (gamma * psi_s2 - psi_s)
            shaped_r = base_r + gamma * phi[s2] - phi[s] + explore_shaping
            visit_count[s2] += 1

            if done:
                td_target = shaped_r
                q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])
                train_steps[ep] = t
                break

            a2 = epsilon_greedy(q[s2[0], s2[1]], epsilon, rng)
            td_target = shaped_r + gamma * q[s2[0], s2[1], a2]
            q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])
            s, a = s2, a2
        else:
            train_steps[ep] = max_steps

        if (ep + 1) % validation_interval == 0 or ep == 0:
            success, avg_steps = validate_policy(env, q, validation_episodes, max_steps, rng)
            val_records.append(
                {
                    "episode": ep + 1,
                    "success_rate": success,
                    "avg_steps": avg_steps,
                }
            )

    return q, train_steps, pd.DataFrame(val_records)


def validate_policy(env: MazeEnv, q: np.ndarray, n_episodes: int, max_steps: int, rng: np.random.Generator):
    successes = 0
    total_steps = 0

    for _ in range(n_episodes):
        s = env.reset()
        for t in range(1, max_steps + 1):
            a = greedy_action(q[s[0], s[1]])
            s, _, done = env.step(s, a, rng)
            if done:
                successes += 1
                total_steps += t
                break
        else:
            total_steps += max_steps

    return successes / n_episodes, total_steps / n_episodes


def run_experiment_on_maze(
    grid: np.ndarray,
    outdir: Path,
    episodes: int,
    runs: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    max_steps: int,
    validation_interval: int,
    validation_episodes: int,
    seed: int,
    potential_distance: str,
    step_reward: float,
    goal_reward: float,
    exploration_bonus: float,
):
    env = MazeEnv(grid, cfg=EnvConfig(step_reward=step_reward, goal_reward=goal_reward))
    if potential_distance == "bfs":
        dist = bfs_distance_map(grid, env.goal)
        finite = np.isfinite(dist)
        max_dist = np.max(dist[finite]) if np.any(finite) else 1.0
        dist = np.where(finite, dist, max_dist)
    elif potential_distance == "manhattan":
        dist = manhattan_distance_map(grid, env.goal).astype(np.float64)
    else:
        raise ValueError(f"Unknown potential_distance: {potential_distance}")

    phi0 = -dist

    scales = [0.0, 0.5, 1.0]
    labels = {
        0.0: "no_shaping",
        0.5: "phi_half",
        1.0: "phi_full",
    }

    learning_rows = []
    validation_rows = []

    for scale in scales:
        phi = scale * phi0
        all_steps = []
        all_val = []

        for run_idx in range(runs):
            q, train_steps, val_df = train_sarsa(
                env=env,
                phi=phi,
                episodes=episodes,
                alpha=alpha,
                epsilon=epsilon,
                gamma=gamma,
                max_steps=max_steps,
                seed=seed + int(scale * 1000) + run_idx,
                validation_interval=validation_interval,
                validation_episodes=validation_episodes,
                exploration_bonus=exploration_bonus,
            )
            all_steps.append(train_steps)
            val_df["run"] = run_idx
            all_val.append(val_df)

        steps_arr = np.stack(all_steps, axis=0)
        mean_steps = steps_arr.mean(axis=0)
        std_steps = steps_arr.std(axis=0)

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

    return phi0


def plot_learning_curve(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(9, 5))
    for cond, g in df.groupby("condition"):
        x = g["episode"].values
        y = g["mean_steps"].values
        s = g["std_steps"].values
        plt.plot(x, y, label=cond)
        plt.fill_between(x, y - s, y + s, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel("Steps to goal (train)")
    plt.title("Maze shaping experiment (ICML-style conditions)")
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

    axes[1].set_title("Validation steps")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg steps")
    axes[1].grid(alpha=0.25)

    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def render_frame(grid: np.ndarray, path: list[tuple[int, int]], agent: tuple[int, int], goal: tuple[int, int], scale: int = 18):
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[grid == 1] = np.array([20, 20, 20], dtype=np.uint8)
    img[grid == 0] = np.array([245, 245, 245], dtype=np.uint8)

    for pr, pc in path:
        if grid[pr, pc] == 0:
            img[pr, pc] = np.array([150, 200, 255], dtype=np.uint8)

    gr, gc = goal
    ar, ac = agent
    img[gr, gc] = np.array([220, 60, 60], dtype=np.uint8)
    img[ar, ac] = np.array([40, 170, 60], dtype=np.uint8)

    big = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
    return big


def rollout_greedy_trajectory(env: MazeEnv, q: np.ndarray, max_steps: int, rng: np.random.Generator):
    s = env.reset()
    traj = [s]
    for _ in range(max_steps):
        a = greedy_action(q[s[0], s[1]])
        s, _, done = env.step(s, a, rng)
        traj.append(s)
        if done:
            break
    return traj


def make_checkpoint_gifs(
    grid: np.ndarray,
    phi0: np.ndarray,
    outdir: Path,
    episodes: int,
    alpha: float,
    epsilon: float,
    gamma: float,
    max_steps: int,
    seed: int,
    step_reward: float,
    goal_reward: float,
    exploration_bonus: float,
):
    env = MazeEnv(grid, cfg=EnvConfig(step_reward=step_reward, goal_reward=goal_reward))
    rng = np.random.default_rng(seed)

    checkpoints = [0, episodes // 4, episodes // 2, (3 * episodes) // 4, episodes]
    checkpoints = sorted(set(checkpoints))

    q = np.zeros((env.h, env.w, 4), dtype=np.float64)
    snapshots = {0: q.copy()}

    phi = phi0
    for ep in range(1, episodes + 1):
        s = env.reset()
        a = epsilon_greedy(q[s[0], s[1]], epsilon, rng)
        visit_count = np.zeros((env.h, env.w), dtype=np.int32)
        visit_count[s] += 1

        for _ in range(max_steps):
            s2, base_r, done = env.step(s, a, rng)
            psi_s = -float(visit_count[s])
            psi_s2 = -float(visit_count[s2])
            explore_shaping = exploration_bonus * (gamma * psi_s2 - psi_s)
            shaped_r = base_r + gamma * phi[s2] - phi[s] + explore_shaping
            visit_count[s2] += 1

            if done:
                td_target = shaped_r
                q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])
                break

            a2 = epsilon_greedy(q[s2[0], s2[1]], epsilon, rng)
            td_target = shaped_r + gamma * q[s2[0], s2[1], a2]
            q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])
            s, a = s2, a2

        if ep in checkpoints:
            snapshots[ep] = q.copy()

    gif_dir = outdir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in checkpoints:
        q_ckpt = snapshots[ckpt]
        traj = rollout_greedy_trajectory(env, q_ckpt, max_steps=max_steps, rng=np.random.default_rng(seed + ckpt + 99))

        frames = []
        path = []
        for s in traj:
            path.append(s)
            frame = render_frame(grid, path, s, env.goal)
            frames.append(Image.fromarray(frame))

        gif_path = gif_dir / f"policy_rollout_ep_{ckpt:04d}.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)


def main():
    parser = argparse.ArgumentParser(description="Run maze shaping experiments and export validation + GIF samples.")
    parser.add_argument("--maze-path", type=Path, default=Path("신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy"))
    parser.add_argument("--outdir", type=Path, default=Path("신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_v2"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--step-reward", type=float, default=0.0)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--exploration-bonus", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--potential-distance", choices=["bfs", "manhattan"], default="manhattan")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    grid = np.load(args.maze_path)

    phi0 = run_experiment_on_maze(
        grid=grid,
        outdir=outdir,
        episodes=args.episodes,
        runs=args.runs,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        max_steps=args.max_steps,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        seed=args.seed,
        potential_distance=args.potential_distance,
        step_reward=args.step_reward,
        goal_reward=args.goal_reward,
        exploration_bonus=args.exploration_bonus,
    )

    make_checkpoint_gifs(
        grid=grid,
        phi0=phi0,
        outdir=outdir,
        episodes=args.episodes,
        alpha=args.alpha,
        epsilon=args.epsilon,
        gamma=args.gamma,
        max_steps=args.max_steps,
        seed=args.seed + 123,
        step_reward=args.step_reward,
        goal_reward=args.goal_reward,
        exploration_bonus=args.exploration_bonus,
    )

    summary = {
        "maze_path": str(args.maze_path.as_posix()),
        "episodes": args.episodes,
        "runs": args.runs,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "step_reward": args.step_reward,
        "goal_reward": args.goal_reward,
        "exploration_bonus": args.exploration_bonus,
        "max_steps": args.max_steps,
        "validation_interval": args.validation_interval,
        "validation_episodes": args.validation_episodes,
        "potential_distance": args.potential_distance,
        "outputs": {
            "learning_curve_csv": "learning_curve.csv",
            "learning_curve_png": "learning_curve.png",
            "validation_csv": "validation_progress.csv",
            "validation_png": "validation_progress.png",
            "gif_dir": "gifs",
        },
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[done] results saved to {outdir}")


if __name__ == "__main__":
    main()
