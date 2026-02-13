import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


ACTIONS = np.array([
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
], dtype=np.int32)


@dataclass(frozen=True)
class GridWorldConfig:
    size: int
    step_reward: float = -1.0
    success_prob: float = 0.8


class GridWorld:
    def __init__(self, cfg: GridWorldConfig):
        self.cfg = cfg
        self.start = (0, 0)
        self.goal = (cfg.size - 1, cfg.size - 1)

    def reset(self):
        return self.start

    def is_terminal(self, state):
        return state == self.goal

    def _clip(self, r, c):
        n = self.cfg.size
        return max(0, min(n - 1, r)), max(0, min(n - 1, c))

    def step(self, state, action, rng):
        if self.is_terminal(state):
            return state, 0.0, True

        if rng.random() < self.cfg.success_prob:
            chosen_action = action
        else:
            chosen_action = int(rng.integers(0, 4))

        dr, dc = ACTIONS[chosen_action]
        nr, nc = self._clip(state[0] + dr, state[1] + dc)
        next_state = (nr, nc)

        done = self.is_terminal(next_state)
        reward = self.cfg.step_reward
        return next_state, reward, done


def manhattan_phi(size, scale=1.0):
    goal = (size - 1, size - 1)
    phi = np.zeros((size, size), dtype=np.float64)
    for r in range(size):
        for c in range(size):
            dist = abs(goal[0] - r) + abs(goal[1] - c)
            phi[r, c] = scale * (-dist / 0.8)
    return phi


def epsilon_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, 4))
    max_val = np.max(q_row)
    ties = np.flatnonzero(np.isclose(q_row, max_val))
    return int(rng.choice(ties))


def sarsa_train(
    env,
    phi,
    episodes,
    alpha,
    epsilon,
    gamma,
    max_steps,
    seed,
):
    rng = np.random.default_rng(seed)
    n = env.cfg.size
    q = np.zeros((n, n, 4), dtype=np.float64)
    steps_to_goal = np.zeros(episodes, dtype=np.int32)

    phi_goal = phi[env.goal]
    phi = phi.copy()
    phi[env.goal] = phi_goal

    for ep in range(episodes):
        s = env.reset()
        a = epsilon_greedy(q[s[0], s[1]], epsilon, rng)

        for t in range(1, max_steps + 1):
            s2, base_r, done = env.step(s, a, rng)
            shaped = base_r + gamma * phi[s2[0], s2[1]] - phi[s[0], s[1]]

            if done:
                td_target = shaped
                q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])
                steps_to_goal[ep] = t
                break

            a2 = epsilon_greedy(q[s2[0], s2[1]], epsilon, rng)
            td_target = shaped + gamma * q[s2[0], s2[1], a2]
            q[s[0], s[1], a] += alpha * (td_target - q[s[0], s[1], a])

            s, a = s2, a2
        else:
            steps_to_goal[ep] = max_steps

    return steps_to_goal


def run_condition(size, phi_scale, episodes, runs, alpha, epsilon, gamma, max_steps, seed):
    env = GridWorld(GridWorldConfig(size=size))
    if phi_scale == 0.0:
        phi = np.zeros((size, size), dtype=np.float64)
    else:
        phi = manhattan_phi(size=size, scale=phi_scale)

    all_runs = []
    for i in trange(runs, desc=f"size={size}, phi={phi_scale}", leave=False):
        steps = sarsa_train(
            env=env,
            phi=phi,
            episodes=episodes,
            alpha=alpha,
            epsilon=epsilon,
            gamma=gamma,
            max_steps=max_steps,
            seed=seed + i,
        )
        all_runs.append(steps)

    arr = np.stack(all_runs, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_results(episodes, results, title, out_path):
    x = np.arange(1, episodes + 1)
    plt.figure(figsize=(9, 5))
    for label, (mean, std) in results.items():
        plt.plot(x, mean, label=label, linewidth=1.8)
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)

    plt.xlabel("Trial number")
    plt.ylabel("Steps taken to reach goal")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Reproduce potential-based shaping experiments.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 50])
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        max_steps = max(200, size * size * 2)
        results = {}

        mean0, std0 = run_condition(size=size, phi_scale=0.0, episodes=args.episodes, runs=args.runs,
                                    alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma,
                                    max_steps=max_steps, seed=args.seed)
        results["no shaping"] = (mean0, std0)

        mean_half, std_half = run_condition(size=size, phi_scale=0.5, episodes=args.episodes, runs=args.runs,
                                            alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma,
                                            max_steps=max_steps, seed=args.seed + 1000)
        results["phi = 0.5 * phi0"] = (mean_half, std_half)

        mean_full, std_full = run_condition(size=size, phi_scale=1.0, episodes=args.episodes, runs=args.runs,
                                            alpha=args.alpha, epsilon=args.epsilon, gamma=args.gamma,
                                            max_steps=max_steps, seed=args.seed + 2000)
        results["phi = phi0"] = (mean_full, std_full)

        csv_path = args.outdir / f"grid_{size}_learning_curves.csv"
        data = np.column_stack([
            np.arange(1, args.episodes + 1),
            mean0, std0,
            mean_half, std_half,
            mean_full, std_full,
        ])
        header = "trial,mean_no,std_no,mean_half,std_half,mean_full,std_full"
        np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

        fig_path = args.outdir / f"grid_{size}_learning_curves.png"
        plot_results(
            episodes=args.episodes,
            results=results,
            title=f"Potential-based shaping on {size}x{size} grid-world",
            out_path=fig_path,
        )

        print(f"[done] size={size} -> {csv_path} , {fig_path}")


if __name__ == "__main__":
    main()
