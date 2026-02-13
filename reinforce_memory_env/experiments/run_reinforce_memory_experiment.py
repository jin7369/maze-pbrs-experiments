import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ACTIONS = np.array([
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
], dtype=np.int32)


@dataclass
class EnvConfig:
    step_reward: float = -0.01
    goal_reward: float = 1.0
    slip_prob: float = 0.0


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
        if self.cfg.slip_prob > 0 and rng.random() < self.cfg.slip_prob:
            action = int(rng.integers(0, 4))

        r, c = state
        dr, dc = ACTIONS[action]
        nr, nc = r + dr, c + dc

        if not self.is_open(nr, nc):
            nr, nc = r, c

        next_state = (nr, nc)
        done = next_state == self.goal
        reward = self.cfg.goal_reward if done else self.cfg.step_reward
        return next_state, reward, done


class PolicyMLP:
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.05, size=(input_dim, hidden_dim)).astype(np.float64)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0, 0.05, size=(hidden_dim, action_dim)).astype(np.float64)
        self.b2 = np.zeros(action_dim, dtype=np.float64)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        logits = h @ self.W2 + self.b2
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)
        return h, probs

    def sample_action(self, x, rng):
        _, probs = self.forward(x)
        a = int(rng.choice(len(probs), p=probs))
        return a, probs

    def greedy_action(self, x):
        _, probs = self.forward(x)
        return int(np.argmax(probs))

    def apply_reinforce_update(self, traj, gamma: float, lr: float, entropy_coef: float):
        # traj entries: (x, h, probs, action, reward)
        T = len(traj)
        returns = np.zeros(T, dtype=np.float64)
        g = 0.0
        for t in reversed(range(T)):
            g = traj[t][4] + gamma * g
            returns[t] = g

        if np.std(returns) > 1e-8:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        for t, (x, h, probs, action, _) in enumerate(traj):
            Gt = returns[t]
            y = np.zeros_like(probs)
            y[action] = 1.0

            dlogits = (probs - y) * Gt

            if entropy_coef > 0:
                # Add entropy regularization gradient for policy smoothness.
                safe_probs = np.clip(probs, 1e-8, 1.0)
                entropy_grad = safe_probs * (np.log(safe_probs) + 1.0)
                dlogits += entropy_coef * entropy_grad

            dW2 += np.outer(h, dlogits)
            db2 += dlogits

            dh = self.W2 @ dlogits
            dz1 = dh * (1.0 - h * h)

            dW1 += np.outer(x, dz1)
            db1 += dz1

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


def state_feature(state, visited, h, w):
    pos = np.zeros(h * w, dtype=np.float64)
    idx = state[0] * w + state[1]
    pos[idx] = 1.0
    vis = visited.astype(np.float64).reshape(-1)
    return np.concatenate([pos, vis], axis=0)


def run_episode(env, policy, gamma, lr, entropy_coef, max_steps, rng, train=True):
    s = env.reset()
    visited = np.zeros((env.h, env.w), dtype=np.int8)
    visited[s] = 1

    traj = []
    path = [s]

    for t in range(1, max_steps + 1):
        x = state_feature(s, visited, env.h, env.w)

        if train:
            a, probs = policy.sample_action(x, rng)
        else:
            a = policy.greedy_action(x)
            _, probs = policy.forward(x)

        h_act, _ = policy.forward(x)
        s2, r, done = env.step(s, a, rng)
        visited[s2] = 1
        path.append(s2)

        if train:
            traj.append((x, h_act, probs, a, r))

        s = s2
        if done:
            if train and len(traj) > 0:
                policy.apply_reinforce_update(traj, gamma=gamma, lr=lr, entropy_coef=entropy_coef)
            return t, True, path

    if train and len(traj) > 0:
        policy.apply_reinforce_update(traj, gamma=gamma, lr=lr, entropy_coef=entropy_coef)
    return max_steps, False, path


def evaluate_policy(env, policy, eval_episodes, max_steps, seed):
    rng = np.random.default_rng(seed)
    succ = 0
    total_steps = 0
    for _ in range(eval_episodes):
        steps, ok, _ = run_episode(env, policy, gamma=1.0, lr=0.0, entropy_coef=0.0, max_steps=max_steps, rng=rng, train=False)
        succ += int(ok)
        total_steps += steps
    return succ / eval_episodes, total_steps / eval_episodes


def render_frame(grid, path, agent, goal, scale=18):
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[grid == 1] = [20, 20, 20]
    img[grid == 0] = [245, 245, 245]

    for pr, pc in path:
        if grid[pr, pc] == 0:
            img[pr, pc] = [150, 200, 255]

    ar, ac = agent
    gr, gc = goal
    img[ar, ac] = [40, 170, 60]
    img[gr, gc] = [220, 60, 60]

    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))


def save_rollout_gif(env, policy, out_path, max_steps, seed):
    rng = np.random.default_rng(seed)
    steps, ok, rollout = run_episode(env, policy, gamma=1.0, lr=0.0, entropy_coef=0.0, max_steps=max_steps, rng=rng, train=False)

    frames = []
    walked = []
    for s in rollout:
        walked.append(s)
        frame = render_frame(env.grid, walked, s, env.goal)
        frames.append(Image.fromarray(frame))

    if frames:
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    return steps, ok


def plot_curves(train_df, val_df, outdir):
    plt.figure(figsize=(9, 5))
    plt.plot(train_df['episode'], train_df['mean_steps'], label='train mean steps')
    plt.fill_between(train_df['episode'], train_df['mean_steps'] - train_df['std_steps'], train_df['mean_steps'] + train_df['std_steps'], alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('REINFORCE training curve')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'learning_curve.png', dpi=170)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(val_df['episode'], val_df['mean_success_rate'], color='tab:blue', label='success rate')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success rate', color='tab:blue')
    ax1.set_ylim(0.0, 1.02)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(val_df['episode'], val_df['mean_val_steps'], color='tab:orange', label='val steps')
    ax2.set_ylabel('Avg steps', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('REINFORCE validation progress')
    fig.tight_layout()
    fig.savefig(outdir / 'validation_progress.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='REINFORCE maze experiment with visited-table state encoding.')
    parser.add_argument('--maze-path', type=Path, default=Path('신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy'))
    parser.add_argument('--outdir', type=Path, default=Path('신규_실험_workspace/reinforce_memory_env/outputs/reinforce_memory_v1'))
    parser.add_argument('--episodes', type=int, default=800)
    parser.add_argument('--runs', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--max-steps', type=int, default=350)
    parser.add_argument('--validation-interval', type=int, default=25)
    parser.add_argument('--validation-episodes', type=int, default=30)
    parser.add_argument('--step-reward', type=float, default=-0.01)
    parser.add_argument('--goal-reward', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=13)
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    grid = np.load(args.maze_path)
    env = MazeEnv(grid, cfg=EnvConfig(step_reward=args.step_reward, goal_reward=args.goal_reward, slip_prob=0.0))

    input_dim = env.h * env.w * 2

    all_train = []
    all_val = []
    snapshots = []

    for run in range(args.runs):
        policy = PolicyMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, action_dim=4, seed=args.seed + run)
        rng = np.random.default_rng(args.seed + run)

        train_steps = []
        val_rows = []

        for ep in range(1, args.episodes + 1):
            steps, _, _ = run_episode(env, policy, gamma=args.gamma, lr=args.lr, entropy_coef=args.entropy_coef, max_steps=args.max_steps, rng=rng, train=True)
            train_steps.append(steps)

            if ep == 1 or ep % args.validation_interval == 0:
                sr, vs = evaluate_policy(env, policy, eval_episodes=args.validation_episodes, max_steps=args.max_steps, seed=args.seed + 10000 + run + ep)
                val_rows.append({'episode': ep, 'success_rate': sr, 'avg_steps': vs, 'run': run})

            if run == 0 and ep in [1, args.episodes // 4, args.episodes // 2, (3 * args.episodes) // 4, args.episodes]:
                snapshots.append((ep, policy.W1.copy(), policy.b1.copy(), policy.W2.copy(), policy.b2.copy()))

        all_train.append(np.array(train_steps, dtype=np.float64))
        all_val.append(val_rows)

    train_arr = np.stack(all_train, axis=0)
    train_df = {
        'episode': np.arange(1, args.episodes + 1),
        'mean_steps': train_arr.mean(axis=0),
        'std_steps': train_arr.std(axis=0),
    }
    import pandas as pd
    train_df = pd.DataFrame(train_df)
    train_df.to_csv(outdir / 'learning_curve.csv', index=False)

    val_records = []
    for rows in all_val:
        val_records.extend(rows)
    val_raw = pd.DataFrame(val_records)
    val_df = val_raw.groupby('episode', as_index=False).agg(
        mean_success_rate=('success_rate', 'mean'),
        std_success_rate=('success_rate', 'std'),
        mean_val_steps=('avg_steps', 'mean'),
        std_val_steps=('avg_steps', 'std'),
    )
    val_df['std_success_rate'] = val_df['std_success_rate'].fillna(0.0)
    val_df['std_val_steps'] = val_df['std_val_steps'].fillna(0.0)
    val_df.to_csv(outdir / 'validation_progress.csv', index=False)

    plot_curves(train_df, val_df, outdir)

    gif_dir = outdir / 'gifs'
    gif_dir.mkdir(parents=True, exist_ok=True)
    for ep, W1, b1, W2, b2 in snapshots:
        policy = PolicyMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, action_dim=4, seed=0)
        policy.W1, policy.b1, policy.W2, policy.b2 = W1, b1, W2, b2
        save_rollout_gif(env, policy, gif_dir / f'policy_rollout_ep_{ep:04d}.gif', max_steps=args.max_steps, seed=args.seed + ep)

    summary = {
        'algorithm': 'REINFORCE (policy gradient, numpy MLP)',
        'state_encoding': 'concat(position one-hot, visited-table flatten)',
        'maze_path': str(args.maze_path.as_posix()),
        'episodes': args.episodes,
        'runs': args.runs,
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'gamma': args.gamma,
        'entropy_coef': args.entropy_coef,
        'max_steps': args.max_steps,
        'validation_interval': args.validation_interval,
        'validation_episodes': args.validation_episodes,
        'step_reward': args.step_reward,
        'goal_reward': args.goal_reward,
        'outputs': {
            'learning_curve_csv': 'learning_curve.csv',
            'learning_curve_png': 'learning_curve.png',
            'validation_csv': 'validation_progress.csv',
            'validation_png': 'validation_progress.png',
            'gif_dir': 'gifs',
        },
    }
    (outdir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'[done] results saved to {outdir}')


if __name__ == '__main__':
    main()
