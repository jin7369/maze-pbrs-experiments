import argparse
import csv
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mazelib import Maze
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator


def shortest_path_length(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    h, w = grid.shape
    q = deque([(start[0], start[1], 0)])
    seen = {start}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        r, c, d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))

    return -1


def dead_end_count(grid: np.ndarray) -> int:
    h, w = grid.shape
    cnt = 0
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != 0:
                continue
            n_open = 0
            for dr, dc in dirs:
                if grid[r + dr, c + dc] == 0:
                    n_open += 1
            if n_open == 1:
                cnt += 1
    return cnt


def generate_one(cells_h: int, cells_w: int, seed: int):
    np.random.seed(seed)
    maze = Maze(seed=seed)
    maze.generator = BacktrackingGenerator(cells_h, cells_w)
    maze.generate()

    grid = np.array(maze.grid, dtype=np.int8)
    start = (1, 1)
    goal = (grid.shape[0] - 2, grid.shape[1] - 2)

    grid[start] = 0
    grid[goal] = 0

    return grid, start, goal


def save_image(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], path: Path):
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[grid == 1] = [0.08, 0.08, 0.08]
    img[grid == 0] = [0.97, 0.97, 0.97]
    img[start] = [0.20, 0.75, 0.20]
    img[goal] = [0.85, 0.20, 0.20]

    plt.figure(figsize=(5, 5))
    plt.imshow(img, interpolation="nearest")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_contact_sheet(entries: list[dict], out_path: Path):
    n = len(entries)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue

        e = entries[i]
        grid = np.load(e["grid_npy"])
        ax.imshow(grid, cmap="gray_r", interpolation="nearest")
        ax.set_title(f"id={e['maze_id']} seed={e['seed']}\\nL={e['shortest_path_len']}", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate and visualize maze samples with mazelib.")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--cells-h", type=int, default=10)
    parser.add_argument("--cells-w", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("신규_실험_workspace/outputs/maze_samples_v1"))
    args = parser.parse_args()

    outdir = args.outdir
    img_dir = outdir / "images"
    npy_dir = outdir / "grids"
    outdir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(args.num_samples):
        seed = args.seed_start + i
        grid, start, goal = generate_one(args.cells_h, args.cells_w, seed)

        shortest = shortest_path_length(grid, start, goal)
        walls = int((grid == 1).sum())
        opens = int((grid == 0).sum())
        total = grid.size

        maze_id = f"maze_{i:02d}"
        img_path = img_dir / f"{maze_id}.png"
        npy_path = npy_dir / f"{maze_id}.npy"

        np.save(npy_path, grid)
        save_image(grid, start, goal, img_path)

        entries.append(
            {
                "maze_id": maze_id,
                "seed": seed,
                "cells_h": args.cells_h,
                "cells_w": args.cells_w,
                "grid_h": int(grid.shape[0]),
                "grid_w": int(grid.shape[1]),
                "start": [int(start[0]), int(start[1])],
                "goal": [int(goal[0]), int(goal[1])],
                "wall_count": walls,
                "open_count": opens,
                "wall_ratio": walls / total,
                "open_ratio": opens / total,
                "shortest_path_len": int(shortest),
                "dead_end_count": int(dead_end_count(grid)),
                "image_path": str(img_path.as_posix()),
                "grid_npy": str(npy_path.as_posix()),
            }
        )

    csv_path = outdir / "maze_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        writer.writeheader()
        writer.writerows(entries)

    json_path = outdir / "maze_metadata.json"
    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    contact_path = outdir / "maze_contact_sheet.png"
    save_contact_sheet(entries, contact_path)

    print(f"saved {args.num_samples} mazes to: {outdir}")
    print(f"images: {img_dir}")
    print(f"metadata: {csv_path} , {json_path}")
    print(f"contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
