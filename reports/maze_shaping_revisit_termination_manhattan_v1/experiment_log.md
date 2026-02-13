# Frontmatter
- Log file: `reports/maze_shaping_revisit_termination_manhattan_v1/experiment_log.md`
- Script: `experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py`
- Output roots:
  - `outputs/maze_shaping_revisit_termination_manhattan_v1`

# Goal
- 재방문 즉시 종료 규칙에서 Manhattan PBRS 성능 평가.

# Setup Snapshot
## `outputs/maze_shaping_revisit_termination_manhattan_v1`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `termination`: `episode ends on goal or revisiting any previously visited state in the same episode`
- `potential_distance`: `manhattan`
- `episodes`: `500`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `0.99`
- `step_reward`: `0.0`
- `goal_reward`: `1.0`
- `validation_interval`: `25`
- `validation_episodes`: `30`

# Results Snapshot
## `outputs/maze_shaping_revisit_termination_manhattan_v1`
- Final training (last episode):
  - `no_shaping`: mean_steps=`2.5` std=`1.118033988749895` (episode `500`)
  - `phi_full`: mean_steps=`2.3333333333333335` std=`0.8498365855987975` (episode `500`)
  - `phi_half`: mean_steps=`2.6666666666666665` std=`1.178511301977579` (episode `500`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`1.0` (episode `500`)
  - `phi_full`: success=`0.0` val_steps=`2.4166666666666665` (episode `500`)
  - `phi_half`: success=`0.0` val_steps=`2.6666666666666665` (episode `500`)
- Best validation success:
  - `no_shaping`: success=`0.0` val_steps=`1.0` (episode `350`)
  - `phi_full`: success=`0.0` val_steps=`1.75` (episode `350`)
  - `phi_half`: success=`0.0` val_steps=`2.3333333333333335` (episode `350`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
