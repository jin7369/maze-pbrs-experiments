# Frontmatter
- Log file: `reports/maze_shaping_noslip_v1/experiment_log.md`
- Script: `experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py`
- Output roots:
  - `outputs/maze_shaping_icml_style_bfs_noslip_v1`
  - `outputs/maze_shaping_icml_style_manhattan_noslip_v1`

# Goal
- no-slip 환경에서 BFS/Manhattan 설정 비교.

# Setup Snapshot
## `outputs/maze_shaping_icml_style_bfs_noslip_v1`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `potential_distance`: `bfs`
- `episodes`: `500`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `1.0`
- `max_steps`: `350`
- `validation_interval`: `25`
- `validation_episodes`: `30`
## `outputs/maze_shaping_icml_style_manhattan_noslip_v1`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `potential_distance`: `manhattan`
- `episodes`: `500`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `1.0`
- `max_steps`: `350`
- `validation_interval`: `25`
- `validation_episodes`: `30`

# Results Snapshot
## `outputs/maze_shaping_icml_style_bfs_noslip_v1`
- Final training (last episode):
  - `no_shaping`: mean_steps=`342.25` std=`16.274340744456186` (episode `500`)
  - `phi_full`: mean_steps=`49.75` std=`3.031088913245535` (episode `500`)
  - `phi_half`: mean_steps=`207.33333333333334` std=`69.7583129255734` (episode `500`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `500`)
  - `phi_full`: success=`1.0` val_steps=`44.0` (episode `500`)
  - `phi_half`: success=`0.0` val_steps=`350.0` (episode `500`)
- Best validation success:
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `350`)
  - `phi_full`: success=`1.0` val_steps=`44.0` (episode `325`)
  - `phi_half`: success=`0.0` val_steps=`350.0` (episode `350`)
## `outputs/maze_shaping_icml_style_manhattan_noslip_v1`
- Final training (last episode):
  - `no_shaping`: mean_steps=`342.25` std=`16.274340744456186` (episode `500`)
  - `phi_full`: mean_steps=`153.25` std=`114.73384054700978` (episode `500`)
  - `phi_half`: mean_steps=`248.75` std=`80.90027297028526` (episode `500`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `500`)
  - `phi_full`: success=`0.0` val_steps=`350.0` (episode `500`)
  - `phi_half`: success=`0.0` val_steps=`350.0` (episode `500`)
- Best validation success:
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `350`)
  - `phi_full`: success=`0.0` val_steps=`350.0` (episode `350`)
  - `phi_half`: success=`0.0` val_steps=`350.0` (episode `350`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
