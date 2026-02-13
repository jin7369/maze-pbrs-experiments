# Frontmatter
- Log file: `reports/maze_shaping_pbrs_manhattan_explore_v1/experiment_log.md`
- Script: `experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py`
- Output roots:
  - `outputs/maze_shaping_icml_style_pbrs_manhattan_explore_v1`

# Goal
- Manhattan PBRS에 first-visit exploration bonus를 추가한 실험.

# Setup Snapshot
## `outputs/maze_shaping_icml_style_pbrs_manhattan_explore_v1`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `potential_distance`: `manhattan`
- `episodes`: `500`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `0.99`
- `max_steps`: `350`
- `step_reward`: `0.0`
- `goal_reward`: `1.0`
- `exploration_bonus`: `0.05`
- `validation_interval`: `25`
- `validation_episodes`: `30`

# Results Snapshot
## `outputs/maze_shaping_icml_style_pbrs_manhattan_explore_v1`
- Final training (last episode):
  - `no_shaping`: mean_steps=`350.0` std=`0.0` (episode `500`)
  - `phi_full`: mean_steps=`350.0` std=`0.0` (episode `500`)
  - `phi_half`: mean_steps=`350.0` std=`0.0` (episode `500`)
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
