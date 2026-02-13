# Frontmatter
- Log file: `reports/maze_shaping_revisit_penalty_threshold_manhattan_v1/experiment_log.md`
- Script: `experiments/maze_shaping_revisit_termination/run_manhattan_revisit_penalty_threshold.py`
- Output roots:
  - `outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1`

# Goal
- 재방문 패널티와 임계치 종료를 적용한 Manhattan PBRS 실험.

# Setup Snapshot
## `outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `termination`: `episode ends on goal or when revisit count in same episode reaches revisit_terminate_count`
- `potential_distance`: `manhattan`
- `episodes`: `5000`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `0.99`
- `step_reward`: `0.0`
- `goal_reward`: `1.0`
- `revisit_penalty`: `-0.05`
- `revisit_terminate_count`: `3`
- `validation_interval`: `25`
- `validation_episodes`: `30`

# Results Snapshot
## `outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1`
- Final training (last episode):
  - `no_shaping`: mean_steps=`25.166666666666668` std=`16.435902436096683` (episode `5000`)
  - `phi_full`: mean_steps=`11.333333333333334` std=`6.00462784488394` (episode `5000`)
  - `phi_half`: mean_steps=`13.166666666666666` std=`6.426939828219615` (episode `5000`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`47.583333333333336` (episode `5000`)
  - `phi_full`: success=`0.0` val_steps=`11.5` (episode `5000`)
  - `phi_half`: success=`0.0` val_steps=`15.333333333333334` (episode `5000`)
- Best validation success:
  - `no_shaping`: success=`0.4166666666666667` val_steps=`36.583333333333336` (episode `4850`)
  - `phi_full`: success=`0.0` val_steps=`10.833333333333334` (episode `3350`)
  - `phi_half`: success=`0.0` val_steps=`15.333333333333334` (episode `3350`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
