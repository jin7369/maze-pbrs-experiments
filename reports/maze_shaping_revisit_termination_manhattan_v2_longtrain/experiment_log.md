# Frontmatter
- Log file: `reports/maze_shaping_revisit_termination_manhattan_v2_longtrain/experiment_log.md`
- Script: `experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py`
- Output roots:
  - `outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain`

# Goal
- v1 대비 episodes를 늘린 장기 학습 재검증.

# Setup Snapshot
## `outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `termination`: `episode ends on goal or revisiting any previously visited state in the same episode`
- `potential_distance`: `manhattan`
- `episodes`: `5000`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `0.99`
- `step_reward`: `0.0`
- `goal_reward`: `1.0`
- `validation_interval`: `25`
- `validation_episodes`: `30`

# Results Snapshot
## `outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain`
- Final training (last episode):
  - `no_shaping`: mean_steps=`1.9166666666666667` std=`0.9537935951882998` (episode `5000`)
  - `phi_full`: mean_steps=`5.333333333333333` std=`2.7182510717166815` (episode `5000`)
  - `phi_half`: mean_steps=`9.166666666666666` std=`4.687453703475077` (episode `5000`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`1.0` (episode `5000`)
  - `phi_full`: success=`0.0` val_steps=`8.416666666666666` (episode `5000`)
  - `phi_half`: success=`0.0` val_steps=`10.083333333333334` (episode `5000`)
- Best validation success:
  - `no_shaping`: success=`0.0` val_steps=`1.0` (episode `3350`)
  - `phi_full`: success=`0.0` val_steps=`7.083333333333333` (episode `3350`)
  - `phi_half`: success=`0.0` val_steps=`8.25` (episode `3350`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
