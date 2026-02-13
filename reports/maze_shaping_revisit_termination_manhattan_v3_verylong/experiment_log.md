# Frontmatter
- Log file: `reports/maze_shaping_revisit_termination_manhattan_v3_verylong/experiment_log.md`
- Script: `experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py`
- Output roots:
  - `outputs/maze_shaping_revisit_termination_manhattan_v3_verylong`

# Goal
- 초장기 학습(very long train)에서 재수렴 여부 확인.

# Setup Snapshot
## `outputs/maze_shaping_revisit_termination_manhattan_v3_verylong`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `termination`: `episode ends on goal or revisiting any previously visited state in the same episode`
- `potential_distance`: `manhattan`
- `episodes`: `20000`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `0.99`
- `step_reward`: `0.0`
- `goal_reward`: `1.0`
- `validation_interval`: `25`
- `validation_episodes`: `30`

# Results Snapshot
## `outputs/maze_shaping_revisit_termination_manhattan_v3_verylong`
- Final training (last episode):
  - `no_shaping`: mean_steps=`1.6666666666666667` std=`1.178511301977579` (episode `20000`)
  - `phi_full`: mean_steps=`6.083333333333333` std=`5.314419587834173` (episode `20000`)
  - `phi_half`: mean_steps=`8.666666666666666` std=`3.5901098714230026` (episode `20000`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`1.0` (episode `20000`)
  - `phi_full`: success=`0.0` val_steps=`12.583333333333334` (episode `20000`)
  - `phi_half`: success=`0.0` val_steps=`19.0` (episode `20000`)
- Best validation success:
  - `no_shaping`: success=`0.0` val_steps=`1.0` (episode `13350`)
  - `phi_full`: success=`0.0` val_steps=`10.666666666666666` (episode `13350`)
  - `phi_half`: success=`0.0` val_steps=`14.916666666666666` (episode `13350`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
