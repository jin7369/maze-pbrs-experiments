# Frontmatter
- Log file: `reports/maze_shaping_icml_style_manhattan_v1/experiment_log.md`
- Script: `experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py`
- Output roots:
  - `outputs/maze_shaping_icml_style_manhattan_v1`

# Goal
- Manhattan potential 기반 ICML-style 3조건 비교.

# Setup Snapshot
## `outputs/maze_shaping_icml_style_manhattan_v1`
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
## `outputs/maze_shaping_icml_style_manhattan_v1`
- Final training (last episode):
  - `no_shaping`: mean_steps=`350.0` std=`0.0` (episode `500`)
  - `phi_full`: mean_steps=`242.66666666666666` std=`110.84323264061827` (episode `500`)
  - `phi_half`: mean_steps=`316.5833333333333` std=`64.82342469269028` (episode `500`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `500`)
  - `phi_full`: success=`0.04722222222222222` val_steps=`345.0805555555556` (episode `500`)
  - `phi_half`: success=`0.0` val_steps=`350.0` (episode `500`)
- Best validation success:
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `350`)
  - `phi_full`: success=`0.0861111111111111` val_steps=`340.9611111111111` (episode `450`)
  - `phi_half`: success=`0.0` val_steps=`350.0` (episode `350`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
