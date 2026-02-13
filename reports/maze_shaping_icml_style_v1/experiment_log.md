# Frontmatter
- Log file: `reports/maze_shaping_icml_style_v1/experiment_log.md`
- Script: `experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py`
- Output roots:
  - `outputs/maze_shaping_icml_style_v1`

# Goal
- ICML-style 3 conditions(no_shaping, phi_half, phi_full) 비교.

# Setup Snapshot
## `outputs/maze_shaping_icml_style_v1`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `episodes`: `500`
- `runs`: `12`
- `alpha`: `0.02`
- `epsilon`: `0.1`
- `gamma`: `1.0`
- `max_steps`: `350`
- `validation_interval`: `25`
- `validation_episodes`: `30`

# Results Snapshot
## `outputs/maze_shaping_icml_style_v1`
- Final training (last episode):
  - `no_shaping`: mean_steps=`350.0` std=`0.0` (episode `500`)
  - `phi_full`: mean_steps=`99.25` std=`29.24073927474019` (episode `500`)
  - `phi_half`: mean_steps=`268.4166666666667` std=`80.99018287559439` (episode `500`)
- Final validation (last checkpoint):
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `500`)
  - `phi_full`: success=`0.6305555555555555` val_steps=`229.25833333333333` (episode `500`)
  - `phi_half`: success=`0.005555555555555556` val_steps=`349.1222222222222` (episode `500`)
- Best validation success:
  - `no_shaping`: success=`0.0` val_steps=`350.0` (episode `350`)
  - `phi_full`: success=`0.9166666666666666` val_steps=`168.8361111111111` (episode `50`)
  - `phi_half`: success=`0.030555555555555555` val_steps=`347.7472222222222` (episode `475`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
