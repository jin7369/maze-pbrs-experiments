# Frontmatter
- Log file: `reinforce_memory_env/reports/reinforce_memory_v1/experiment_log.md`
- Script: `reinforce_memory_env/experiments/run_reinforce_memory_experiment.py`
- Output roots:
  - `reinforce_memory_env/outputs/reinforce_memory_v1`

# Goal
- visited-table state encoding 기반 REINFORCE 성능 확인.

# Setup Snapshot
## `reinforce_memory_env/outputs/reinforce_memory_v1`
- `maze_path`: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- `episodes`: `800`
- `runs`: `8`
- `gamma`: `0.99`
- `max_steps`: `350`
- `step_reward`: `-0.01`
- `goal_reward`: `1.0`
- `validation_interval`: `25`
- `validation_episodes`: `30`
- `algorithm`: `REINFORCE (policy gradient, numpy MLP)`
- `state_encoding`: `concat(position one-hot, visited-table flatten)`
- `hidden_dim`: `64`
- `lr`: `0.002`
- `entropy_coef`: `0.001`

# Results Snapshot
## `reinforce_memory_env/outputs/reinforce_memory_v1`
- Final training: mean_steps=`350.0` std=`0.0` (episode `800`)
- Final validation: success=`0.0` val_steps=`350.0` (episode `800`)
- Best validation success: success=`0.0` val_steps=`350.0` (episode `550`)

# Notes
- This file was normalized to UTF-8 to fix broken text rendering.
- For full narrative, see the paired report `.tex/.pdf` in the same folder.
