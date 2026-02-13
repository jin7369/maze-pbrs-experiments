# Frontmatter
- Source: `신규_실험_workspace/reinforce_memory_env/reports/reinforce_memory_v1/report_reinforce_memory_v1.tex`
- Output root: `신규_실험_workspace/reinforce_memory_env/outputs/reinforce_memory_v1`
- Script: `신규_실험_workspace/reinforce_memory_env/experiments/run_reinforce_memory_experiment.py`

# Goal
- visited-table 메모리 상태를 포함한 REINFORCE 정책이 미로를 학습 가능한지 검증.

# Hypothesis
- Inferred: `concat(position one-hot, visited-table flatten)` 인코딩이 탐색 효율을 높여 validation 성공률을 개선한다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- deterministic transition, max_steps `350`
- step reward `-0.01`, goal reward `1.0`

# Algorithm
- `REINFORCE (policy gradient, numpy MLP)`
- state_encoding: `concat(position one-hot, visited-table flatten)`

# Reward
- Environment reward only (`-0.01` per step, `+1.0` at goal)
- potential-based shaping 없음.
- Leakage Risk: none/low (oracle/heuristic potential 미사용)

# Hyperparameters
- episodes `800`, runs `8`
- hidden_dim `64`, lr `0.002`
- gamma `0.99`, entropy_coef `0.001`
- validation_interval `25`, validation_episodes `30`

# Metrics & Artifacts
- `신규_실험_workspace/reinforce_memory_env/outputs/reinforce_memory_v1/learning_curve.csv`
- `신규_실험_workspace/reinforce_memory_env/outputs/reinforce_memory_v1/learning_curve.png`
- `신규_실험_workspace/reinforce_memory_env/outputs/reinforce_memory_v1/validation_progress.csv`
- `신규_실험_workspace/reinforce_memory_env/outputs/reinforce_memory_v1/validation_progress.png`
- `신규_실험_workspace/reinforce_memory_env/outputs/reinforce_memory_v1/run_summary.json`

# Results
- Final train mean steps (ep 800): `350.0`
- Final validation success rate (ep 800): `0.0`
- Final validation mean steps (ep 800): `350.0`
- Best validation success rate (all checkpoints): `0.0`

# Anomalies / Notes
- 학습/검증 모두 상한(`350`)에 고정되어 완전 실패.
- 메모리 인코딩 추가 효과가 관측되지 않음.

# Interpretation (tentative)
- 현재 보상/탐색/분산 설정에서 REINFORCE가 신호를 학습하지 못함.

# Next Actions
1. Actor-Critic(A2C)로 분산 감소.
2. reward scale 및 gamma sweep.
3. entropy_coef 상향과 curriculum maze 도입.

# Questions to resolve
- 실패 원인이 sparse reward인가 policy-gradient variance인가?
- visited-table 특징이 실제로 정책에 사용되는가?

