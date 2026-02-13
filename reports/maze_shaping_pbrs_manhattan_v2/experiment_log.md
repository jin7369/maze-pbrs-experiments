# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_pbrs_manhattan_v2/report_pbrs_manhattan_v2.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_v2`

# Goal
- Manhattan PBRS에 감가율(`gamma=0.99`)과 terminal `+1` 보상을 적용한 성능 평가.

# Hypothesis
- Inferred: step `0.0`, goal `1.0`, `gamma=0.99` 조합이 sparse reward를 완화해 학습을 촉진한다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- episodes `500`, runs `12`, max_steps `350`

# Algorithm
- Tabular SARSA.

# Reward
- `step_reward = 0.0`
- `goal_reward = 1.0`
- PBRS: `R'_k = R + gamma*Phi_k(s') - Phi_k(s)`
- `Phi(s)=-(|r-r_g|+|c-c_g|)`, `Phi_k(s)=k*Phi(s)`
- `gamma = 0.99`
- Leakage Risk: heuristic (Manhattan), 목표 좌표 정보 포함.

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `0.99`
- validation_interval `25`, validation_episodes `30`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_v2/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_v2/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_v2/run_summary.json`

# Results
- Episode 500:
- `no_shaping`: val success `0.0`, val steps `350.0`
- `phi_half`: val success `0.0`, val steps `350.0`
- `phi_full`: val success `0.0`, val steps `350.0`
- train mean steps도 전 조건 `350.0`

# Anomalies / Notes
- shaping 유무와 관계없이 완전 정체(전 조건 동일 실패).

# Interpretation (tentative)
- 현재 보상 스케일/탐색 설정에서 유효한 학습 신호가 형성되지 않음.

# Next Actions
1. epsilon schedule 도입(감소형/하한 조정).
2. goal reward scale sweep (`1.0, 2.0, 5.0`).
3. max_steps/episodes 증가 실험.

# Questions to resolve
- 실패 원인이 sparse reward인가 exploration 부족인가?
- Manhattan PBRS 항이 실제 행동 편향을 만들고 있는가?

