# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_noslip_v1/report_noslip.tex`
- Output roots:
- `신규_실험_workspace/outputs/maze_shaping_icml_style_bfs_noslip_v1`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_manhattan_noslip_v1`

# Goal
- slip 제거(no-slip) 시 BFS/Manhattan PBRS의 성능 변화를 비교.

# Hypothesis
- Inferred: no-slip 환경에서는 shaping 효과가 더 뚜렷해진다.

# Environment & Data
- Maze: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- deterministic transition (no slip)
- episodes `500`, runs `12`, max_steps `350`

# Algorithm
- Tabular SARSA.

# Reward
- Base reward: step `-1`
- PBRS with either BFS or Manhattan distance potential.
- Leakage Risk:
- BFS: heuristic/oracle-like (벽 반영 최단경로)
- Manhattan: heuristic (벽 비반영)

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `1.0`
- validation_interval `25`, validation_episodes `30`

# Metrics & Artifacts
- BFS no-slip: `신규_실험_workspace/outputs/maze_shaping_icml_style_bfs_noslip_v1/...`
- Manhattan no-slip: `신규_실험_workspace/outputs/maze_shaping_icml_style_manhattan_noslip_v1/...`

# Results
- Episode 500 (BFS no-slip):
- `no_shaping` train `342.25`, val success `0.0`, val steps `350.0`
- `phi_half` train `207.33`, val success `0.0`, val steps `350.0`
- `phi_full` train `49.75`, val success `1.0`, val steps `44.0`
- Episode 500 (Manhattan no-slip):
- `no_shaping` train `342.25`, val success `0.0`, val steps `350.0`
- `phi_half` train `248.75`, val success `0.0`, val steps `350.0`
- `phi_full` train `153.25`, val success `0.0`, val steps `350.0`

# Anomalies / Notes
- no-slip에서도 Manhattan 조건은 validation 전구간 실패.
- BFS `phi_full`만 완전 성공(`1.0`)으로 분리됨.

# Interpretation (tentative)
- 환경 stochasticity보다 potential 정합성(BFS vs Manhattan)이 지배적 요인.

# Next Actions
1. no-slip 상태에서 Manhattan potential scale/offset sweep.
2. BFS potential noise 주입해 robust성 테스트.
3. 목표 도달률 외 경로 최적성 지표 추가.

# Questions to resolve
- Manhattan 실패는 미로 구조 특이성인가 일반 현상인가?
- BFS success가 다른 maze seed에서도 재현되는가?

