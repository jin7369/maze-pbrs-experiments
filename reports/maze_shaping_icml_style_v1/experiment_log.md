# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_icml_style_v1/report.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_icml_style_v1`
- Script: `신규_실험_workspace/experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py`

# Goal
- ICML-style 3조건(`no_shaping`, `phi_half`, `phi_full`)에서 미로 학습 성능 비교.

# Hypothesis
- Inferred: potential-based shaping, 특히 `phi_full`,은 `no_shaping` 대비 학습/검증 성능을 개선한다.

# Environment & Data
- Maze: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- Maze ID: `maze_00`, seed `0`
- Cell size: `10 x 10`, Grid size: `21 x 21`
- Start/Goal: `(1,1) / (19,19)`
- Shortest path length (BFS): `44`
- Wall count/ratio: `242 / 0.5488`
- Dead-end count: `13`

# Algorithm
- Tabular SARSA, epsilon-greedy.

# Reward
- Base reward: step `-1`.
- PBRS: `R'(s,a,s') = R(s,a,s') + gamma*Phi_k(s') - Phi_k(s)`.
- `Phi_0(s)`는 BFS distance 기반, `Phi_k(s)=k*Phi_0(s)`, `k in {0,0.5,1.0}`.
- Leakage Risk: heuristic (BFS shortest-path potential), oracle에 가까운 구조 정보 활용 가능성 있음.

# Hyperparameters
- episodes `500`, runs `12`
- alpha `0.02`, epsilon `0.10`, gamma `1.0`
- max_steps `350`
- validation_interval `25`, validation_episodes `30`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_icml_style_v1/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_v1/learning_curve.png`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_v1/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_v1/validation_progress.png`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_v1/run_summary.json`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_v1/gifs/policy_rollout_ep_*.gif`

# Results
- Episode 500:
- `no_shaping`: train `350.00`, val success `0.0000`, val steps `350.00`
- `phi_half`: train `268.42`, val success `0.0056`, val steps `349.12`
- `phi_full`: train `99.25`, val success `0.6306`, val steps `229.26`

# Anomalies / Notes
- `phi_half`는 train steps는 개선됐지만 validation 성공률은 거의 `0`.
- 동일 PBRS 형식이라도 potential scale에 따라 성능 격차가 큼.

# Interpretation (tentative)
- BFS 기반 `phi_full`은 이 미로에서 유효한 탐색 방향 신호를 제공.
- `phi_half`는 신호 강도가 부족하거나 탐색-수렴 균형이 맞지 않은 것으로 보임.

# Next Actions
1. `maze_00~maze_09` 전수 평균으로 재평가.
2. `phi` scale sweep (`0.1~1.0`) 수행.
3. validation에 성공 trajectory 길이 분포 추가 기록.

# Questions to resolve
- BFS potential의 성능이 미로별로 일관적인가?
- `phi_half`가 실패하는 주요 원인이 신호 강도인가 탐색 부족인가?

