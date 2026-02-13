# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_icml_style_manhattan_v1/report_manhattan.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_icml_style_manhattan_v1`
- Script: `신규_실험_workspace/experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py`

# Goal
- Manhattan potential 기반 PBRS가 동일 미로에서 학습/검증 개선을 주는지 확인.

# Hypothesis
- Inferred: Manhattan PBRS도 `no_shaping` 대비 성능 개선을 제공할 수 있다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- episodes `500`, runs `12`, max_steps `350`

# Algorithm
- Tabular SARSA, epsilon-greedy.

# Reward
- Base reward: step `-1`
- PBRS with Manhattan potential:
- `Phi(s)=-(|r-r_g|+|c-c_g|)`, `Phi_k(s)=k*Phi(s)`, `k in {0,0.5,1.0}`
- `R'_k = R + gamma*Phi_k(s') - Phi_k(s)` with `gamma=1.0`
- Leakage Risk: heuristic (Manhattan), 목표 위치 정보는 직접 사용.

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `1.0`
- validation_interval `25`, validation_episodes `30`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_icml_style_manhattan_v1/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_manhattan_v1/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_manhattan_v1/run_summary.json`

# Results
- Episode 500:
- `no_shaping`: train `350.0`, val success `0.0`, val steps `350.0`
- `phi_half`: train `316.58`, val success `0.0`, val steps `350.0`
- `phi_full`: train `242.67`, val success `0.0472`, val steps `345.08`
- Peak success (`phi_full`): `0.0861`

# Anomalies / Notes
- train steps 개선 대비 validation 성공률은 매우 낮음.
- 성공률/steps 곡선 분산이 크고, 안정적 수렴 신호 약함.

# Interpretation (tentative)
- Manhattan potential이 벽 구조를 반영하지 못해 유도 신호가 왜곡될 가능성.
- finite-sample에서 정책 불변성과 sample efficiency는 별개로 나타남.

# Next Actions
1. BFS vs Manhattan를 동일 seed 세트로 통계 비교.
2. Manhattan + obstacle-aware correction potential 실험.
3. train/validation 정책 갭(탐험 포함 vs greedy) 원인 분해.

# Questions to resolve
- Manhattan potential의 실패는 구조적 한계인가, 하이퍼파라미터 문제인가?
- validation 0 근처에서 train 개선이 의미 있는 탐색인가?

