# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_pbrs_manhattan_explore_v1/report_pbrs_manhattan_explore_v1.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_v1`

# Goal
- Manhattan PBRS에 first-visit exploration bonus를 추가했을 때 성능 변화를 평가.

# Hypothesis
- Inferred: `exploration_bonus = 0.05`가 탐색 폭을 늘려 validation 성공률을 높인다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- episodes `500`, runs `12`, max_steps `350`

# Algorithm
- Tabular SARSA.

# Reward
- `r = r_env + gamma*Phi_k(s') - Phi_k(s) + r_explore`
- `r_explore = 0.05` if first-visit in current episode else `0`
- `gamma=0.99`, `step_reward=0.0`, `goal_reward=1.0`
- Leakage Risk:
- PBRS part: heuristic (Manhattan)
- exploration part: learned/trajectory-dependent intrinsic signal (oracle 아님)

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `0.99`
- validation_interval `25`, validation_episodes `30`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_v1/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_v1/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_v1/run_summary.json`

# Results
- Episode 500:
- `no_shaping`: val success `0.0`, val steps `350.0`
- `phi_half`: val success `0.0`, val steps `350.0`
- `phi_full`: val success `0.0`, val steps `350.0`

# Anomalies / Notes
- exploration bonus 추가에도 기존 v2와 동일하게 전조건 실패.

# Interpretation (tentative)
- first-visit bonus 크기(`0.05`) 또는 형태가 정책 개선에 불충분.

# Next Actions
1. exploration bonus scale sweep (`0.01, 0.05, 0.1, 0.2`).
2. first-visit 기준을 전역 visit-count로 변경 비교.
3. bonus를 episode 초반에만 주는 schedule 도입.

# Questions to resolve
- bonus가 실제 탐색 증가를 만들었는지 방문 다양성 로그가 필요한가?
- PBRS와 intrinsic bonus의 상호작용이 상쇄되고 있나?

