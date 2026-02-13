# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_pbrs_manhattan_explore_potential_v1/report_pbrs_manhattan_explore_potential_v1.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_potential_v1`

# Goal
- 탐색 보상을 potential-based 형태(`Psi`)로 바꿨을 때 Manhattan PBRS 성능 검증.

# Hypothesis
- Inferred: `Psi(s)=-N(s)` 기반 exploration shaping이 first-visit bonus보다 안정적 성능을 제공한다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- episodes `500`, runs `12`, max_steps `350`

# Algorithm
- Tabular SARSA.

# Reward
- `r = r_env + gamma*Phi_k(s') - Phi_k(s) + beta*(gamma*Psi(s') - Psi(s))`
- `Phi(s)=-(|r-r_g|+|c-c_g|)`, `Phi_k(s)=k*Phi(s)`, `k in {0,0.5,1.0}`
- `Psi(s)=-N(s)` (`N(s)`: in-episode visit count)
- `beta=0.05`, `gamma=0.99`, `step_reward=0.0`, `goal_reward=1.0`
- Leakage Risk:
- Phi: heuristic (Manhattan)
- Psi: learned trajectory statistic (oracle 아님)

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `0.99`
- validation_interval `25`, validation_episodes `30`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_potential_v1/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_potential_v1/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_icml_style_pbrs_manhattan_explore_potential_v1/run_summary.json`

# Results
- Episode 500:
- `no_shaping`: train `299.416667`, val success `0.166667`, val steps `299.0`
- `phi_half`: train `350.000000`, val success `0.000000`, val steps `350.0`
- `phi_full`: train `350.000000`, val success `0.000000`, val steps `350.0`

# Anomalies / Notes
- `no_shaping`만 일부 성공, shaping 조건(`phi_half/full`)은 전부 실패.
- Success=0인 조건에서 train steps도 `350`으로 고정되어 완전 정체.

# Interpretation (tentative)
- Manhattan Phi와 exploration Psi 조합이 충돌하거나 잘못된 편향을 줄 가능성.

# Next Actions
1. `beta` sweep (`0.0, 0.01, 0.05, 0.1`)으로 민감도 확인.
2. `Psi`를 count 대신 novelty density 기반으로 교체 비교.
3. no_shaping 대비 shaping-only, explore-only ablation 수행.

# Questions to resolve
- 실패 원인이 Phi(맨해튼)인지 Psi(방문카운트)인지 분리 가능한가?
- shaping 조건에서 어떤 루프 패턴이 나타나는가?

