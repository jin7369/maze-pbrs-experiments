# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_revisit_termination_manhattan_v1/report_revisit_termination_manhattan_v1.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v1`
- Script: `신규_실험_workspace/experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py`

# Goal
- Manhattan PBRS에서 에피소드 종료를 `goal 도달 또는 재방문`으로 바꿨을 때 학습/검증 성능을 평가.

# Hypothesis
- Inferred: 재방문 종료 규칙이 루프를 조기에 차단하여 더 효율적인 탐색을 유도할 수 있다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- 종료 조건: `goal 도달` 또는 `같은 에피소드 내 이전 방문 상태 재방문`
- episodes `500`, runs `12`
- validation_interval `25`, validation_episodes `30`

# Algorithm
- Tabular SARSA, epsilon-greedy.

# Reward
- Base reward: `step_reward = 0.0`, `goal_reward = 1.0`
- PBRS:
- `R'_k = R + gamma*Phi_k(s') - Phi_k(s)`
- `Phi(s)=-(|r-r_g|+|c-c_g|)`, `Phi_k(s)=k*Phi(s)`, `k in {0,0.5,1.0}`
- `gamma = 0.99`
- Leakage Risk: heuristic (Manhattan potential; 목표 좌표 정보 사용)

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `0.99`
- step_reward `0.0`, goal_reward `1.0`
- conditions: `no_shaping`, `phi_half`, `phi_full`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v1/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v1/learning_curve.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v1/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v1/validation_progress.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v1/run_summary.json`

# Results
- Episode 500:
- `no_shaping`: train `2.500000`, val success `0.0`, val steps `1.000000`
- `phi_half`: train `2.666667`, val success `0.0`, val steps `2.666667`
- `phi_full`: train `2.333333`, val success `0.0`, val steps `2.416667`
- Best validation success: 전 조건 `0.0`

# Anomalies / Notes
- train steps가 `1~3` 수준으로 매우 짧게 붕괴됨.
- 모든 조건에서 validation 성공률 `0.0`.
- `no_shaping` validation mean steps가 `1.000000`으로, 사실상 즉시 종료 패턴.

# Interpretation (tentative)
- 재방문 즉시 종료 규칙이 과도하게 강해서 goal 도달 전에 에피소드가 조기 종료됨.
- 결과적으로 탐색/학습 신호가 거의 사라져 PBRS 효과를 평가하기 어려운 세팅이 됨.
- 학습 횟수 부족 가능성도 존재하지만, 현재 패턴(평균 1~3 step 종료)상 종료 규칙 영향이 더 지배적인 것으로 판단됨.

# Next Actions
1. 동일 규칙을 유지한 채 episodes를 대폭 확대(예: `500 -> 5000`)하여 학습 횟수 부족 가설을 먼저 검증.
2. 재방문 종료를 즉시 종료 대신 패널티 부여 방식으로 완화.
3. `k-step warmup`(초기 몇 step은 재방문 종료 비활성) 실험.

# Questions to resolve
- 현재 종료 규칙이 본래 연구 질문(Manhattan PBRS 효과 비교)에 적합한가?
- 재방문 억제를 종료 대신 보상으로만 처리하는 편이 더 안정적인가?
