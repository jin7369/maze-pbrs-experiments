# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_revisit_termination_manhattan_v2_longtrain/report_revisit_termination_manhattan_v2_longtrain.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain`
- Script: `신규_실험_workspace/experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py`

# Goal
- 재방문 종료 규칙에서 학습 횟수를 크게 늘리면(`500 -> 5000`) 성능이 회복되는지 확인.

# Hypothesis
- Inferred: 학습 횟수 부족이 주요 원인이었다면 장기 학습에서 validation 성공률이 상승한다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- Termination: `goal reached` or `revisit any previously visited state in same episode`
- episodes `5000`, runs `12`
- validation_interval `25`, validation_episodes `30`

# Algorithm
- Tabular SARSA, epsilon-greedy.

# Reward
- `step_reward = 0.0`, `goal_reward = 1.0`
- Manhattan PBRS:
- `R'_k = R + gamma*Phi_k(s') - Phi_k(s)`
- `Phi(s)=-(|r-r_g|+|c-c_g|)`, `Phi_k(s)=k*Phi(s)`, `k in {0,0.5,1.0}`
- `gamma = 0.99`
- Leakage Risk: heuristic (Manhattan potential, 목표 좌표 정보 사용)

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `0.99`
- conditions: `no_shaping`, `phi_half`, `phi_full`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain/learning_curve.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain/validation_progress.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v2_longtrain/run_summary.json`

# Results
- Episode 5000:
- `no_shaping`: train `1.916667`, val success `0.0`, val steps `1.000000`
- `phi_half`: train `9.166667`, val success `0.0`, val steps `10.083333`
- `phi_full`: train `5.333333`, val success `0.0`, val steps `8.416667`
- Best validation success: 전 조건 `0.0`

# Anomalies / Notes
- 학습 횟수를 10배 늘려도 성공률은 여전히 `0.0`.
- steps 값은 증가했지만(특히 shaping), goal success로 이어지지 않음.
- `success=0`인데 train steps가 늘어나는 패턴은 종료 규칙 하에서 “더 오래 버티기”에 가까운 신호일 수 있음.

# Interpretation (tentative)
- 이번 결과는 “단순 학습 횟수 부족” 가설을 지지하지 않음.
- 실패의 주원인은 재방문 즉시 종료 규칙 자체일 가능성이 높음.

# Next Actions
1. 재방문 시 즉시 종료 대신 패널티 방식으로 변경해 재실험.
2. 재방문 종료를 `N회 이상 재방문` 임계치 종료로 완화.
3. 학습 초반에는 재방문 종료 비활성(warm-up) 후 점진 적용.

# Questions to resolve
- 종료 규칙 완화 시 Manhattan PBRS가 실제 성공률 개선을 보이는가?
- 현재 steps 증가가 유의미한 탐색인지 단순 반복 회피 행동인지 어떻게 판별할 것인가?

