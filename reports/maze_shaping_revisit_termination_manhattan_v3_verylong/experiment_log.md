# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_revisit_termination_manhattan_v3_verylong/report_revisit_termination_manhattan_v3_verylong.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v3_verylong`
- Script: `신규_실험_workspace/experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py`

# Goal
- 재방문 종료 규칙에서 episodes를 `20000`까지 늘렸을 때 goal success가 나타나는지 확인.

# Hypothesis
- Inferred: 학습 횟수를 크게 늘리면 PBRS 조건의 증가된 steps-to-termination이 goal 성공으로 전환될 수 있다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- Termination: `goal reached` or `revisit any previously visited state in same episode`
- episodes `20000`, runs `12`
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
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v3_verylong/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v3_verylong/learning_curve.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v3_verylong/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v3_verylong/validation_progress.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_termination_manhattan_v3_verylong/run_summary.json`

# Results
- Episode 20000:
- `no_shaping`: train `1.666667`, val success `0.0`, val steps `1.000000`
- `phi_half`: train `8.666667`, val success `0.0`, val steps `19.000000`
- `phi_full`: train `6.083333`, val success `0.0`, val steps `12.583333`
- Best validation success: 전 조건 `0.0`

# Anomalies / Notes
- PBRS 조건에서 validation steps는 늘었지만 success는 여전히 `0.0`.
- `success=0`인데 steps만 증가하는 패턴은 종료 지연 학습 가능성을 시사.

# Interpretation (tentative)
- 단순 학습 횟수 증가로는 실패가 해소되지 않음.
- 현재 종료 규칙이 goal-reaching 학습 신호를 구조적으로 차단하는 것으로 보임.

# Next Actions
1. 재방문 즉시 종료를 제거하고 penalty 방식으로 대체.
2. 재방문 종료 임계치 방식(`>=N회`)으로 완화.
3. success-dependent curriculum (초기엔 일반 종료, 이후 재방문 제약 추가) 실험.

# Questions to resolve
- PBRS가 실제로 goal 방향 정보를 주고 있는지, 아니면 종료 회피 전략만 강화하는가?
- 종료 규칙을 완화하면 steps 증가가 성공률 증가로 전환되는가?

