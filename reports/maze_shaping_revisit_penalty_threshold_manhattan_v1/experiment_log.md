# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_revisit_penalty_threshold_manhattan_v1/report_revisit_penalty_threshold_manhattan_v1.tex`
- Output root: `신규_실험_workspace/outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1`
- Script: `신규_실험_workspace/experiments/maze_shaping_revisit_termination/run_manhattan_revisit_penalty_threshold.py`

# Goal
- Revisit 즉시 종료 규칙의 경직성을 완화하기 위해, 같은 에피소드 내 재방문을 허용하되 누적 재방문 횟수가 임계치(`revisit_terminate_count=3`)에 도달하면 종료하는 방식으로 Manhattan PBRS를 재검증한다.

# Hypothesis
- Inferred: 재방문 완화(+패널티)로 탐색 지속 시간이 늘고 목표 도달 성공률이 개선될 수 있다.

# Environment & Data
- Maze path: `신규_실험_workspace/outputs/maze_samples_v1/grids/maze_00.npy`
- Termination: `goal reached` or `revisit count in same episode reaches revisit_terminate_count`
- episodes `5000`, runs `12`
- validation_interval `25`, validation_episodes `30`

# Algorithm
- Tabular SARSA, epsilon-greedy.

# Reward
- Base reward: `step_reward=0.0`, `goal_reward=1.0`
- Revisit penalty: `revisit_penalty=-0.05`
- Potential-based shaping:
- `R'_k = R + gamma*Phi_k(s') - Phi_k(s)`
- `Phi(s)=-(|r-r_g|+|c-c_g|)`, `Phi_k(s)=k*Phi(s)`, `k in {0,0.5,1.0}`
- `gamma=0.99`
- Leakage Risk: heuristic (goal-coordinate-aware Manhattan potential)

# Hyperparameters
- alpha `0.02`, epsilon `0.10`, gamma `0.99`
- conditions: `no_shaping`, `phi_half`, `phi_full`

# Metrics & Artifacts
- `신규_실험_workspace/outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1/learning_curve.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1/learning_curve.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1/validation_progress.csv`
- `신규_실험_workspace/outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1/validation_progress.png`
- `신규_실험_workspace/outputs/maze_shaping_revisit_penalty_threshold_manhattan_v1/run_summary.json`

# Results
- Episode 5000 train mean steps:
- `no_shaping`: `25.166666666666668`
- `phi_half`: `13.166666666666666`
- `phi_full`: `11.333333333333334`
- Episode 5000 validation:
- all conditions success `0.0`
- validation mean steps: `no_shaping=47.583333333333336`, `phi_half=15.333333333333334`, `phi_full=11.5`
- Best validation success:
- `no_shaping=0.4166666666666667` (episode `4850`)
- `phi_half=0.0`, `phi_full=0.0`

# Anomalies / Notes
- 학습 곡선상 종료까지 스텝은 유지/개선되지만, PBRS 조건의 validation 성공률은 끝까지 `0.0`.
- `no_shaping`은 후반(episode `4850`)에만 일시적으로 `0.4166666666666667` 성공률을 보이고 최종값은 `0.0`으로 회귀.
- 성공률과 스텝 지표가 분리되어 움직이며, 이는 현재 종료 규칙+패널티 조합이 목표 도달 대신 조기 종료 전략을 강화했을 가능성을 시사.

# Interpretation (tentative)
- 재방문 임계치 완화만으로는 Manhattan PBRS 조건의 goal-reaching 정책을 유도하지 못했다.
- shaping이 termination step을 줄이는 방향으로 작동했지만, 실제 성공 상태로 수렴하는 학습 신호는 부족했다.

# Next Actions
1. `revisit_terminate_count`를 `3 -> 5/8`로 늘려 조기 종료 편향을 약화.
2. `revisit_penalty` 절대값을 낮춰(`-0.05 -> -0.01`) goal 보상 대비 과도한 회피를 완화.
3. epsilon schedule(초기 high, 점진 감쇠) 도입으로 초반 탐색량 확대.
4. validation을 `greedy`/`epsilon=0.05` 두 조건으로 분리해 정책 자체 실패와 탐색 실패를 분해.

# Questions to resolve
- PBRS 조건에서 성공률 0이 유지되는 원인이 potential scale, termination bias, exploration 부족 중 무엇이 지배적인가?
- 동일 종료 규칙에서 목표 도달 빈도를 높이기 위한 최소 보상/탐색 수정은 무엇인가?
