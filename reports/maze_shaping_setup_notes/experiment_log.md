# Frontmatter
- Source: `신규_실험_workspace/reports/maze_shaping_setup_notes/maze_shaping_setup_notes.tex`
- Type: configuration document (not a result report)

# Goal
- 실험 코드 세팅(환경/보상/검증/아티팩트)을 재현 가능하게 명세.

# Hypothesis
- Inferred: 명시적 설정 문서화는 실험 재현성과 비교 실험 품질을 개선한다.

# Environment & Data
- Maze MDP, 4-action, 벽 충돌 시 제자리.
- Start `(1,1)`, Goal `(h-2,w-2)`.

# Algorithm
- Tabular SARSA (기본 실험 코드 기준).

# Reward
- 기본 문서: step reward `-1`, PBRS 조건 `k in {0,0.5,1.0}`.
- `bfs`와 `manhattan` potential 옵션 명시.
- Leakage Risk:
- BFS: heuristic/oracle-like
- Manhattan: heuristic

# Hyperparameters
- 기본값 표: episodes `500`, runs `12`, alpha `0.02`, epsilon `0.10`, gamma `1.0`, max_steps `350`, validation_interval `25`, validation_episodes `30`.

# Metrics & Artifacts
- `learning_curve.csv/png`
- `validation_progress.csv/png`
- `run_summary.json`
- `gifs/policy_rollout_ep_*.gif`

# Results
- N/A (설정 문서; 실험 결과 미포함)

# Anomalies / Notes
- 결과치가 아닌 구성 노트이므로 수치 성능 비교 불가.

# Interpretation (tentative)
- 이후 로그들이 동일 구조를 쓰면 실험 간 비교가 쉬워짐.

# Next Actions
1. 각 실험 실행 후 동일 템플릿 로그 자동 생성.
2. setup note와 실제 run_summary 일치 여부 자동 검증.
3. 결과 없는 문서와 결과 문서를 분리 라벨링.

# Questions to resolve
- 설정 변경 이력이 언제/왜 바뀌는지 추적 체계가 필요한가?

