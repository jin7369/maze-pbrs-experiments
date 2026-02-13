# Experiment Log Index

아래는 현재 정리된 실험 로그 목록입니다.

## Logs
- `reports/maze_shaping_icml_style_v1/experiment_log.md`
  - BFS potential 기반 ICML-style 3조건 비교 로그
- `reports/maze_shaping_icml_style_manhattan_v1/experiment_log.md`
  - Manhattan potential 기반 ICML-style 3조건 비교 로그
- `reports/maze_shaping_noslip_v1/experiment_log.md`
  - no-slip 환경에서 BFS/Manhattan 비교 로그
- `reports/maze_shaping_pbrs_manhattan_v2/experiment_log.md`
  - `gamma=0.99`, `step=0`, `goal=1` 설정의 Manhattan PBRS 로그
- `reports/maze_shaping_pbrs_manhattan_explore_v1/experiment_log.md`
  - first-visit exploration bonus 추가 버전 로그
- `reports/maze_shaping_pbrs_manhattan_explore_potential_v1/experiment_log.md`
  - exploration을 potential 기반으로 준 버전 로그
- `reports/maze_shaping_revisit_termination_manhattan_v1/experiment_log.md`
  - 재방문 즉시 종료 + Manhattan PBRS 로그
- `reports/maze_shaping_revisit_termination_manhattan_v2_longtrain/experiment_log.md`
  - 재방문 종료 규칙 + 장기 학습(5000 episodes) 로그
- `reports/maze_shaping_revisit_termination_manhattan_v3_verylong/experiment_log.md`
  - 재방문 종료 규칙 + 초장기 학습(20000 episodes) 로그
- `reports/maze_shaping_revisit_penalty_threshold_manhattan_v1/experiment_log.md`
  - 재방문 패널티 + 임계치 종료(`revisit_terminate_count=3`) Manhattan PBRS 로그
- `reports/maze_shaping_setup_notes/experiment_log.md`
  - 실험 설정 설명 문서 로그
- `reinforce_memory_env/reports/reinforce_memory_v1/experiment_log.md`
  - REINFORCE + visited-table state encoding 실험 로그

## Notes
- 로그 템플릿은 `Frontmatter` ~ `Questions to resolve` 구조를 따릅니다.
- 상세 수치/해석/이상 징후는 각 실험 폴더의 `experiment_log.md`를 확인하세요.
