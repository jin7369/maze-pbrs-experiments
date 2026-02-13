# maze-pbrs-experiments

Maze 환경에서 Potential-Based Reward Shaping(PBRS)을 비교한 강화학습 실험 저장소입니다.

핵심 질문:
- PBRS가 학습 효율(steps, success rate)을 개선하는가?
- 종료 조건(고정 step / 재방문 종료 / 재방문 임계치 종료)에 따라 성능이 어떻게 달라지는가?
- Manhattan/BFS potential 및 exploration shaping이 결과에 어떤 영향을 주는가?

## Project Structure

```text
.
├─ experiments/
│  ├─ maze_generation_demo/
│  ├─ maze_shaping_icml_style/
│  ├─ maze_shaping_revisit_termination/
│  └─ paper_reproduction_icml1999/
├─ reinforce_memory_env/
├─ outputs/
├─ reports/
├─ experiments_log_index.md
└─ README.md
```

- `outputs/`: csv/png/json/gif 실험 산출물
- `reports/`: 실험별 LaTeX/PDF/experiment log
- `experiments_log_index.md`: 전체 실험 로그 인덱스

## Setup

권장: Python 3.10

```powershell
py -3.10 -m venv venv310
.\venv310\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install numpy pandas matplotlib pillow tqdm mazelib
```

PDF 빌드가 필요하면 MiKTeX(또는 다른 TeX 배포판)를 설치하세요.

## Run

### 1) Maze 생성
```powershell
python experiments/maze_generation_demo/generate_maze_samples.py
```

### 2) ICML-style PBRS 실험
```powershell
python experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py
```

### 3) 재방문 종료 실험
```powershell
python experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py
```

### 4) 재방문 패널티+임계치 실험
```powershell
python experiments/maze_shaping_revisit_termination/run_manhattan_revisit_penalty_threshold.py
```

### 5) REINFORCE 실험
```powershell
python reinforce_memory_env/experiments/run_reinforce_memory_experiment.py
```

## Results

- 메인 결과: `outputs/`
- 보고서 PDF: `reports/`
- 실험 로그 모음: `experiments_log_index.md`

대표 문서:
- `reports/maze_shaping_icml_style_v1/report.pdf`
- `reports/maze_shaping_revisit_penalty_threshold_manhattan_v1/report_revisit_penalty_threshold_manhattan_v1.pdf`

## Reference

- Ng, Harada, Russell (1999), *Policy Invariance under Reward Transformations*
