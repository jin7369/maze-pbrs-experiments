# Potential-based Shaping Reward Experiments (Maze RL)

이 저장소는 **Potential-Based Reward Shaping(PBRS)** 를 미로 탐색 강화학습 환경에서 검증한 실험 모음입니다.  
주요 목표는 다음 질문에 답하는 것입니다.

- 잠재함수 기반 shaping이 실제로 학습을 빠르게 만드는가?
- 종료 조건(고정 step, 재방문 종료, 재방문 임계치 종료)에 따라 결과가 어떻게 달라지는가?
- Manhattan/BFS 기반 potential과 exploration 설계가 성능에 어떤 영향을 주는가?

## 1) Repository Scope

현재 워크스페이스에는 아래 3가지 축의 실험이 포함되어 있습니다.

1. ICML 1999 스타일 재현 및 변형
2. 미로 종료 규칙 변경 실험(재방문 기반 종료)
3. 별도 REINFORCE(신경망 정책) 실험

---

## 2) Directory Guide

```text
신규_실험_workspace/
├─ experiments/
│  ├─ maze_generation_demo/
│  │  └─ generate_maze_samples.py
│  ├─ maze_shaping_icml_style/
│  │  └─ run_maze_shaping_experiment.py
│  ├─ maze_shaping_revisit_termination/
│  │  ├─ run_manhattan_revisit_termination.py
│  │  └─ run_manhattan_revisit_penalty_threshold.py
│  └─ paper_reproduction_icml1999/
│     └─ run_shaping_experiments.py
├─ reinforce_memory_env/
│  ├─ experiments/
│  │  └─ run_reinforce_memory_experiment.py
│  ├─ outputs/
│  └─ reports/
├─ outputs/        # 각 실험의 csv/png/json/gif 산출물
├─ reports/        # 각 실험의 tex/pdf/log 문서
├─ experiments_log_index.md
└─ README.md
```

추가로 루트에는 로컬 가상환경 폴더(`.venv`, `venv310`)가 존재합니다.

---

## 3) Environment Setup

권장: Python 3.10

```powershell
py -3.10 -m venv venv310
.\venv310\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install numpy pandas matplotlib pillow tqdm mazelib
```

LaTeX PDF를 생성하려면 MiKTeX(또는 TeX 배포판)가 필요합니다.

---

## 4) Quick Reproduction

### 4.1 Maze 샘플 생성

```powershell
python 신규_실험_workspace/experiments/maze_generation_demo/generate_maze_samples.py
```

기본 출력:
- `신규_실험_workspace/outputs/maze_samples_v1/maze_contact_sheet.png`
- `신규_실험_workspace/outputs/maze_samples_v1/maze_metadata.csv`

### 4.2 ICML-style PBRS 실험

```powershell
python 신규_실험_workspace/experiments/maze_shaping_icml_style/run_maze_shaping_experiment.py
```

기본 출력:
- `learning_curve.csv/png`
- `validation_progress.csv/png`
- `run_summary.json`
- `gifs/`

### 4.3 재방문 즉시 종료 실험

```powershell
python 신규_실험_workspace/experiments/maze_shaping_revisit_termination/run_manhattan_revisit_termination.py
```

### 4.4 재방문 패널티 + 임계치 종료 실험

```powershell
python 신규_실험_workspace/experiments/maze_shaping_revisit_termination/run_manhattan_revisit_penalty_threshold.py
```

### 4.5 REINFORCE 실험(별도 환경 분기)

```powershell
python 신규_실험_workspace/reinforce_memory_env/experiments/run_reinforce_memory_experiment.py
```

---

## 5) Results and Documents

- 실험 로그 인덱스: `신규_실험_workspace/experiments_log_index.md`
- 실험별 문서(LaTeX/PDF/로그): `신규_실험_workspace/reports/`
- 주요 결과 데이터: `신규_실험_workspace/outputs/`

대표 PDF 예시:
- `신규_실험_workspace/reports/maze_shaping_icml_style_v1/report.pdf`
- `신규_실험_workspace/reports/maze_shaping_revisit_penalty_threshold_manhattan_v1/report_revisit_penalty_threshold_manhattan_v1.pdf`

---

## 6) Public Release Checklist (GitHub)

업로드 전에 아래를 점검하세요.

1. 가상환경/캐시 제외
- `.venv/`, `venv310/`, `__pycache__/`, `*.pyc`

2. 중간 산출물 정리
- LaTeX 중간파일(`*.aux`, `*.log`)은 필요 없다면 제외

3. 인코딩 점검
- `experiments_log_index.md` 및 일부 문서는 UTF-8로 저장 권장

4. 대용량 파일 정책
- `outputs/`와 `reports/`의 이미지/PDF가 많으면 저장소가 커질 수 있으므로
  필요한 핵심 결과만 남기거나 Git LFS 사용 고려

---

## 7) Current Caveats

- 재방문 기반 종료 규칙에서는 "종료까지 step" 지표와 "goal 성공률"이 분리되는 현상이 관찰됩니다.
- 일부 실험에서는 PBRS가 step dynamics는 바꾸지만 validation success를 개선하지 못했습니다.
- 따라서 해석 시 **성공률(success)** 과 **평균 step** 을 함께 확인해야 합니다.

---

## 8) Citation

이 워크스페이스는 다음 이론적 배경을 중심으로 설계되었습니다.

- Ng, Harada, Russell (1999), Policy Invariance under Reward Transformations

