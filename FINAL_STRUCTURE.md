# 최종 파일 구조

## 📂 디렉토리 구조

```
stock_predict/
│
├── 📊 데이터 파일
│   ├── train.csv                           # 전체 학습 데이터 (8,990 샘플)
│   ├── train_90.csv                        # Train split (8,091 샘플, 90%)
│   ├── valid_10.csv                        # Valid split (899 샘플, 10%)
│   └── test.csv                            # 테스트 데이터 (10 샘플)
│
├── 🧠 모델 파일
│   ├── best_chronos.pth                    # Chronos 모델 가중치 (2.4MB)
│   └── best_informer.pth                   # Informer 모델 가중치 (2.8MB)
│
├── 📓 노트북
│   ├── ensemble_model7_chronos.ipynb       # ✨ 최종 앙상블 (Model 7 + Chronos)
│   ├── hull-starter-notebook_17.ipynb      # 원본 17점 노트북 (참고용)
│   ├── notebook-17score.ipynb              # 분석용 노트북
│   └── high.ipynb                          # 실험용 노트북
│
├── 🐍 Python 스크립트
│   ├── stock_deep.py                       # Chronos & Informer 구현
│   ├── train_model7.py                     # Model 7 훈련 스크립트
│   ├── compare_all_models.py               # 전체 모델 비교
│   ├── stock_prediction_model.py           # 기본 모델
│   ├── improved_model.py                   # 개선 모델
│   ├── feature_importance_analysis.py      # Feature 중요도 분석
│   └── find_best_window.py                 # 윈도우 최적화
│
├── 📚 문서
│   ├── README.md                           # 프로젝트 메인 문서
│   ├── model7_analysis.md                  # Model 7 Data Leakage 분석
│   ├── FEATURE_SELECTION_STRATEGY.md       # Feature 선택 전략
│   └── result.md                           # 기본 모델 결과
│
├── 📈 분석 결과
│   ├── feature_importance_analysis.csv     # Feature 중요도 데이터
│   └── feature_importance_comparison.png   # Feature 중요도 시각화
│
└── 📦 Kaggle Evaluation
    └── kaggle_evaluation/                  # Kaggle 평가 라이브러리
        ├── __init__.py
        ├── default_gateway.py
        ├── default_inference_server.py
        └── core/
            ├── __init__.py
            ├── base_gateway.py
            ├── relay.py
            ├── templates.py
            └── generated/
                ├── __init__.py
                ├── kaggle_evaluation_pb2.py
                └── kaggle_evaluation_pb2_grpc.py
```

## 📝 파일 설명

### 핵심 실행 파일

| 파일 | 용도 | 설명 |
|------|------|------|
| `ensemble_model7_chronos.ipynb` | ⭐ 최종 제출용 | Model 7 + Chronos 앙상블 (Informer 추가 가능) |
| `stock_deep.py` | 딥러닝 모델 훈련 | Chronos & Informer 학습 스크립트 |
| `train_model7.py` | Model 7 훈련 | Powell 최적화 실행 |

### 분석/비교 파일

| 파일 | 용도 |
|------|------|
| `compare_all_models.py` | 9개 모델 성능 비교 |
| `feature_importance_analysis.py` | Feature 중요도 분석 |
| `find_best_window.py` | 최적 윈도우 크기 탐색 |

### 참고/실험 파일

| 파일 | 용도 |
|------|------|
| `hull-starter-notebook_17.ipynb` | 원본 17점 노트북 (참고용) |
| `notebook-17score.ipynb` | 분석 및 검증용 |
| `high.ipynb` | 실험용 노트북 |
| `stock_prediction_model.py` | 기본 모델 (초기 버전) |
| `improved_model.py` | 개선 모델 (중간 버전) |

## 🗑️ 삭제된 파일

다음 파일들은 중복/구버전으로 삭제되었습니다:

**노트북:**
- `hull-starter-notebook.ipynb` (초기 버전)
- `start.ipynb` (실험용)
- `submission_notebook.ipynb` (구버전)

**앙상블 스크립트:**
- `ensemble_final.py`
- `ensemble_high_model7_final.py`
- `ensemble_high_model7.ipynb`
- `ensemble_improved_chronos.py`
- `ensemble_improved_informer.py`
- `ensemble_valid.py`

**기타:**
- `improve_high.py` (실험용)

**구버전 제출 파일:**
- `submission.csv`
- `submission.parquet`
- `submission_improved.csv`
- `submission_ensemble_*.csv` (여러 개)

## 💾 디스크 사용량

**데이터 파일:** ~25MB
- train.csv: 12MB
- train_90.csv: 11MB
- valid_10.csv: 1.4MB
- test.csv: 17KB

**모델 파일:** ~5MB
- best_chronos.pth: 2.4MB
- best_informer.pth: 2.8MB

**전체:** ~30MB (kaggle_evaluation 제외)

## 🚀 빠른 시작

### 1. 최종 앙상블 실행
```bash
cd /home/klcube/lim/kaggle/stock_predict
jupyter notebook ensemble_model7_chronos.ipynb
```

### 2. 딥러닝 모델 재훈련 (필요시)
```bash
python stock_deep.py
```

### 3. 모델 비교 실행
```bash
python compare_all_models.py
```

## 📊 현재 최고 성능

- **Chronos (단독)**: MSE 0.000109
- **Informer (단독)**: MSE 0.000110
- **Model 7 (단독)**: 17.396 (노트북), 0.365 (Valid)
- **목표 앙상블**: 18+ (Model 7 + Chronos + Informer)

---

**정리 완료일:** 2025-12-16
**총 파일 수:** 11 Python/Notebooks + 4 문서 + 6 데이터/모델 = 21개 핵심 파일
