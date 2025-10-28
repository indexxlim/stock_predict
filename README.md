# 주가 예측 모델 프로젝트

## 프로젝트 개요
금융 시계열 데이터를 활용한 주가 수익률(forward returns) 예측 모델

## 완료된 작업
- [x] 데이터 탐색 및 분석 (EDA)
- [x] 데이터 전처리 및 특성 엔지니어링
- [x] 모델 학습 (LightGBM + XGBoost + CatBoost 앙상블)
- [x] 모델 검증 및 튜닝 (5-Fold TimeSeriesSplit)
- [x] 예측 생성 및 제출 파일 생성
- [x] 모델 실행

## 현재 모델 성능
- **평균 RMSE**: 0.010937
- **평균 R2**: 0.000550
- **앙상블 구성**: LightGBM(40%) + XGBoost(30%) + CatBoost(30%)
- **특성 수**: 144개 (기본 94개 + 생성 50개)

## TODO: 모델 개선 및 추가 작업

### 1. 모델 성능 개선
- [ ] 하이퍼파라미터 최적화 (Optuna, GridSearch 등)
  - LightGBM, XGBoost, CatBoost 각각의 최적 파라미터 탐색
  - learning_rate, num_leaves, max_depth 등 튜닝
- [ ] 더 많은 특성 엔지니어링
  - 시계열 기반 특성 (rolling mean, rolling std, lag features)
  - 도메인 지식 기반 금융 지표 추가
  - PCA 또는 차원 축소 기법 적용
- [ ] 앙상블 가중치 최적화
  - 각 모델의 최적 가중치 탐색
  - Stacking, Blending 기법 적용

### 2. 추가 모델 시도
- [ ] Neural Network 기반 모델
  - LSTM/GRU for 시계열 예측
  - Transformer 기반 모델
  - 1D-CNN 모델
- [ ] 다른 Gradient Boosting 모델
  - HistGradientBoosting
  - NGBoost
- [ ] 앙상블 기법 확장
  - Voting Regressor
  - Stacking with Meta-learner

### 3. 데이터 분석 및 시각화
- [ ] 심층 EDA 수행
  - 각 특성 그룹별 상관관계 분석
  - 타겟 변수와의 관계 시각화
  - 시계열 패턴 분석
- [ ] 특성 중요도 분석
  - SHAP values 계산
  - Feature importance plot 생성
  - 불필요한 특성 제거

### 4. 모델 검증 및 진단
- [ ] 예측 오차 분석
  - 잔차 플롯 및 분석
  - 오버피팅/언더피팅 진단
- [ ] 시간대별 성능 분석
  - 각 시간 구간별 모델 성능 비교
- [ ] Cross-validation 전략 개선
  - Purged TimeSeriesSplit 적용
  - Walk-forward validation

### 5. 코드 개선 및 자동화
- [ ] 모델 파이프라인 구축
  - 전처리-학습-예측 자동화
  - Config 파일로 설정 관리
- [ ] 로깅 및 모니터링
  - 학습 과정 로그 기록
  - MLflow 또는 Weights & Biases 연동
- [ ] 모델 저장 및 버전 관리
  - 학습된 모델 pickle/joblib 저장
  - 각 실험별 결과 추적

### 6. 실험 및 연구
- [ ] 다양한 결측치 처리 방법 실험
  - 중앙값/평균 대체 vs MICE vs KNN imputation
- [ ] 이상치 탐지 및 처리
  - IQR, Z-score 기반 이상치 제거
- [ ] 타겟 변수 변환
  - Log transformation
  - Winsorization

### 7. 문서화
- [ ] 모델 설명 문서 작성
  - 모델 아키텍처 다이어그램
  - 각 컴포넌트 설명
- [ ] 코드 주석 및 Docstring 추가
- [ ] 실험 결과 리포트 작성

## 파일 구조
```
stock_predict/
├── README.md                       # 현재 파일
├── stock_prediction_model.py       # 메인 모델 코드
├── train.csv                       # 학습 데이터
├── test.csv                        # 테스트 데이터
├── submission.csv                  # 제출 파일 (CSV)
└── submission.parquet              # 제출 파일 (Parquet)
```

## 실행 방법
```bash
conda activate kag
python stock_prediction_model.py
```

## 데이터 정보
- **Train 샘플 수**: 8,990개
- **Test 샘플 수**: 10개
- **특성 그룹**:
  - D (Date): 9개 특성
  - E (Economic): 20개 특성
  - I (Industry): 9개 특성
  - M (Market): 18개 특성
  - P (Price): 13개 특성
  - S (Sentiment): 12개 특성
  - V (Volume): 13개 특성
- **타겟**: forward_returns (미래 수익률)

## 참고사항
- TimeSeriesSplit을 사용하여 시계열 데이터의 특성을 고려한 교차 검증 수행
- Early stopping을 적용하여 과적합 방지
- 결측치는 중앙값으로 대체 (향후 개선 필요)
