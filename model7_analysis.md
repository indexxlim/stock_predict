# Model_7 성능 분석: 왜 Test에서는 좋고 Valid에서는 나쁠까?

## 📊 성능 비교

| Dataset | Score | 설명 |
|---------|-------|------|
| **Train (최적화 구간)** | 17.396 | Powell 최적화로 직접 최대화 |
| **Test (노트북)** | 17.396 | 동일하게 높은 점수! |
| **Valid (우리 split)** | 0.365 | 형편없는 점수... |

**질문**: 왜 Test에서는 17.396점인데 Valid에서는 0.365점일까?

---

## 🚨 핵심 발견: 100% Data Leakage

### 원본 노트북의 데이터 구성

```
전체 train.csv: date_id 0 ~ 8989 (8,990일)

┌─────────────────────────────────────────────────────────────────┐
│                    train.csv (0 ~ 8989)                         │
├─────────────────────────────────────┬───────────────────────────┤
│  Earlier data (0 ~ 8809)            │  Last 180 days (8810~8989)│
│                                     │                           │
│                                     │  ┌──────────────────┐     │
│                                     │  │ Model_7 최적화   │     │
│                                     │  │ 8810 ~ 8989      │     │
│                                     │  │                  │     │
│                                     │  │  ┌──────────┐    │     │
│                                     │  │  │ TEST     │    │     │
│                                     │  │  │ 8980~8989│ ◄──┼─────┼─ 100% OVERLAP!
│                                     │  │  └──────────┘    │     │
│                                     │  └──────────────────┘     │
└─────────────────────────────────────┴───────────────────────────┘
```

### 🔍 Data Leakage 상세 분석

| 항목 | 범위 | 일수 |
|------|------|------|
| Model_7 최적화 구간 | 8810 ~ 8989 | 180일 |
| Test set | 8980 ~ 8989 | 10일 |
| **겹치는 구간** | **8980 ~ 8989** | **10일 (100%)** |

### 💥 문제점

1. **Model_7은 Powell 최적화로 8810~8989의 각 날짜에 대한 최적 position을 계산**
2. **Test set(8980~8989)은 이 최적화 구간 안에 포함됨**
3. **Model_7은 test의 정답을 이미 알고 있는 상태에서 최적화됨**
4. **예측이 아니라 단순 조회(lookup)!**

```python
# Model_7의 예측 방식
def predict_Model_7(test: pl.DataFrame) -> float:
    global i_M7, opt_preds
    pred = np.float64(opt_preds[i_M7])  # 미리 계산된 최적값 그대로 반환
    i_M7 = i_M7 + 1
    return pred
```

---

## ✅ 우리의 올바른 Split

### 우리가 만든 데이터 구성

```
전체 train.csv: date_id 0 ~ 8989

┌─────────────────────────────────────────────────────────────────┐
│              train_90.csv (0 ~ 8090)                            │
├─────────────────────────────┬───────────────┐                   │
│  Earlier data (0 ~ 7910)    │ Last 180 days │                   │
│                             │ (7911 ~ 8090) │                   │
│                             │               │                   │
│                             │ ┌───────────┐ │                   │
│                             │ │ Model_7   │ │                   │
│                             │ │ 최적화    │ │                   │
│                             │ └───────────┘ │                   │
└─────────────────────────────┴───────────────┴───────────────────┘
                                                │
                                                │ 1 day gap
                                                ▼
                              ┌──────────────────────────────────┐
                              │  valid_10.csv (8091 ~ 8989)      │
                              │                                  │
                              │  ← 완전히 새로운 미래 데이터!     │
                              └──────────────────────────────────┘
```

### 🎯 올바른 설정

| 항목 | 범위 | 일수 |
|------|------|------|
| Train (90%) | 0 ~ 8090 | 8,091일 |
| Model_7 최적화 구간 | 7911 ~ 8090 | 180일 |
| Valid (10%) | 8091 ~ 8989 | 899일 |
| **겹치는 구간** | **없음** | **0일 (0%)** |
| Gap | - | 1일 |

### ✓ 장점

- Valid set은 Model_7이 **한 번도 보지 못한 데이터**
- 진짜 미래 예측 성능 측정 가능
- Leakage 없음

---

## 📈 성능 차이의 원인

### 노트북 (Test Score: 17.396)

```python
# 노트북의 Model_7
opt_window = train[8810:8989]  # 180일
test_set = train[8980:8989]    # 10일 - 최적화 구간 안에 있음!

# Powell 최적화
optimize(opt_window)  # test_set의 정답을 포함해서 최적화
→ test_set에 대한 최적 position을 이미 알고 있음
→ Score: 17.396 (완벽!)
```

### 우리 Valid (Score: 0.365)

```python
# 우리의 Model_7
opt_window = train[7911:8090]  # 180일
valid_set = train[8091:8989]   # 899일 - 완전히 밖에 있음!

# Powell 최적화
optimize(opt_window)  # valid_set 정보 전혀 없음
→ 최적화된 값의 평균(0.19)을 모든 날에 예측
→ Score: 0.365 (일반화 실패)
```

---

## 🎓 교훈

### 1. **Data Leakage는 미묘하게 발생한다**

```
일반적인 Leakage: Train에 Test 데이터가 섞임
이 경우의 Leakage: 최적화 구간에 Test가 포함됨
```

### 2. **시간 순서 Split이 중요하다**

❌ **잘못된 예시 (노트북)**
```python
train = full_data  # 0 ~ 8989
model.optimize(train[-180:])  # 8810 ~ 8989
test = full_data[8980:8989]    # 최적화 구간 안에 있음!
```

✅ **올바른 예시 (우리)**
```python
train = full_data[:int(len(full_data)*0.9)]  # 0 ~ 8090
model.optimize(train[-180:])  # 7911 ~ 8090
valid = full_data[int(len(full_data)*0.9):]  # 8091 ~ 8989 (완전히 미래)
```

### 3. **Powell 최적화는 Interpolation이지 Extrapolation이 아니다**

Model_7은:
- ✅ **Interpolation**: 최적화 구간 내의 값들 사이를 보간 → 17.396점
- ❌ **Extrapolation**: 최적화 구간 밖의 미래 예측 → 0.365점

---

## 🔬 실험 결과 정리

### 모든 모델의 Valid 성능

| Model | Valid Score | 비고 |
|-------|------------|------|
| Model_6 | **10.358** | forward_returns > 0이면 0.09 (cheating) |
| Model_5 | 10.267 | threshold 기반 (cheating) |
| Model_4 | 10.251 | 조건부 노출 (cheating) |
| Model_1 | 10.184 | 이진 전략 (cheating) |
| Model_2 | 10.015 | Market excess returns |
| **Chronos (DL)** | **MSE 0.000109** | 진짜 예측 모델 |
| **Informer (DL)** | **MSE 0.000110** | 진짜 예측 모델 |
| Model_7 | 0.365 | Powell (일반화 실패) |
| Model_3 | 0.047 | Stacking (과소적합) |

### 결론

1. **Cheating 모델들 (1,4,5,6)**: 정답을 보고 예측하므로 당연히 높은 점수
2. **딥러닝 (Chronos, Informer)**: 진짜 예측 모델 중 최고 성능
3. **Model_7**: 최적화 구간 밖에서는 전혀 작동하지 않음
4. **Model_3**: 복잡한 앙상블이지만 성능 최악

---

## 💡 최종 결론

### Model_7이 Test에서 높은 점수를 받은 이유

**한 문장 요약:**
> Test set이 Model_7의 최적화 구간 안에 100% 포함되어 있어서, 예측이 아니라 이미 계산된 최적값을 그대로 조회한 것이다.

### 시각화

```
Notebook:
[────── Model_7 Optimization (8810~8989) ──────]
                                    [── Test (8980~8989) ──]
                                    ↑
                                    100% Overlap!
                                    = Data Leakage
                                    = Score: 17.396

Our Split:
[──── Model_7 Optimization (7911~8090) ────]
                                           |Gap
                                           ↓
                        [────────── Valid (8091~8989) ──────────]
                        ↑
                        No overlap!
                        = Fair evaluation
                        = Score: 0.365
```

### 실전 의미

1. **노트북의 17.396 점수는 착시**
   - 실제 미래 예측 능력이 아님
   - 과거 데이터 암기의 결과

2. **실제 제출용 모델로는 부적합**
   - 새로운 데이터에 일반화 안 됨
   - Valid에서 0.365점이 진짜 성능

3. **진짜 사용해야 할 모델**
   - Chronos / Informer (MSE 0.0001)
   - 또는 cheating 모델의 패턴을 학습한 ML 모델

---

## 📁 생성된 파일

- `train_90.csv` - 학습 데이터 (90%, date_id 0~8090)
- `valid_10.csv` - 검증 데이터 (10%, date_id 8091~8989)
- `compare_all_models.py` - 7개 모델 비교 스크립트
- `stock_deep.py` - Chronos & Informer 구현
- `best_chronos.pth` - 최고 성능 Chronos 모델
- `best_informer.pth` - 최고 성능 Informer 모델
- **`model7_analysis.md`** - 본 분석 보고서

---

**작성일**: 2025-11-03
**분석 대상**: hull-starter-notebook_17.ipynb
**핵심 발견**: 100% Data Leakage in Test Set
