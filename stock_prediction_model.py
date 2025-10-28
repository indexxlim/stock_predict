import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("주가 예측 모델 - 데이터 로드 및 탐색")
print("=" * 80)

# 데이터 로드
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"\n[데이터 정보]")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns: {list(train.columns)}")

# 타겟 변수 확인
target = 'forward_returns'
print(f"\n[타겟 변수: {target}]")
print(f"Mean: {train[target].mean():.6f}")
print(f"Std: {train[target].std():.6f}")
print(f"Min: {train[target].min():.6f}")
print(f"Max: {train[target].max():.6f}")

# 결측치 확인
print(f"\n[결측치 정보]")
missing_train = train.isnull().sum()
missing_cols = missing_train[missing_train > 0]
if len(missing_cols) > 0:
    print(f"Train 결측치가 있는 컬럼 수: {len(missing_cols)}")
    print(f"Train 전체 결측치 비율: {(train.isnull().sum().sum() / (train.shape[0] * train.shape[1])) * 100:.2f}%")
else:
    print("Train에 결측치가 없습니다.")

missing_test = test.isnull().sum()
missing_test_cols = missing_test[missing_test > 0]
if len(missing_test_cols) > 0:
    print(f"Test 결측치가 있는 컬럼 수: {len(missing_test_cols)}")
    print(f"Test 전체 결측치 비율: {(test.isnull().sum().sum() / (test.shape[0] * test.shape[1])) * 100:.2f}%")
else:
    print("Test에 결측치가 없습니다.")

print("\n" + "=" * 80)
print("데이터 전처리 및 특성 엔지니어링")
print("=" * 80)

# 특성 그룹별로 분류
feature_groups = {
    'D': [col for col in train.columns if col.startswith('D')],
    'E': [col for col in train.columns if col.startswith('E')],
    'I': [col for col in train.columns if col.startswith('I')],
    'M': [col for col in train.columns if col.startswith('M')],
    'P': [col for col in train.columns if col.startswith('P')],
    'S': [col for col in train.columns if col.startswith('S')],
    'V': [col for col in train.columns if col.startswith('V')]
}

print(f"\n[특성 그룹]")
for group, cols in feature_groups.items():
    print(f"{group} 그룹: {len(cols)}개 특성")

# 제외할 컬럼
exclude_cols = ['date_id', 'is_scored', target, 'risk_free_rate', 'market_forward_excess_returns']
base_features = [col for col in train.columns if col not in exclude_cols]

print(f"\n기본 특성 수: {len(base_features)}")

# 추가 특성 생성
def create_features(df, is_train=True):
    df = df.copy()

    # 각 그룹별 통계 특성
    for group, cols in feature_groups.items():
        available_cols = [col for col in cols if col in df.columns]
        if len(available_cols) > 0:
            # 결측치가 아닌 값들로만 계산
            df[f'{group}_mean'] = df[available_cols].mean(axis=1)
            df[f'{group}_std'] = df[available_cols].std(axis=1)
            df[f'{group}_min'] = df[available_cols].min(axis=1)
            df[f'{group}_max'] = df[available_cols].max(axis=1)
            df[f'{group}_range'] = df[f'{group}_max'] - df[f'{group}_min']

    # E, M, P, S, V 그룹의 상호작용 특성 (주요 그룹만)
    for i, (g1, cols1) in enumerate(list(feature_groups.items())[1:]):  # D 제외
        available_cols1 = [col for col in cols1 if col in df.columns]
        if len(available_cols1) > 0 and f'{g1}_mean' in df.columns:
            for g2, cols2 in list(feature_groups.items())[i+2:]:
                available_cols2 = [col for col in cols2 if col in df.columns]
                if len(available_cols2) > 0 and f'{g2}_mean' in df.columns:
                    df[f'{g1}_{g2}_interaction'] = df[f'{g1}_mean'] * df[f'{g2}_mean']

    # 결측치를 중앙값으로 대체
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df

print("\n특성 생성 중...")
train_processed = create_features(train, is_train=True)
test_processed = create_features(test, is_train=False)

# 최종 특성 선택
feature_cols = [col for col in train_processed.columns
                if col not in exclude_cols and col in test_processed.columns]

print(f"최종 특성 수: {len(feature_cols)}")

# 학습 데이터 준비
X = train_processed[feature_cols].values
y = train_processed[target].values

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\n" + "=" * 80)
print("모델 학습 - 앙상블 (LightGBM + XGBoost + CatBoost)")
print("=" * 80)

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# 모델별 예측 저장
lgb_predictions = []
xgb_predictions = []
cat_predictions = []

lgb_test_preds = np.zeros(len(test_processed))
xgb_test_preds = np.zeros(len(test_processed))
cat_test_preds = np.zeros(len(test_processed))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"\n[Fold {fold}/5]")

    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # LightGBM
    print("  LightGBM 학습 중...")
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }

    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
    lgb_val = lgb.Dataset(X_val_fold, y_val_fold, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )

    lgb_val_pred = lgb_model.predict(X_val_fold, num_iteration=lgb_model.best_iteration)
    lgb_test_pred = lgb_model.predict(test_processed[feature_cols].values, num_iteration=lgb_model.best_iteration)
    lgb_test_preds += lgb_test_pred / 5

    # XGBoost
    print("  XGBoost 학습 중...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'tree_method': 'hist',
        'verbosity': 0
    }

    dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
    dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    xgb_val_pred = xgb_model.predict(dval)
    xgb_test_pred = xgb_model.predict(xgb.DMatrix(test_processed[feature_cols].values))
    xgb_test_preds += xgb_test_pred / 5

    # CatBoost
    print("  CatBoost 학습 중...")
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    cat_model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        verbose=100
    )

    cat_val_pred = cat_model.predict(X_val_fold)
    cat_test_pred = cat_model.predict(test_processed[feature_cols].values)
    cat_test_preds += cat_test_pred / 5

    # 앙상블 예측 (가중 평균)
    ensemble_val_pred = (lgb_val_pred * 0.4 + xgb_val_pred * 0.3 + cat_val_pred * 0.3)

    # 평가
    rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_val_pred))
    r2 = r2_score(y_val_fold, ensemble_val_pred)

    fold_scores.append({'fold': fold, 'rmse': rmse, 'r2': r2})
    print(f"  Fold {fold} - RMSE: {rmse:.6f}, R2: {r2:.6f}")

print("\n" + "=" * 80)
print("교차 검증 결과")
print("=" * 80)
for score in fold_scores:
    print(f"Fold {score['fold']}: RMSE = {score['rmse']:.6f}, R2 = {score['r2']:.6f}")

avg_rmse = np.mean([s['rmse'] for s in fold_scores])
avg_r2 = np.mean([s['r2'] for s in fold_scores])
print(f"\n평균 RMSE: {avg_rmse:.6f}")
print(f"평균 R2: {avg_r2:.6f}")

print("\n" + "=" * 80)
print("최종 예측 생성")
print("=" * 80)

# 최종 앙상블 예측
final_predictions = (lgb_test_preds * 0.4 + xgb_test_preds * 0.3 + cat_test_preds * 0.3)

print(f"\n예측 통계:")
print(f"Mean: {final_predictions.mean():.6f}")
print(f"Std: {final_predictions.std():.6f}")
print(f"Min: {final_predictions.min():.6f}")
print(f"Max: {final_predictions.max():.6f}")

# 제출 파일 생성
submission = pd.DataFrame({
    'date_id': test['date_id'],
    target: final_predictions
})

submission.to_parquet('submission.parquet', index=False)
submission.to_csv('submission.csv', index=False)

print(f"\n제출 파일이 생성되었습니다:")
print(f"  - submission.parquet")
print(f"  - submission.csv")
print(f"\nSubmission shape: {submission.shape}")
print(f"\nSubmission 샘플:")
print(submission.head(10))

print("\n" + "=" * 80)
print("모델 학습 완료!")
print("=" * 80)
