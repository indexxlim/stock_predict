"""
Improved Stock Prediction Model
- 기존 모델 분석 기반으로 개선
- LightGBM + XGBoost + CatBoost 앙상블
- 향상된 특성 엔지니어링
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

# ============ CONFIG ============
DATA_PATH = Path("/kaggle/input/hull-tactical-market-prediction/")
MIN_SIGNAL = 0.0
MAX_SIGNAL = 2.0
SIGNAL_MULTIPLIER = 400.0

print("=" * 80)
print("개선된 주가 예측 모델")
print("=" * 80)


# ============ DATA LOADING ============
def load_data():
    """데이터 로드 및 전처리"""
    train = (
        pl.read_csv(DATA_PATH / "train.csv")
        .rename({"market_forward_excess_returns": "target"})
        .with_columns(pl.exclude("date_id").cast(pl.Float64, strict=False))
        .head(-10)  # 마지막 10개는 validation용
    )

    test = (
        pl.read_csv(DATA_PATH / "test.csv")
        .rename({"lagged_forward_returns": "target"})
        .with_columns(pl.exclude("date_id").cast(pl.Float64, strict=False))
    )

    return train, test


# ============ FEATURE ENGINEERING ============
def create_advanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """향상된 특성 엔지니어링"""

    # 모델의 핵심 특성
    base_features = [
        "S2",
        "E2",
        "E3",
        "P9",
        "S1",
        "S5",
        "I2",
        "P8",
        "P10",
        "P12",
        "P13",
    ]

    # 추가 유망 특성 (상관관계 분석 기반)
    additional_features = [
        "M1",
        "M2",
        "M11",
        "V1",
        "V2",
        "E1",
        "E4",
        "E5",
        "I1",
        "I7",
        "I9",
        "P1",
        "P2",
        "S3",
        "S4",
    ]

    all_base = list(set(base_features + additional_features))

    # 특성 생성
    df = df.with_columns(
        [
            # 모델의 특성
            (pl.col("I2") - pl.col("I1")).alias("U1"),
            (
                pl.col("M11")
                / ((pl.col("I2") + pl.col("I9") + pl.col("I7")) / 3 + 1e-8)
            ).alias("U2"),
            # 추가 비율 특성
            (pl.col("M1") / (pl.col("M2") + 1e-8)).alias("M_ratio"),
            (pl.col("P1") / (pl.col("P2") + 1e-8)).alias("P_ratio"),
            (pl.col("V1") / (pl.col("V2") + 1e-8)).alias("V_ratio"),
            # 그룹별 평균 특성
            ((pl.col("E1") + pl.col("E2") + pl.col("E3")) / 3).alias("E_mean"),
            ((pl.col("S1") + pl.col("S2") + pl.col("S5")) / 3).alias("S_mean"),
            ((pl.col("P8") + pl.col("P9") + pl.col("P10")) / 3).alias("P_mean"),
            # 상호작용 특성
            (pl.col("S2") * pl.col("E2")).alias("SE_interact"),
            (pl.col("P9") * pl.col("M1")).alias("PM_interact"),
            (pl.col("I2") * pl.col("M11")).alias("IM_interact"),
        ]
    )

    # 최종 특성 리스트
    feature_cols = all_base + [
        "U1",
        "U2",
        "M_ratio",
        "P_ratio",
        "V_ratio",
        "E_mean",
        "S_mean",
        "P_mean",
        "SE_interact",
        "PM_interact",
        "IM_interact",
    ]

    # 결측치 처리 - EWM 사용 (모델 방식)
    for col in feature_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(pl.col(col).ewm_mean(com=0.5)))

    return df.select(["date_id", "target"] + feature_cols).drop_nulls()


# ============ MODEL TRAINING ============
def train_models(X_train, y_train, X_val, y_val):
    """3개 모델 앙상블 학습"""

    print("\n[모델 학습 시작]")

    # LightGBM
    print("  LightGBM 학습 중...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)],
    )

    # XGBoost
    print("  XGBoost 학습 중...")
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "tree_method": "hist",
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=0,
    )

    # CatBoost
    print("  CatBoost 학습 중...")
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    return lgb_model, xgb_model, cat_model


def predict_ensemble(models, X, scaler=None):
    """앙상블 예측"""
    lgb_model, xgb_model, cat_model = models

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # 각 모델의 예측
    lgb_pred = lgb_model.predict(X_scaled, num_iteration=lgb_model.best_iteration)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_scaled))
    cat_pred = cat_model.predict(X_scaled)

    # 가중 평균 (LightGBM에 더 높은 가중치)
    ensemble_pred = 0.5 * lgb_pred + 0.25 * xgb_pred + 0.25 * cat_pred

    return ensemble_pred


def convert_ret_to_signal(ret_arr):
    """수익률을 신호로 변환"""
    return np.clip(ret_arr * SIGNAL_MULTIPLIER + 1, MIN_SIGNAL, MAX_SIGNAL)


# ============ MAIN ============
if __name__ == "__main__":
    # 데이터 로드
    print("\n[데이터 로드]")
    train_pl, test_pl = load_data()
    print(f"Train shape: {train_pl.shape}")
    print(f"Test shape: {test_pl.shape}")

    # 특성 생성
    print("\n[특성 엔지니어링]")
    df = pl.concat([train_pl, test_pl], how="vertical")
    df_processed = create_advanced_features(df)

    train_processed = df_processed.filter(
        pl.col("date_id").is_in(train_pl.get_column("date_id"))
    )
    test_processed = df_processed.filter(
        pl.col("date_id").is_in(test_pl.get_column("date_id"))
    )

    # 특성 리스트
    feature_cols = [
        col for col in train_processed.columns if col not in ["date_id", "target"]
    ]
    print(f"특성 수: {len(feature_cols)}")

    # 데이터 분리
    X_train = train_processed.select(feature_cols).to_pandas().values
    y_train = train_processed.get_column("target").to_numpy()
    X_test = test_processed.select(feature_cols).to_pandas().values
    y_test = test_processed.get_column("target").to_numpy()

    # 스케일링
    print("\n[데이터 스케일링]")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Validation 분할 (마지막 20%를 validation으로)
    split_idx = int(len(X_train_scaled) * 0.8)
    X_train_final = X_train_scaled[:split_idx]
    y_train_final = y_train[:split_idx]
    X_val = X_train_scaled[split_idx:]
    y_val = y_train[split_idx:]

    # 모델 학습
    models = train_models(X_train_final, y_train_final, X_val, y_val)

    # Validation 성능 평가
    print("\n[Validation 성능]")
    val_pred = predict_ensemble(models, X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_r2 = 1 - np.sum((val_pred - y_val) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
    print(f"Validation RMSE: {val_rmse:.6f}")
    print(f"Validation R2: {val_r2:.6f}")

    # 전체 데이터로 재학습
    print("\n[전체 데이터로 최종 모델 학습]")
    final_models = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

    # 테스트 예측
    print("\n[테스트 예측 생성]")
    test_pred_raw = predict_ensemble(final_models, X_test_scaled)
    test_pred_signal = convert_ret_to_signal(test_pred_raw)

    print(f"\n예측 통계:")
    print(
        f"Raw predictions - Mean: {test_pred_raw.mean():.6f}, Std: {test_pred_raw.std():.6f}"
    )
    print(
        f"Signal - Mean: {test_pred_signal.mean():.6f}, Std: {test_pred_signal.std():.6f}"
    )
    print(
        f"Signal - Min: {test_pred_signal.min():.6f}, Max: {test_pred_signal.max():.6f}"
    )

    # 제출 파일 생성
    submission = pd.DataFrame(
        {
            "date_id": test_processed.get_column("date_id").to_pandas(),
            "signal": test_pred_signal,
        }
    )

    submission.to_csv("submission_improved.csv", index=False)
    print(f"\n제출 파일 생성 완료: submission_improved.csv")
    print(f"\nSubmission 샘플:")
    print(submission.head(10))

    print("\n" + "=" * 80)
    print("모델 학습 완료!")
    print("=" * 80)
