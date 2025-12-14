#!/usr/bin/env python
# coding: utf-8
"""
Ensemble: high.ipynb + Model_7 (No Cheating Version)

이 스크립트는 Kaggle 제출 시 cheating 방지를 위해:
1. Test 구간(8980~8989)을 최적화에 사용하지 않음
2. 대신 이전 구간(예: 8600~8800)에서 최적화
3. 학습된 패턴을 기반으로 미래 예측
"""

import os
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = None) -> float:
    """
    Calculates volatility-adjusted Sharpe ratio metric.
    """
    MIN_INVESTMENT = 0
    MAX_INVESTMENT = 2

    solut = solution.copy()
    solut['position'] = submission['prediction'].values

    if solut['position'].max() > MAX_INVESTMENT:
        raise ValueError(f'Position exceeds maximum of {MAX_INVESTMENT}')
    if solut['position'].min() < MIN_INVESTMENT:
        raise ValueError(f'Position below minimum of {MIN_INVESTMENT}')

    solut['strategy_returns'] = \
        solut['risk_free_rate'] * (1 - solut['position']) + \
        solut['forward_returns'] * solut['position']

    # Strategy Sharpe
    strategy_excess_returns = solut['strategy_returns'] - solut['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solut)) - 1
    strategy_std = solut['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ZeroDivisionError("Strategy std is zero")
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Market stats
    market_excess_returns = solut['forward_returns'] - solut['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solut)) - 1
    market_std = solut['forward_returns'].std()
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    # Penalties
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)


def train_powell_model(train_df, start_idx, end_idx, model_name="Model"):
    """
    Powell 최적화 (특정 구간)
    """
    solution = train_df.loc[start_idx:end_idx, ["forward_returns", "risk_free_rate"]]

    print(f"\n{model_name}: Optimizing on date_id {start_idx}~{end_idx} ({len(solution)} days)")

    def safe_score(x):
        x_clipped = np.clip(x, 0, 2)
        submission = pd.DataFrame({"prediction": x_clipped}, index=solution.index)
        return score(solution, submission, None)

    # Find best constant
    const_scores = []
    for const in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        preds = np.full(len(solution), const)
        const_scores.append((const, safe_score(preds)))

    best_const = max(const_scores, key=lambda x: x[1])[0]
    print(f"Best constant: {best_const} with score {max(const_scores, key=lambda x: x[1])[1]:.4f}")

    # Powell optimization
    print("Running Powell optimization...")
    res = minimize(
        lambda x: -safe_score(x),
        x0=np.full(solution.shape[0], best_const),
        method="Powell",
        bounds=[(0, 2)] * solution.shape[0],
        tol=1e-8,
        options={'maxiter': 200}
    )

    best_predictions = np.clip(res.x, 0, 2)
    best_score_val = safe_score(best_predictions)

    # Fallback to constant if needed
    if best_score_val < const_scores[-1][1]:
        best_predictions = np.full(len(solution), best_const)
        best_score_val = const_scores[-1][1]

    print(f"Final score: {best_score_val:.4f}")
    print(f"Mean position: {best_predictions.mean():.4f}")
    print(f"Std position: {best_predictions.std():.4f}")

    # Calculate statistics for extrapolation
    mean_pos = best_predictions.mean()
    median_pos = np.median(best_predictions)

    # Use median as default prediction (more robust)
    return median_pos, mean_pos, best_score_val


def create_ensemble_prediction(model1_pred, model2_pred, weight1=0.5):
    """
    Create ensemble prediction with weighted average.
    """
    weight2 = 1 - weight1
    ensemble_pred = weight1 * model1_pred + weight2 * model2_pred
    return np.clip(ensemble_pred, 0, 2)


if __name__ == "__main__":
    print("=" * 80)
    print("Ensemble Model Training (No Cheating Version)")
    print("=" * 80)

    # Load data
    train = pd.read_csv("train.csv", index_col="date_id")
    print(f"\nTrain shape: {train.shape}")
    print(f"Date range: {train.index.min()} ~ {train.index.max()}")

    # Strategy: Train on earlier periods to avoid data leakage
    # Model 1: Optimize on 8600~8800 (181 days, before test set)
    # Model 2: Optimize on 8700~8900 (201 days, before test set)

    print("\n" + "=" * 80)
    print("Model 1: Powell Optimization (8600~8800)")
    print("=" * 80)
    model1_median, model1_mean, model1_score = train_powell_model(
        train, start_idx=8600, end_idx=8800, model_name="Model1"
    )

    print("\n" + "=" * 80)
    print("Model 2: Powell Optimization (8700~8900)")
    print("=" * 80)
    model2_median, model2_mean, model2_score = train_powell_model(
        train, start_idx=8700, end_idx=8900, model_name="Model2"
    )

    # Test different ensemble weights on validation set (8901~8979)
    print("\n" + "=" * 80)
    print("Ensemble Weight Optimization on Validation Set (8901~8979)")
    print("=" * 80)

    valid_solution = train.loc[8901:8979, ["forward_returns", "risk_free_rate"]]

    best_weight = 0.5
    best_ensemble_score = 0

    weights_to_test = np.arange(0, 1.05, 0.1)

    for w in weights_to_test:
        ensemble_pred = create_ensemble_prediction(model1_median, model2_median, weight1=w)

        # Apply same prediction to all validation days
        preds = np.full(len(valid_solution), ensemble_pred)
        submission = pd.DataFrame({"prediction": preds}, index=valid_solution.index)

        try:
            ensemble_score = score(valid_solution, submission, None)
            print(f"Weight(model1)={w:.1f}, Weight(model2)={1-w:.1f} -> Pred: {ensemble_pred:.4f}, Score: {ensemble_score:.4f}")

            if ensemble_score > best_ensemble_score:
                best_ensemble_score = ensemble_score
                best_weight = w
        except Exception as e:
            print(f"Weight(model1)={w:.1f} failed: {e}")

    # Final prediction using best weight
    final_prediction = create_ensemble_prediction(model1_median, model2_median, weight1=best_weight)

    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"\nModel 1 (8600~8800): median={model1_median:.4f}, score={model1_score:.4f}")
    print(f"Model 2 (8700~8900): median={model2_median:.4f}, score={model2_score:.4f}")
    print(f"\nBest Ensemble Weight: model1={best_weight:.1f}, model2={1-best_weight:.1f}")
    print(f"Best Validation Score: {best_ensemble_score:.4f}")
    print(f"Final Prediction Value: {final_prediction:.4f}")

    # Test on final test set (8980~8989)
    print("\n" + "=" * 80)
    print("Test Set Performance (8980~8989)")
    print("=" * 80)

    test_solution = train.loc[8980:8989, ["forward_returns", "risk_free_rate"]]
    test_preds = np.full(len(test_solution), final_prediction)
    test_submission = pd.DataFrame({"prediction": test_preds}, index=test_solution.index)

    try:
        test_score = score(test_solution, test_submission, None)
        print(f"Test Score: {test_score:.4f}")
    except Exception as e:
        print(f"Test score failed: {e}")

    # Create submission file
    print("\n" + "=" * 80)
    print("Creating Submission")
    print("=" * 80)

    # For Kaggle submission, use the final prediction for all dates
    test_date_ids = test_solution.index.values

    submission_df = pd.DataFrame({
        "date_id": test_date_ids,
        "prediction": [final_prediction] * len(test_date_ids)
    })

    submission_df.to_csv("submission_ensemble_valid.csv", index=False)
    print("Submission saved to submission_ensemble_valid.csv")
    print(submission_df)

    print("\n" + "=" * 80)
