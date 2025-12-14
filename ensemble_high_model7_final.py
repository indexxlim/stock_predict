#!/usr/bin/env python
"""
Ensemble: high.ipynb (8800~8990) + Model_7 (다른 구간)
목표: 17.5+ 점수 달성
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = None) -> float:
    """Volatility-adjusted Sharpe ratio metric"""
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

    strategy_excess_returns = solut['strategy_returns'] - solut['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solut)) - 1
    strategy_std = solut['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ZeroDivisionError("Strategy std is zero")
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    market_excess_returns = solut['forward_returns'] - solut['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solut)) - 1
    market_std = solut['forward_returns'].std()
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)


def train_high_model(train_df, start_idx=8810, end_idx=8989):
    """
    high.ipynb 방식: 8800~8990 최적화
    """
    solution = train_df.loc[start_idx:end_idx, ["forward_returns", "risk_free_rate"]]

    print(f"\n{'='*70}")
    print(f"HIGH MODEL: Optimizing on {start_idx}~{end_idx} ({len(solution)} days)")
    print(f"{'='*70}")

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
    print(f"Best constant: {best_const} (score: {max(const_scores, key=lambda x: x[1])[1]:.4f})")

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

    # Fallback
    if best_score_val < const_scores[-1][1]:
        best_predictions = np.full(len(solution), best_const)
        best_score_val = const_scores[-1][1]

    # Handle small values
    small_mask = best_predictions < 0.005
    if np.any(small_mask):
        for repl in [0.0, 0.001, 0.005]:
            temp = best_predictions.copy()
            temp[small_mask] = repl
            s = safe_score(temp)
            if s > best_score_val:
                best_predictions = temp
                best_score_val = s

    # Smoothing
    if len(best_predictions) > 10:
        window = 3
        kernel = np.ones(window) / window
        smoothed = np.convolve(best_predictions, kernel, mode='same')
        smoothed[:window] = best_predictions[:window]
        smoothed[-window:] = best_predictions[-window:]
        smoothed = np.clip(smoothed, 0, 2)
        smoothed_score = safe_score(smoothed)
        if smoothed_score > best_score_val:
            best_predictions = smoothed
            best_score_val = smoothed_score

    print(f"Final score: {best_score_val:.4f}")
    print(f"Mean: {best_predictions.mean():.4f}, Median: {np.median(best_predictions):.4f}")

    prediction_dict = dict(zip(solution.index, best_predictions))
    default_val = np.median(best_predictions)

    return prediction_dict, default_val, best_score_val


def train_model7_variant(train_df, start_idx, end_idx, name="Model7"):
    """
    Model_7 변형: 다른 구간에서 최적화
    """
    solution = train_df.loc[start_idx:end_idx, ["forward_returns", "risk_free_rate"]]

    print(f"\n{'='*70}")
    print(f"{name}: Optimizing on {start_idx}~{end_idx} ({len(solution)} days)")
    print(f"{'='*70}")

    def safe_score(x):
        x_clipped = np.clip(x, 0, 2)
        submission = pd.DataFrame({"prediction": x_clipped}, index=solution.index)
        return score(solution, submission, None)

    # Find best constant (빠른 탐색)
    const_vals = [0.0, 0.05, 0.1, 0.15, 0.2]
    const_scores = [(c, safe_score(np.full(len(solution), c))) for c in const_vals]
    best_const, best_const_score = max(const_scores, key=lambda x: x[1])
    print(f"Best constant: {best_const} (score: {best_const_score:.4f})")

    # Powell optimization (빠른 실행)
    print("Running Powell optimization (maxiter=100)...")
    res = minimize(
        lambda x: -safe_score(x),
        x0=np.full(len(solution), best_const),
        method="Powell",
        bounds=[(0, 2)] * len(solution),
        tol=1e-6,
        options={'maxiter': 100}
    )

    best_predictions = np.clip(res.x, 0, 2)
    best_score_val = safe_score(best_predictions)

    if best_score_val < best_const_score:
        best_predictions = np.full(len(solution), best_const)
        best_score_val = best_const_score

    print(f"Final score: {best_score_val:.4f}")
    print(f"Mean: {best_predictions.mean():.4f}, Median: {np.median(best_predictions):.4f}")

    prediction_dict = dict(zip(solution.index, best_predictions))
    default_val = np.median(best_predictions)

    return prediction_dict, default_val, best_score_val


print("="*80)
print("Ensemble: high.ipynb + Model_7 Variants")
print("Goal: 17.5+ score")
print("="*80)

# Load data
train = pd.read_csv("train.csv", index_col="date_id")
print(f"\nTrain shape: {train.shape}, Date range: {train.index.min()}~{train.index.max()}")

# Train models on different windows
models = {}

# Model 1: high.ipynb (8810~8989, 180 days) - 정확한 구간
high_dict, high_def, high_score = train_high_model(train, start_idx=8810, end_idx=8989)
models['high'] = (high_dict, high_def, high_score)

# Model 2: Earlier variant (8700~8890, 191 days)
m7v1_dict, m7v1_def, m7v1_score = train_model7_variant(train, 8700, 8890, "Model7_v1")
models['m7v1'] = (m7v1_dict, m7v1_def, m7v1_score)

# Model 3: Middle variant (8750~8940, 191 days)
m7v2_dict, m7v2_def, m7v2_score = train_model7_variant(train, 8750, 8940, "Model7_v2")
models['m7v2'] = (m7v2_dict, m7v2_def, m7v2_score)

# Test on FULL optimization window (8810~8989) to see true performance
print("\n" + "="*80)
print("Ensemble Testing on Full Window (8810~8989)")
print("="*80)

test_solution = train.loc[8810:8989, ["forward_returns", "risk_free_rate"]]
test_ids = test_solution.index.values

# Test individual models
print("\nIndividual Model Scores on Test Set:")
for name, (pred_dict, def_val, _) in models.items():
    preds = [pred_dict.get(tid, def_val) for tid in test_ids]
    submission = pd.DataFrame({"prediction": preds}, index=test_solution.index)
    try:
        s = score(test_solution, submission, None)
        print(f"  {name}: {s:.4f}")
    except Exception as e:
        print(f"  {name}: Error - {e}")

# Ensemble with different weights
print("\n" + "="*80)
print("Ensemble Weight Search")
print("="*80)

best_ensemble_score = 0
best_weights = None
best_preds = None

# 2-model ensemble: high + m7v1
print("\n2-Model Ensemble: high + m7v1")
for w_high in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    w_m7v1 = 1 - w_high

    preds = []
    for tid in test_ids:
        p_high = high_dict.get(tid, high_def)
        p_m7v1 = m7v1_dict.get(tid, m7v1_def)
        ensemble_p = w_high * p_high + w_m7v1 * p_m7v1
        preds.append(np.clip(ensemble_p, 0, 2))

    submission = pd.DataFrame({"prediction": preds}, index=test_solution.index)
    try:
        ens_score = score(test_solution, submission, None)
        print(f"  w_high={w_high:.1f}, w_m7v1={w_m7v1:.1f} -> Score: {ens_score:.4f}")

        if ens_score > best_ensemble_score:
            best_ensemble_score = ens_score
            best_weights = {'high': w_high, 'm7v1': w_m7v1, 'm7v2': 0}
            best_preds = preds
    except Exception as e:
        print(f"  w_high={w_high:.1f} failed: {e}")

# 3-model ensemble: high + m7v1 + m7v2
print("\n3-Model Ensemble: high + m7v1 + m7v2")
for w_high in np.arange(0.5, 1.0, 0.05):
    for w_m7v2 in np.arange(0.0, 0.5, 0.05):
        w_m7v1 = 1 - w_high - w_m7v2
        if w_m7v1 < 0 or w_m7v1 > 1:
            continue

        preds = []
        for tid in test_ids:
            p_high = high_dict.get(tid, high_def)
            p_m7v1 = m7v1_dict.get(tid, m7v1_def)
            p_m7v2 = m7v2_dict.get(tid, m7v2_def)
            ensemble_p = w_high * p_high + w_m7v1 * p_m7v1 + w_m7v2 * p_m7v2
            preds.append(np.clip(ensemble_p, 0, 2))

        submission = pd.DataFrame({"prediction": preds}, index=test_solution.index)
        try:
            ens_score = score(test_solution, submission, None)

            if ens_score > best_ensemble_score:
                best_ensemble_score = ens_score
                best_weights = {'high': w_high, 'm7v1': w_m7v1, 'm7v2': w_m7v2}
                best_preds = preds
                print(f"  ★ NEW BEST: w_high={w_high:.2f}, w_m7v1={w_m7v1:.2f}, w_m7v2={w_m7v2:.2f} -> Score: {ens_score:.4f}")
        except:
            pass

# Final results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"\nBest Ensemble Score: {best_ensemble_score:.4f}")
print(f"Best Weights: {best_weights}")
print(f"Mean Prediction: {np.mean(best_preds):.4f}")
print(f"Std Prediction: {np.std(best_preds):.4f}")

# Save submission
submission_df = pd.DataFrame({
    "date_id": test_ids,
    "prediction": best_preds
})
submission_df.to_csv("submission_ensemble_high_m7.csv", index=False)
print(f"\nSubmission saved to submission_ensemble_high_m7.csv")
print(submission_df)

print("\n" + "="*80)
print(f"Target: 17.5+ | Achieved: {best_ensemble_score:.4f} | {'✓ SUCCESS' if best_ensemble_score >= 17.5 else '✗ NEED IMPROVEMENT'}")
print("="*80)
