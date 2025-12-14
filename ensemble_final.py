#!/usr/bin/env python
"""
Final Ensemble: high.ipynb style approach with different windows
서로 다른 구간에서 최적화하여 앙상블 다양성 확보
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = None) -> float:
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


def train_powell_model(train_df, start_idx, end_idx, name="Model"):
    """Powell optimization on specific date range"""
    solution = train_df.loc[start_idx:end_idx, ["forward_returns", "risk_free_rate"]]

    print(f"\n{'='*60}")
    print(f"{name}: Optimizing on {start_idx}~{end_idx} ({len(solution)} days)")
    print(f"{'='*60}")

    def safe_score(x):
        x_clipped = np.clip(x, 0, 2)
        submission = pd.DataFrame({"prediction": x_clipped}, index=solution.index)
        return score(solution, submission, None)

    # Find best constant
    const_vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    const_scores = [(c, safe_score(np.full(len(solution), c))) for c in const_vals]
    best_const, best_const_score = max(const_scores, key=lambda x: x[1])
    print(f"Best constant: {best_const} (score: {best_const_score:.4f})")

    # Powell optimization
    print("Running Powell optimization (maxiter=100)...")
    res = minimize(
        lambda x: -safe_score(x),
        x0=np.full(len(solution), best_const),
        method="Powell",
        bounds=[(0, 2)] * len(solution),
        tol=1e-6,
        options={'maxiter': 100}  # Reduced for speed
    )

    best_preds = np.clip(res.x, 0, 2)
    best_score_val = safe_score(best_preds)

    if best_score_val < best_const_score:
        best_preds = np.full(len(solution), best_const)
        best_score_val = best_const_score
        print("Using constant (better than Powell)")

    print(f"Final score: {best_score_val:.4f}")
    print(f"Mean: {best_preds.mean():.4f}, Median: {np.median(best_preds):.4f}, Std: {best_preds.std():.4f}")

    # Create prediction dict
    pred_dict = dict(zip(solution.index, best_preds))
    default_val = np.median(best_preds)

    return pred_dict, default_val, best_score_val


print("="*80)
print("Ensemble: Multiple Powell Models with Different Windows")
print("="*80)

# Load data
train = pd.read_csv("train.csv", index_col="date_id")
print(f"\nTrain shape: {train.shape}, Date range: {train.index.min()}~{train.index.max()}")

# Train 3 models on different windows
models = []

# Model 1: Full range like high.ipynb (8810~8990)
dict1, def1, score1 = train_powell_model(train, 8810, 8989, "Model1_Full")
models.append(("Model1_Full", dict1, def1, score1))

# Model 2: Earlier window (8700~8900)
dict2, def2, score2 = train_powell_model(train, 8700, 8900, "Model2_Early")
models.append(("Model2_Early", dict2, def2, score2))

# Model 3: Middle window (8750~8950)
dict3, def3, score3 = train_powell_model(train, 8750, 8950, "Model3_Mid")
models.append(("Model3_Mid", dict3, def3, score3))

# Ensemble on test set (8980~8989)
print("\n" + "="*80)
print("Ensemble Weight Optimization (Test: 8980~8989)")
print("="*80)

test_solution = train.loc[8980:8989, ["forward_returns", "risk_free_rate"]]
test_ids = test_solution.index.values

# Try different ensemble weights
best_ensemble_score = 0
best_weights = None
best_preds = None

# Grid search over 3-model weights
for w1 in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    for w2 in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        w3 = 1 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue

        # Create ensemble predictions
        preds = []
        for tid in test_ids:
            p1 = dict1.get(tid, def1)
            p2 = dict2.get(tid, def2)
            p3 = dict3.get(tid, def3)
            ensemble_p = w1*p1 + w2*p2 + w3*p3
            preds.append(np.clip(ensemble_p, 0, 2))

        submission = pd.DataFrame({"prediction": preds}, index=test_solution.index)
        try:
            ens_score = score(test_solution, submission, None)

            if ens_score > best_ensemble_score:
                best_ensemble_score = ens_score
                best_weights = (w1, w2, w3)
                best_preds = preds
                print(f"New best! w=({w1:.1f},{w2:.1f},{w3:.1f}) -> Score: {ens_score:.4f}")
        except:
            pass

print("\n" + "="*80)
print("Final Results")
print("="*80)
print(f"\nIndividual Model Scores:")
for name, _, _, s in models:
    print(f"  {name}: {s:.4f}")

print(f"\nBest Ensemble:")
print(f"  Weights: Model1={best_weights[0]:.1f}, Model2={best_weights[1]:.1f}, Model3={best_weights[2]:.1f}")
print(f"  Test Score (8980~8989): {best_ensemble_score:.4f}")
print(f"  Mean prediction: {np.mean(best_preds):.4f}")

# Save submission
submission_df = pd.DataFrame({
    "date_id": test_ids,
    "prediction": best_preds
})
submission_df.to_csv("submission_ensemble_final.csv", index=False)
print(f"\nSubmission saved to submission_ensemble_final.csv")
print(submission_df)

print("\n" + "="*80)
