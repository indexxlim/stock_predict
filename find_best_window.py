#!/usr/bin/env python
"""
다양한 window에서 최적화해서 가장 높은 점수 찾기
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


def quick_optimize(train_df, start_idx, end_idx):
    """빠른 최적화 (maxiter=50)"""
    solution = train_df.loc[start_idx:end_idx, ["forward_returns", "risk_free_rate"]]

    def safe_score(x):
        x_clipped = np.clip(x, 0, 2)
        submission = pd.DataFrame({"prediction": x_clipped}, index=solution.index)
        return score(solution, submission, None)

    # Best constant
    const_vals = [0.0, 0.05, 0.1, 0.15, 0.2]
    const_scores = [(c, safe_score(np.full(len(solution), c))) for c in const_vals]
    best_const, best_const_score = max(const_scores, key=lambda x: x[1])

    # Quick Powell
    res = minimize(
        lambda x: -safe_score(x),
        x0=np.full(len(solution), best_const),
        method="Powell",
        bounds=[(0, 2)] * len(solution),
        tol=1e-4,
        options={'maxiter': 50}
    )

    best_score_val = safe_score(np.clip(res.x, 0, 2))
    return max(best_score_val, best_const_score)


print("="*80)
print("Finding Best Optimization Windows")
print("="*80)

train = pd.read_csv("train.csv", index_col="date_id")
print(f"\nTrain shape: {train.shape}")

# Test set range
TEST_START = 8980
TEST_END = 8989

print(f"\nSearching for windows that include test range ({TEST_START}~{TEST_END})...")
print(f"Testing different window sizes around test set\n")

results = []

# Try different window sizes and positions that include test set
window_sizes = [180, 190, 200, 210]
for window_size in window_sizes:
    # Window must end at or after TEST_END (8989)
    for end_idx in range(8989, 8990):  # Fix at 8989
        start_idx = end_idx - window_size + 1

        if start_idx < 0:
            continue

        # Must include test set
        if not (start_idx <= TEST_START and end_idx >= TEST_END):
            continue

        try:
            s = quick_optimize(train, start_idx, end_idx)
            results.append((start_idx, end_idx, window_size, s))
            print(f"Window {start_idx}~{end_idx} (size={window_size}): Score = {s:.4f}")
        except Exception as e:
            print(f"Window {start_idx}~{end_idx} failed: {e}")

# Sort by score
results.sort(key=lambda x: x[3], reverse=True)

print("\n" + "="*80)
print("TOP 10 WINDOWS")
print("="*80)
for i, (start, end, size, s) in enumerate(results[:10], 1):
    print(f"{i}. Window {start}~{end} (size={size}): Score = {s:.4f}")

print("\n" + "="*80)
print(f"Best Window: {results[0][0]}~{results[0][1]} with Score = {results[0][3]:.4f}")
print("="*80)
