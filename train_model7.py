"""
Model_7 Training and Validation Script
Based on hull-starter-notebook_17.ipynb

This script implements Powell optimization to directly maximize
the adjusted Sharpe ratio metric.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# Constants
MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


def ScoreMetric(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = ''
) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).
    This metric penalizes strategies that take on significantly more volatility
    than the underlying market.

    Returns: The calculated adjusted Sharpe ratio.
    """
    solut = solution.copy()
    solut['position'] = submission['prediction'].values

    if solut['position'].max() > MAX_INVESTMENT:
        raise ValueError(
            f'Position of {solut["position"].max()} exceeds maximum of {MAX_INVESTMENT}')

    if solut['position'].min() < MIN_INVESTMENT:
        raise ValueError(
            f'Position of {solut["position"].min()} below minimum of {MIN_INVESTMENT}')

    solut['strategy_returns'] =\
        solut['risk_free_rate'] * (1 - solut['position']) +\
        solut['forward_returns'] * solut['position']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solut['strategy_returns'] - solut['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solut)) - 1
    strategy_std = solut['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        raise ZeroDivisionError("Strategy std is zero")
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solut['forward_returns'] - solut['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solut)) - 1
    market_std = solut['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate the volatility penalty
    excess_vol =\
        max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0

    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap =\
        max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)

    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)

    return min(float(adjusted_sharpe), 1_000_000)


def train_model7(train_df: pd.DataFrame, window_size: int = 180, verbose: bool = True):
    """
    Train Model_7 using Powell optimization on the last 'window_size' days.

    Args:
        train_df: Training dataframe with columns: date_id, forward_returns, risk_free_rate
        window_size: Number of recent days to optimize on
        verbose: Print optimization progress

    Returns:
        Optimized predictions array and optimization result
    """
    # Use last window_size days for optimization
    train_subset = train_df.iloc[-window_size:].copy()

    if verbose:
        print(f"Training Model_7 on last {window_size} days")
        print(f"Date range: {train_subset['date_id'].min()} ~ {train_subset['date_id'].max()}")

    def objective_function(x):
        """Objective function to minimize (negative score to maximize)"""
        solution = train_subset.copy()
        submission = pd.DataFrame({'prediction': x.clip(0, 2)}, index=solution.index)
        return -ScoreMetric(solution, submission, '')

    # Initial guess: small positive position
    x0 = np.full(window_size, 0.05)

    if verbose:
        print(f"Starting Powell optimization...")

    # Run optimization
    result = minimize(
        objective_function,
        x0,
        method='Powell',
        bounds=Bounds(lb=0, ub=2),
        tol=1e-8
    )

    if verbose:
        print(f"\nOptimization Result:")
        print(f"  Success: {result.success}")
        print(f"  Status: {result.status}")
        print(f"  Message: {result.message}")
        print(f"  Score: {-result.fun:.6f}")
        print(f"  Function evaluations: {result.nfev}")

    return result.x, result


def evaluate_model7(valid_df: pd.DataFrame, predictions: np.ndarray, verbose: bool = True):
    """
    Evaluate Model_7 predictions on validation set.

    Args:
        valid_df: Validation dataframe
        predictions: Predicted positions for validation set
        verbose: Print evaluation details

    Returns:
        Score on validation set
    """
    if len(predictions) != len(valid_df):
        raise ValueError(f"Predictions length {len(predictions)} != validation length {len(valid_df)}")

    solution = valid_df.copy()
    submission = pd.DataFrame({'prediction': predictions}, index=solution.index)

    score = ScoreMetric(solution, submission, '')

    if verbose:
        print(f"\nValidation Score: {score:.6f}")
        print(f"Validation date range: {valid_df['date_id'].min()} ~ {valid_df['date_id'].max()}")
        print(f"Mean position: {predictions.mean():.4f}")
        print(f"Std position: {predictions.std():.4f}")
        print(f"Min position: {predictions.min():.4f}")
        print(f"Max position: {predictions.max():.4f}")

    return score


def simple_predict_strategy(valid_df: pd.DataFrame, strategy: str = 'zero'):
    """
    Generate simple baseline predictions.

    Args:
        valid_df: Validation dataframe
        strategy: 'zero', 'mean', 'forward_positive', 'constant'

    Returns:
        Predictions array
    """
    n = len(valid_df)

    if strategy == 'zero':
        return np.zeros(n)
    elif strategy == 'mean':
        return np.full(n, 1.0)  # Middle of [0, 2]
    elif strategy == 'forward_positive':
        # Predict based on forward returns sign (cheating, for comparison)
        return np.where(valid_df['forward_returns'].values > 0, MAX_INVESTMENT, MIN_INVESTMENT)
    elif strategy == 'constant':
        return np.full(n, 0.05)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Load data
    print("="*80)
    print("Model_7 Training and Validation")
    print("="*80)

    train_df = pd.read_csv('train_90.csv')
    valid_df = pd.read_csv('valid_10.csv')

    print(f"\nTrain size: {len(train_df)}")
    print(f"Valid size: {len(valid_df)}")

    # Train Model_7 on last 180 days of training data
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    opt_preds, result = train_model7(train_df, window_size=180, verbose=True)

    # For validation, we need a prediction strategy
    # Option 1: Use last optimized value
    # Option 2: Use mean of optimized values
    # Option 3: Train a simple model to predict position based on features

    # For now, let's use simple strategies as baselines
    print("\n" + "="*80)
    print("BASELINE EVALUATION")
    print("="*80)

    strategies = ['zero', 'constant', 'mean']
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        preds = simple_predict_strategy(valid_df, strategy=strategy)
        try:
            score = evaluate_model7(valid_df, preds, verbose=True)
        except Exception as e:
            print(f"Error: {e}")

    # "Cheating" strategy to see upper bound
    print(f"\n--- Strategy: forward_positive (CHEATING - for reference only) ---")
    preds = simple_predict_strategy(valid_df, strategy='forward_positive')
    try:
        score = evaluate_model7(valid_df, preds, verbose=True)
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*80)
    print("NOTE: Model_7 optimizes positions for specific training days.")
    print("It cannot generalize to new days without additional modeling.")
    print("To use Model_7 for validation, you would need to:")
    print("1. Train a ML model to predict positions based on features")
    print("2. Use a heuristic based on optimized patterns")
    print("3. Re-optimize on validation set (but this defeats the purpose)")
    print("="*80)
