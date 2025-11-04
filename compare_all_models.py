"""
All Models Comparison Script
Based on hull-starter-notebook_17.ipynb

Implements all 7 models and compares their performance on validation set.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# Constants
MIN_INVESTMENT = 0
MAX_INVESTMENT = 2


def ScoreMetric(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = '') -> float:
    """Calculate volatility-adjusted Sharpe ratio."""
    solut = solution.copy()
    solut['position'] = submission['prediction'].values

    if solut['position'].max() > MAX_INVESTMENT:
        raise ValueError(f'Position of {solut["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solut['position'].min() < MIN_INVESTMENT:
        raise ValueError(f'Position of {solut["position"].min()} below minimum of {MIN_INVESTMENT}')

    solut['strategy_returns'] = solut['risk_free_rate'] * (1 - solut['position']) + solut['forward_returns'] * solut['position']

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

    # Market volatility
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


# ==============================================================================
# MODEL 1: Binary strategy based on forward_returns sign
# ==============================================================================
class Model1:
    """Predict MAX if forward_returns > 0, else MIN (cheating model)"""
    def __init__(self):
        self.name = "Model_1: Binary (forward_returns > 0)"

    def fit(self, train_df):
        # No training needed - uses ground truth
        self.true_targets = dict(zip(train_df['date_id'], train_df['forward_returns']))
        return self

    def predict(self, valid_df):
        preds = []
        for date_id, fwd_ret in zip(valid_df['date_id'], valid_df['forward_returns']):
            pred = MAX_INVESTMENT if fwd_ret > 0 else MIN_INVESTMENT
            preds.append(pred)
        return np.array(preds)


# ==============================================================================
# MODEL 2: Market excess returns scaled
# ==============================================================================
class Model2:
    """Use market_forward_excess_returns * 400 + 1, clipped to [0, 2]"""
    def __init__(self):
        self.name = "Model_2: Market excess returns scaled"
        self.signal_multiplier = 400.0

    def fit(self, train_df):
        # No training needed
        return self

    def predict(self, valid_df):
        if 'market_forward_excess_returns' not in valid_df.columns:
            print("WARNING: market_forward_excess_returns not found, using 0")
            return np.full(len(valid_df), 1.0)

        raw_pred = valid_df['market_forward_excess_returns'].values
        pred = np.clip(raw_pred * self.signal_multiplier + 1, MIN_INVESTMENT, MAX_INVESTMENT)
        return pred


# ==============================================================================
# MODEL 3: Stacking ensemble (CatBoost, XGBoost, LGBM, RF, ET, GB)
# ==============================================================================
class Model3:
    """Stacking regressor with 6 base models"""
    def __init__(self):
        self.name = "Model_3: Stacking (6 models)"
        self.main_features = ['E1','E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19',
                              'E2', 'E20', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9',
                              "S2", "P9", "S1", "S5", "I2", "P8", "P10", "P12", "P13"]

    def fit(self, train_df):
        print(f"  Training {self.name}...")

        # Prepare data
        train_clean = train_df.dropna()
        X = train_clean[self.main_features].fillna(0)
        y = train_clean['forward_returns']

        # Define models
        catboost_params = {'iterations': 3000, 'learning_rate': 0.01, 'depth': 6, 'l2_leaf_reg': 5.0,
                          'min_child_samples': 100, 'colsample_bylevel': 0.7, 'od_wait': 100,
                          'random_state': 42, 'od_type': 'Iter', 'bootstrap_type': 'Bayesian',
                          'grow_policy': 'Depthwise', 'logging_level': 'Silent', 'loss_function': 'MultiRMSE'}

        xgb_params = {"n_estimators": 1500, "learning_rate": 0.05, "max_depth": 6,
                     "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 1.0,
                     "reg_lambda": 1.0, "random_state": 42}

        lgbm_params = {"n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 50,
                      "max_depth": 8, "reg_alpha": 1.0, "reg_lambda": 1.0,
                      "random_state": 42, 'verbosity': -1}

        rf_params = {'n_estimators': 100, 'min_samples_split': 5, 'max_depth': 15,
                    'min_samples_leaf': 3, 'max_features': 'sqrt', 'random_state': 42}

        et_params = {'n_estimators': 100, 'min_samples_split': 5, 'max_depth': 12,
                    'min_samples_leaf': 3, 'max_features': 'sqrt', 'random_state': 42}

        gb_params = {"learning_rate": 0.1, "min_samples_split": 500, "min_samples_leaf": 50,
                    "max_depth": 8, "max_features": 'sqrt', "subsample": 0.8, "random_state": 10}

        estimators = [
            ('CatBoost', CatBoostRegressor(**catboost_params)),
            ('XGBoost', XGBRegressor(**xgb_params)),
            ('LGBM', LGBMRegressor(**lgbm_params)),
            ('RandomForest', RandomForestRegressor(**rf_params)),
            ('ExtraTrees', ExtraTreesRegressor(**et_params)),
            ('GBRegressor', GradientBoostingRegressor(**gb_params))
        ]

        self.model = StackingRegressor(estimators, final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]), cv=3)
        self.model.fit(X, y)
        print(f"  {self.name} training complete")
        return self

    def predict(self, valid_df):
        X = valid_df[self.main_features].fillna(0)
        raw_pred = self.model.predict(X)
        return np.clip(raw_pred, MIN_INVESTMENT, MAX_INVESTMENT)


# ==============================================================================
# MODEL 4: Conditional exposure (alpha=0.80007 if forward_returns > 0)
# ==============================================================================
class Model4:
    """If forward_returns > 0: predict 0.80007, else 0"""
    def __init__(self):
        self.name = "Model_4: Conditional exposure (0.80007)"
        self.alpha = 0.80007

    def fit(self, train_df):
        return self

    def predict(self, valid_df):
        preds = np.where(valid_df['forward_returns'].values > 0, self.alpha, 0.0)
        return preds


# ==============================================================================
# MODEL 5: Conditional exposure with threshold
# ==============================================================================
class Model5:
    """If forward_returns > threshold: predict 0.60013, else 0"""
    def __init__(self):
        self.name = "Model_5: Conditional exposure (0.60013, threshold)"
        self.alpha = 0.6001322487531852
        self.tau = 9.437170708744412e-05

    def fit(self, train_df):
        return self

    def predict(self, valid_df):
        preds = np.where(valid_df['forward_returns'].values > self.tau, self.alpha, 0.0)
        return preds


# ==============================================================================
# MODEL 6: Constant 0.09 if forward_returns > 0
# ==============================================================================
class Model6:
    """If forward_returns > 0: predict 0.09, else 0"""
    def __init__(self):
        self.name = "Model_6: Constant 0.09"

    def fit(self, train_df):
        return self

    def predict(self, valid_df):
        preds = np.where(valid_df['forward_returns'].values > 0, 0.09, 0.0)
        return preds


# ==============================================================================
# MODEL 7: Powell optimization on last 180 days
# ==============================================================================
class Model7:
    """Direct optimization of metric on last 180 days"""
    def __init__(self, window_size=180):
        self.name = f"Model_7: Powell optimization (window={window_size})"
        self.window_size = window_size
        self.opt_preds = None

    def fit(self, train_df):
        print(f"  Training {self.name}...")
        train_subset = train_df.iloc[-self.window_size:].copy()

        def objective_function(x):
            solution = train_subset.copy()
            submission = pd.DataFrame({'prediction': x.clip(0, 2)}, index=solution.index)
            return -ScoreMetric(solution, submission, '')

        x0 = np.full(self.window_size, 0.05)
        result = minimize(objective_function, x0, method='Powell', bounds=Bounds(lb=0, ub=2), tol=1e-8)

        self.opt_preds = result.x
        self.train_score = -result.fun
        print(f"  {self.name} optimization complete - Train score: {self.train_score:.6f}")
        return self

    def predict(self, valid_df):
        # Model 7 cannot generalize - use mean of optimized values
        mean_pred = np.mean(self.opt_preds)
        return np.full(len(valid_df), mean_pred)


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("="*80)
    print("All Models Comparison")
    print("="*80)

    # Load data
    train_df = pd.read_csv('train_90.csv')
    valid_df = pd.read_csv('valid_10.csv')

    print(f"\nTrain size: {len(train_df)}")
    print(f"Valid size: {len(valid_df)}")

    # Initialize models
    models = [
        Model1(),
        Model2(),
        Model3(),
        Model4(),
        Model5(),
        Model6(),
        Model7(window_size=180)
    ]

    results = []

    # Train and evaluate each model
    for i, model in enumerate(models, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(models)}] {model.name}")
        print(f"{'='*80}")

        try:
            # Train
            model.fit(train_df)

            # Predict
            preds = model.predict(valid_df)

            # Evaluate
            submission = pd.DataFrame({'prediction': preds})
            score = ScoreMetric(valid_df, submission, '')

            results.append({
                'Model': model.name,
                'Score': score,
                'Mean Position': preds.mean(),
                'Std Position': preds.std(),
                'Min Position': preds.min(),
                'Max Position': preds.max()
            })

            print(f"\n✓ Validation Score: {score:.6f}")
            print(f"  Mean position: {preds.mean():.4f}")
            print(f"  Std position: {preds.std():.4f}")
            print(f"  Range: [{preds.min():.4f}, {preds.max():.4f}]")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            results.append({
                'Model': model.name,
                'Score': 'ERROR',
                'Mean Position': 'N/A',
                'Std Position': 'N/A',
                'Min Position': 'N/A',
                'Max Position': 'N/A'
            })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("NOTE:")
    print("- Models 1, 4, 5, 6 use ground truth (cheating) - not realistic")
    print("- Model 2 uses market_forward_excess_returns (may not generalize)")
    print("- Model 3 is the only truly predictive ML model")
    print("- Model 7 cannot generalize to new data (using mean of optimized values)")
    print(f"{'='*80}")
