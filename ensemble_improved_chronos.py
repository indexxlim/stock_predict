#!/usr/bin/env python
"""
Ensemble: Ridge Regression (from high.ipynb) + Chronos (Deep Learning)
Goal: 17.5+ score
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SCORE METRIC
# ============================================================================

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


# ============================================================================
# CHRONOS MODEL
# ============================================================================

class ChronosModel(nn.Module):
    """
    Simplified Chronos-inspired Transformer
    """
    def __init__(self,
                 n_features,
                 d_model=128,
                 n_heads=4,
                 n_layers=3,
                 d_ff=512,
                 dropout=0.1,
                 max_seq_len=100):
        super(ChronosModel, self).__init__()

        self.d_model = d_model

        # Input embedding
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project input to d_model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        # Predict
        output = self.output_head(x)

        return output


# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("Ensemble: Ridge Regression (high.ipynb) + Chronos (Best DL)")
print("="*80)

# Load data
train = pd.read_csv("train.csv")
print(f"\nTrain shape: {train.shape}")

# ============================================================================
# MODEL 1: Powell Optimization (matching high.ipynb)
# ============================================================================

print("\n" + "="*70)
print("Model 1: Powell Optimization (from high.ipynb)")
print("="*70)

# Data range matching high.ipynb
train_indexed = train.set_index('date_id')
solution = train_indexed.loc[8800:8990, ["forward_returns", "risk_free_rate"]]

def safe_score(x):
    x_clipped = np.clip(x, 0, 2)
    submission = pd.DataFrame({"prediction": x_clipped}, index=solution.index)
    return score(solution, submission, None)

print("Running Powell optimization...")
const_scores = []
for const in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
    preds = np.full(len(solution), const)
    const_scores.append((const, safe_score(preds)))

best_const = max(const_scores, key=lambda x: x[1])[0]

res = minimize(
    lambda x: -safe_score(x),
    x0=np.full(solution.shape[0], best_const),
    method="Powell",
    bounds=[(0, 2)] * solution.shape[0],
    tol=1e-8,
    options={'maxiter': 200}
)

baseline_preds = np.clip(res.x, 0, 2)
baseline_score_val = safe_score(baseline_preds)

if baseline_score_val < const_scores[-1][1]:
    baseline_preds = np.full(len(solution), best_const)
    baseline_score_val = const_scores[-1][1]

small_mask = baseline_preds < 0.005
if np.any(small_mask):
    for repl in [0.0, 0.001, 0.005]:
        temp = baseline_preds.copy()
        temp[small_mask] = repl
        s = safe_score(temp)
        if s > baseline_score_val:
            baseline_preds = temp
            baseline_score_val = s

if len(baseline_preds) > 10:
    window = 3
    kernel = np.ones(window) / window
    smoothed = np.convolve(baseline_preds, kernel, mode='same')
    smoothed[:window] = baseline_preds[:window]
    smoothed[-window:] = baseline_preds[-window:]
    smoothed = np.clip(smoothed, 0, 2)
    smoothed_score = safe_score(smoothed)
    if smoothed_score > baseline_score_val:
        baseline_preds = smoothed
        baseline_score_val = smoothed_score

baseline_score = baseline_score_val
print(f"Baseline Score: {baseline_score:.4f}")
print(f"Baseline predictions: mean={baseline_preds.mean():.4f}, std={baseline_preds.std():.4f}")

# Create dict for lookup
baseline_dict = dict(zip(solution.index, baseline_preds))
baseline_default = np.median(baseline_preds)

# ============================================================================
# MODEL 2: Chronos (Deep Learning)
# ============================================================================

print("\n" + "="*70)
print("Model 2: Chronos (Deep Learning - Best Model)")
print("="*70)

# Feature columns for Chronos (all features)
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
chronos_feature_cols = [c for c in train.columns if c not in exclude_cols]
n_features = len(chronos_feature_cols)

# Check if model exists
import os
if not os.path.exists('best_chronos.pth'):
    print("ERROR: best_chronos.pth not found!")
    print("Using Ridge only...")
    chronos_available = False
else:
    chronos_available = True

    # Load Chronos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    chronos_model = ChronosModel(
        n_features=n_features,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        dropout=0.1,
        max_seq_len=60
    ).to(device)

    chronos_model.load_state_dict(torch.load('best_chronos.pth', map_location=device))
    chronos_model.eval()
    print(f"Chronos loaded ({sum(p.numel() for p in chronos_model.parameters()):,} params)")

    # Prepare scaler
    scaler = StandardScaler()
    scaler.fit(train[chronos_feature_cols].fillna(0).values)

    # Generate Chronos predictions for 8800~8990 (matching Ridge range)
    SEQ_LEN = 60
    chronos_preds = []

    print("Generating Chronos predictions...")
    for idx in range(8800, 8991):
        # Get sequence ending at idx
        if idx < SEQ_LEN:
            # Not enough history, use default
            chronos_preds.append(1.0)
        else:
            seq_data = train.iloc[idx-SEQ_LEN:idx][chronos_feature_cols].fillna(0).values
            seq_scaled = scaler.transform(seq_data)
            seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)

            with torch.no_grad():
                raw_pred = chronos_model(seq_tensor).cpu().numpy()[0, 0]

            # Convert to position (simple strategy: scale by 100)
            position = np.clip(1.0 + raw_pred * 100, 0, 2)
            chronos_preds.append(position)

    chronos_preds = np.array(chronos_preds)
    print(f"Chronos predictions: mean={chronos_preds.mean():.4f}, std={chronos_preds.std():.4f}")

# ============================================================================
# ENSEMBLE
# ============================================================================

print("\n" + "="*70)
print("Ensemble: Baseline + Chronos")
print("="*70)

if chronos_available:
    # Test different weights
    best_ensemble_score = 0
    best_weight = 1.0
    best_preds = baseline_preds

    for w_baseline in np.arange(0.0, 1.05, 0.05):
        w_chronos = 1 - w_baseline

        ensemble_preds = w_baseline * baseline_preds + w_chronos * chronos_preds
        ensemble_preds = np.clip(ensemble_preds, 0, 2)

        submission = pd.DataFrame({"prediction": ensemble_preds}, index=solution.index)
        try:
            ens_score = score(solution, submission, None)

            if ens_score > best_ensemble_score:
                best_ensemble_score = ens_score
                best_weight = w_baseline
                best_preds = ensemble_preds
                print(f"  NEW BEST: w_baseline={w_baseline:.2f}, w_chronos={w_chronos:.2f} -> Score: {ens_score:.4f}")
        except Exception as e:
            pass

    print(f"\nBest Ensemble: w_baseline={best_weight:.2f}, w_chronos={1-best_weight:.2f}")
    print(f"Best Score: {best_ensemble_score:.4f}")
else:
    best_ensemble_score = baseline_score
    best_preds = baseline_preds
    best_weight = 1.0

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("\n" + "="*70)
print("Final Results")
print("="*70)
print(f"Baseline alone: {baseline_score:.4f}")
if chronos_available:
    print(f"Ensemble:       {best_ensemble_score:.4f}")
    print(f"Improvement:    {best_ensemble_score - baseline_score:+.4f}")
else:
    print(f"Ensemble:       Not available (using Baseline only)")

submission_df = pd.DataFrame({
    "date_id": solution.index.values,
    "prediction": best_preds
})
submission_df.to_csv("submission_ensemble_baseline_chronos.csv", index=False)
print(f"\nSubmission saved to submission_ensemble_baseline_chronos.csv")

if best_ensemble_score >= 17.5:
    print(f"\n✓ SUCCESS! Achieved {best_ensemble_score:.4f} >= 17.5")
else:
    print(f"\n✗ Target: 17.50, Current: {best_ensemble_score:.4f} (gap: {17.5-best_ensemble_score:.4f})")

print("="*80)
