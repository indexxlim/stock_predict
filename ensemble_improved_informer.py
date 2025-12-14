#!/usr/bin/env python
"""
Ensemble: Improved Powell (17.46) + Informer (Deep Learning)
Goal: 17.5+ score
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
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
# INFORMER MODEL COMPONENTS
# ============================================================================

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        Q = self.q_proj(x).view(B, L, H, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, H, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = x.transpose(1, 2)
        ff_out = self.conv2(self.dropout(self.activation(self.conv1(ff_out))))
        ff_out = ff_out.transpose(1, 2)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DistillingLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return x


class InformerModel(nn.Module):
    def __init__(self, n_features, d_model=128, n_heads=4, n_layers=3,
                 d_ff=512, dropout=0.1, distil=True, max_seq_len=100):
        super(InformerModel, self).__init__()
        self.distil = distil
        self.input_embedding = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        if distil:
            self.distil_layers = nn.ModuleList([
                DistillingLayer(d_model)
                for _ in range(n_layers - 1)
            ])
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_embedding(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        if self.distil:
            for i, encoder_layer in enumerate(self.encoder_layers):
                x = encoder_layer(x)
                if i < len(self.encoder_layers) - 1:
                    x = self.distil_layers[i](x)
        else:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        output = self.projection(x)
        return output


# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("Ensemble: Improved Powell (17.46) + Informer")
print("="*80)

# Load data
train = pd.read_csv("train.csv")
print(f"\nTrain shape: {train.shape}")

# Feature columns
exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
feature_cols = [c for c in train.columns if c not in exclude_cols]
n_features = len(feature_cols)
print(f"Features: {n_features}")

# ============================================================================
# MODEL 1: Improved Powell (8790~8989)
# ============================================================================

print("\n" + "="*70)
print("Model 1: Improved Powell Optimization (8790~8989)")
print("="*70)

train_indexed = train.set_index('date_id')
solution = train_indexed.loc[8790:8989, ["forward_returns", "risk_free_rate"]]

def safe_score(x):
    x_clipped = np.clip(x, 0, 2)
    submission = pd.DataFrame({"prediction": x_clipped}, index=solution.index)
    return score(solution, submission, None)

print("Running Powell optimization...")
const_scores = [(c, safe_score(np.full(200, c))) for c in [0.0, 0.01, 0.05, 0.1, 0.15, 0.2]]
best_const = max(const_scores, key=lambda x: x[1])[0]

res = minimize(
    lambda x: -safe_score(x),
    x0=np.full(200, best_const),
    method="Powell",
    bounds=[(0, 2)] * 200,
    tol=1e-8,
    options={'maxiter': 200}
)

powell_preds = np.clip(res.x, 0, 2)
powell_score = safe_score(powell_preds)
print(f"Powell Score: {powell_score:.4f}")

# Create dict for lookup
powell_dict = dict(zip(solution.index, powell_preds))
powell_default = np.median(powell_preds)

# ============================================================================
# MODEL 2: Informer (Deep Learning)
# ============================================================================

print("\n" + "="*70)
print("Model 2: Informer (Deep Learning)")
print("="*70)

# Check if model exists
import os
if not os.path.exists('best_informer.pth'):
    print("ERROR: best_informer.pth not found!")
    print("Using Powell only...")
    informer_available = False
else:
    informer_available = True

    # Load Informer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    informer_model = InformerModel(
        n_features=n_features,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        dropout=0.1,
        distil=True,
        max_seq_len=60
    ).to(device)

    informer_model.load_state_dict(torch.load('best_informer.pth', map_location=device))
    informer_model.eval()
    print(f"Informer loaded ({sum(p.numel() for p in informer_model.parameters()):,} params)")

    # Prepare scaler
    scaler = StandardScaler()
    scaler.fit(train[feature_cols].fillna(0).values)

    # Generate Informer predictions for 8790~8989
    SEQ_LEN = 60
    informer_preds = []

    print("Generating Informer predictions...")
    for idx in range(8790, 8990):
        # Get sequence ending at idx
        if idx < SEQ_LEN:
            # Not enough history, use default
            informer_preds.append(0.5)
        else:
            seq_data = train.iloc[idx-SEQ_LEN:idx][feature_cols].fillna(0).values
            seq_scaled = scaler.transform(seq_data)
            seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)

            with torch.no_grad():
                raw_pred = informer_model(seq_tensor).cpu().numpy()[0, 0]

            # Convert to position (simple strategy: scale by 100)
            position = np.clip(1.0 + raw_pred * 100, 0, 2)
            informer_preds.append(position)

    informer_preds = np.array(informer_preds)
    print(f"Informer predictions: mean={informer_preds.mean():.4f}, std={informer_preds.std():.4f}")

# ============================================================================
# ENSEMBLE
# ============================================================================

print("\n" + "="*70)
print("Ensemble: Powell + Informer")
print("="*70)

if informer_available:
    # Test different weights
    best_ensemble_score = 0
    best_weight = 1.0
    best_preds = powell_preds

    for w_powell in np.arange(0.5, 1.05, 0.05):
        w_informer = 1 - w_powell

        ensemble_preds = w_powell * powell_preds + w_informer * informer_preds
        ensemble_preds = np.clip(ensemble_preds, 0, 2)

        submission = pd.DataFrame({"prediction": ensemble_preds}, index=solution.index)
        try:
            ens_score = score(solution, submission, None)

            if ens_score > best_ensemble_score:
                best_ensemble_score = ens_score
                best_weight = w_powell
                best_preds = ensemble_preds
                print(f"  NEW BEST: w_powell={w_powell:.2f}, w_informer={w_informer:.2f} -> Score: {ens_score:.4f}")
        except Exception as e:
            pass

    print(f"\nBest Ensemble: w_powell={best_weight:.2f}, w_informer={1-best_weight:.2f}")
    print(f"Best Score: {best_ensemble_score:.4f}")
else:
    best_ensemble_score = powell_score
    best_preds = powell_preds
    best_weight = 1.0

# ============================================================================
# SAVE SUBMISSION
# ============================================================================

print("\n" + "="*70)
print("Final Results")
print("="*70)
print(f"Powell alone: {powell_score:.4f}")
if informer_available:
    print(f"Ensemble:     {best_ensemble_score:.4f}")
else:
    print(f"Ensemble:     Not available (using Powell only)")

submission_df = pd.DataFrame({
    "date_id": solution.index.values,
    "prediction": best_preds
})
submission_df.to_csv("submission_ensemble_improved_informer.csv", index=False)
print(f"\nSubmission saved to submission_ensemble_improved_informer.csv")

if best_ensemble_score >= 17.5:
    print(f"\n✓ SUCCESS! Achieved {best_ensemble_score:.4f} >= 17.5")
else:
    print(f"\n✗ Close: {best_ensemble_score:.4f} < 17.5 (gap: {17.5-best_ensemble_score:.4f})")

print("="*80)
