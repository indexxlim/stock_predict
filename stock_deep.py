"""
Deep Learning Models for Stock Prediction
1. Chronos (TimeLLM-based, Amazon)
2. Informer (Long Sequence Transformer)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# DATASET
# ==============================================================================

class StockSequenceDataset(Dataset):
    """Dataset for sequence-based models"""
    def __init__(self, data, seq_len=60, feature_cols=None, target_col='forward_returns'):
        """
        Args:
            data: DataFrame
            seq_len: Sequence length for lookback
            feature_cols: List of feature columns (None = auto-detect)
            target_col: Target column name
        """
        self.seq_len = seq_len
        self.target_col = target_col

        # Auto-detect features if not provided
        if feature_cols is None:
            exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
            self.feature_cols = [c for c in data.columns if c not in exclude_cols]
        else:
            self.feature_cols = feature_cols

        # Prepare data
        self.X = data[self.feature_cols].fillna(0).values.astype(np.float32)
        self.y = data[self.target_col].values.astype(np.float32)

        # Normalize features
        self.scaler_X = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.X)

        print(f"  Dataset: {len(self.X)} samples, {len(self.feature_cols)} features, seq_len={seq_len}")

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # Get sequence [idx : idx+seq_len]
        X_seq = self.X[idx:idx+self.seq_len]
        y = self.y[idx+self.seq_len]  # Predict next day

        return torch.FloatTensor(X_seq), torch.FloatTensor([y])


# ==============================================================================
# MODEL 1: CHRONOS (Simplified Transformer-based Time Series)
# ==============================================================================

class ChronosModel(nn.Module):
    """
    Simplified version inspired by Chronos
    Uses Transformer encoder with learnable embeddings
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
        """
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            output: [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape

        # Project input to d_model dimension
        x = self.input_projection(x)  # [B, L, d_model]

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Transformer encoding
        x = self.transformer(x)  # [B, L, d_model]

        # Global average pooling over sequence
        x = x.mean(dim=1)  # [B, d_model]

        # Predict
        output = self.output_head(x)  # [B, 1]

        return output


# ==============================================================================
# MODEL 2: INFORMER
# ==============================================================================

class ProbSparseAttention(nn.Module):
    """Simplified ProbSparse Self-Attention"""
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

        # Project
        Q = self.q_proj(x).view(B, L, H, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.k_proj(x).view(B, L, H, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, H, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, L, d_k]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]

        return self.out_proj(out)


class InformerEncoderLayer(nn.Module):
    """Informer Encoder Layer with distilling"""
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
        # Self-attention
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = x.transpose(1, 2)  # [B, D, L]
        ff_out = self.conv2(self.dropout(self.activation(self.conv1(ff_out))))
        ff_out = ff_out.transpose(1, 2)  # [B, L, D]

        x = self.norm2(x + self.dropout(ff_out))

        return x


class DistillingLayer(nn.Module):
    """Distilling operation to reduce sequence length"""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # [B, L', D]
        return x


class InformerModel(nn.Module):
    """
    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    Paper: https://arxiv.org/abs/2012.07436
    """
    def __init__(self,
                 n_features,
                 d_model=128,
                 n_heads=4,
                 n_layers=3,
                 d_ff=512,
                 dropout=0.1,
                 distil=True,
                 max_seq_len=100):
        super(InformerModel, self).__init__()

        self.distil = distil

        # Input embedding
        self.input_embedding = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Distilling layers (optional)
        if distil:
            self.distil_layers = nn.ModuleList([
                DistillingLayer(d_model)
                for _ in range(n_layers - 1)
            ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            output: [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape

        # Embedding
        x = self.input_embedding(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Encoder with distilling
        if self.distil:
            for i, encoder_layer in enumerate(self.encoder_layers):
                x = encoder_layer(x)
                if i < len(self.encoder_layers) - 1:
                    x = self.distil_layers[i](x)
        else:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)

        x = self.norm(x)

        # Global pooling
        x = x.mean(dim=1)  # [B, d_model]

        # Output
        output = self.projection(x)

        return output


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_model(model, train_loader, valid_loader, config, model_name="Model"):
    """
    Train deep learning model

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        valid_loader: Validation DataLoader
        config: Training configuration dict
        model_name: Name for logging

    Returns:
        Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.get('lr_patience', 5)
    )

    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{config['epochs']:3d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{model_name.lower().replace(" ", "_")}.pth')
            print(f"  → New best model saved (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(f'best_{model_name.lower().replace(" ", "_")}.pth'))
    print(f"\n✓ Training complete. Best Val Loss: {best_val_loss:.6f}")

    return model


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================

def predict_with_model(model, dataset, batch_size=64):
    """Make predictions with trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            predictions.extend(output.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())

    return np.array(predictions), np.array(actuals)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Deep Learning Models for Stock Prediction")
    print("="*80)

    # Load data
    train_df = pd.read_csv('train_90.csv')
    valid_df = pd.read_csv('valid_10.csv')

    print(f"\nTrain size: {len(train_df)}")
    print(f"Valid size: {len(valid_df)}")

    # Configuration
    config = {
        'seq_len': 60,           # Look back 60 days
        'batch_size': 32,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'd_ff': 512,
        'dropout': 0.1,
        'lr': 0.0001,
        'weight_decay': 0.01,
        'epochs': 100,
        'patience': 15,
        'lr_patience': 5
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create datasets
    print(f"\n{'='*80}")
    print("Creating datasets...")
    print(f"{'='*80}")

    train_dataset = StockSequenceDataset(train_df, seq_len=config['seq_len'])
    valid_dataset = StockSequenceDataset(valid_df, seq_len=config['seq_len'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

    n_features = len(train_dataset.feature_cols)

    # ==============================================================================
    # MODEL 1: CHRONOS
    # ==============================================================================

    print(f"\n{'='*80}")
    print("MODEL 1: CHRONOS")
    print(f"{'='*80}")

    chronos_model = ChronosModel(
        n_features=n_features,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=config['seq_len']
    )

    chronos_trained = train_model(chronos_model, train_loader, valid_loader, config, "Chronos")

    # Predict
    chronos_preds, chronos_actuals = predict_with_model(chronos_trained, valid_dataset, batch_size=64)
    chronos_mse = np.mean((chronos_preds - chronos_actuals) ** 2)
    chronos_mae = np.mean(np.abs(chronos_preds - chronos_actuals))

    print(f"\nChronos Validation Results:")
    print(f"  MSE: {chronos_mse:.6f}")
    print(f"  MAE: {chronos_mae:.6f}")
    print(f"  Pred mean: {chronos_preds.mean():.6f}")
    print(f"  Pred std: {chronos_preds.std():.6f}")

    # ==============================================================================
    # MODEL 2: INFORMER
    # ==============================================================================

    print(f"\n{'='*80}")
    print("MODEL 2: INFORMER")
    print(f"{'='*80}")

    informer_model = InformerModel(
        n_features=n_features,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        distil=True,
        max_seq_len=config['seq_len']
    )

    informer_trained = train_model(informer_model, train_loader, valid_loader, config, "Informer")

    # Predict
    informer_preds, informer_actuals = predict_with_model(informer_trained, valid_dataset, batch_size=64)
    informer_mse = np.mean((informer_preds - informer_actuals) ** 2)
    informer_mae = np.mean(np.abs(informer_preds - informer_actuals))

    print(f"\nInformer Validation Results:")
    print(f"  MSE: {informer_mse:.6f}")
    print(f"  MAE: {informer_mae:.6f}")
    print(f"  Pred mean: {informer_preds.mean():.6f}")
    print(f"  Pred std: {informer_preds.std():.6f}")

    # ==============================================================================
    # COMPARISON
    # ==============================================================================

    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")

    comparison = pd.DataFrame({
        'Model': ['Chronos', 'Informer'],
        'MSE': [chronos_mse, informer_mse],
        'MAE': [chronos_mae, informer_mae],
        'Pred Mean': [chronos_preds.mean(), informer_preds.mean()],
        'Pred Std': [chronos_preds.std(), informer_preds.std()]
    })

    print(comparison.to_string(index=False))

    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"Models saved:")
    print(f"  - best_chronos.pth")
    print(f"  - best_informer.pth")
