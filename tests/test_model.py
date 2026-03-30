import numpy as np
import torch
import pytest
from src.model import StrategyLSTM, SequenceDataset, train, mc_dropout_predict, TrainConfig

def test_lstm_forward():
    m = StrategyLSTM(input_dim=10, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 14, 10)
    out = m(x)
    assert out.shape == (4,)

def test_lstm_train_overfit():
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(100, 14, 5).astype(np.float32)
    y = np.sin(np.arange(100)).astype(np.float32)

    m = StrategyLSTM(input_dim=5, hidden_dim=16, num_layers=1, dropout=0.0)
    cfg = TrainConfig(epochs=50, lr=1e-2, batch_size=16, patience=50)
    result = train(m, X[:80], y[:80], X[80:], y[80:], cfg)
    assert result["best_val_loss"] < 1.0

def test_mc_dropout():
    m = StrategyLSTM(input_dim=5, hidden_dim=16)
    X = np.random.randn(10, 14, 5).astype(np.float32)
    mean, low, high = mc_dropout_predict(m, X, n_samples=20)
    assert mean.shape == (10,)
    assert (high >= low).all()

def test_sequence_dataset():
    X = np.random.randn(50, 14, 10).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    ds = SequenceDataset(X, y)
    assert len(ds) == 50
    xb, yb = ds[0]
    assert xb.shape == (14, 10)
