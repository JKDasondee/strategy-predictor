import numpy as np
import torch
import pytest
from src.nbeats import NBEATS, NBEATSBlock, train_nbeats, mc_dropout_nbeats
from src.model import TrainConfig

def test_block_forward():
    b = NBEATSBlock(in_dim=70, theta_dim=16, hidden=64)
    x = torch.randn(4, 70)
    back, fore = b(x)
    assert back.shape == (4, 70)
    assert fore.shape == (4,)

def test_nbeats_forward():
    m = NBEATS(input_dim=5, seq_len=14, n_stacks=2, n_blocks=3, hidden=64)
    x = torch.randn(4, 14, 5)
    out = m(x)
    assert out.shape == (4,)

def test_nbeats_train():
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(100, 14, 5).astype(np.float32)
    y = np.sin(np.arange(100)).astype(np.float32)

    m = NBEATS(input_dim=5, seq_len=14, n_stacks=2, n_blocks=2, hidden=32)
    cfg = TrainConfig(epochs=30, lr=1e-3, batch_size=16, patience=30)
    result = train_nbeats(m, X[:80], y[:80], X[80:], y[80:], cfg)
    assert result["best_val_loss"] < 2.0

def test_mc_dropout_nbeats():
    m = NBEATS(input_dim=5, seq_len=14, n_stacks=2, n_blocks=2, hidden=32)
    X = np.random.randn(10, 14, 5).astype(np.float32)
    mean, low, high = mc_dropout_nbeats(m, X, n_samples=10)
    assert mean.shape == (10,)
    assert (high >= low).all()
