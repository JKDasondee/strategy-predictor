import numpy as np
import pandas as pd
import pytest
from src.features import (
    log_returns, rolling_returns, rolling_vol, portfolio_return,
    make_target, build_features, train_test_split, make_sequences,
)

def _price_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02)),
        "B": 50 * np.exp(np.cumsum(np.random.randn(n) * 0.01)),
    }, index=dates)

def test_log_returns_shape():
    p = _price_df()
    lr = log_returns(p)
    assert lr.shape == (99, 2)

def test_rolling_returns():
    p = _price_df()
    r = rolling_returns(p, 7)
    assert len(r) == 93
    assert not r.isna().any().any()

def test_rolling_vol():
    p = _price_df()
    lr = log_returns(p)
    v = rolling_vol(lr, 7)
    assert (v >= 0).all().all()

def test_portfolio_return():
    p = _price_df()
    lr = log_returns(p)
    pr = portfolio_return(lr, {"A": 0.6, "B": 0.4})
    assert len(pr) == 99
    assert isinstance(pr, pd.Series)

def test_make_target():
    p = _price_df()
    t = make_target(p, 7)
    assert len(t) == 93

def test_build_features():
    p = _price_df()
    feat, tgt = build_features(p)
    assert len(feat) == len(tgt)
    assert feat.shape[1] > 0

def test_train_test_split():
    p = _price_df()
    feat, tgt = build_features(p)
    Xtr, Xte, ytr, yte = train_test_split(feat, tgt, 0.2)
    assert len(Xtr) + len(Xte) == len(feat)
    assert Xtr.index[-1] < Xte.index[0]

def test_make_sequences():
    X = np.random.randn(50, 10).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    xs, ys = make_sequences(X, y, seq_len=14)
    assert xs.shape == (36, 14, 10)
    assert ys.shape == (36,)
