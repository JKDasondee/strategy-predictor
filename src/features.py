import pandas as pd
import numpy as np
from typing import Optional

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def rolling_returns(prices: pd.DataFrame, w: int = 7) -> pd.DataFrame:
    return (prices / prices.shift(w) - 1).dropna()

def rolling_vol(returns: pd.DataFrame, w: int = 7) -> pd.DataFrame:
    return returns.rolling(w).std().dropna()

def rolling_sharpe(returns: pd.DataFrame, w: int = 30, rf: float = 0.0) -> pd.DataFrame:
    mu = returns.rolling(w).mean()
    sig = returns.rolling(w).std()
    return ((mu - rf) / sig).replace([np.inf, -np.inf], np.nan).dropna()

def rolling_drawdown(prices: pd.DataFrame, w: int = 30) -> pd.DataFrame:
    roll_max = prices.rolling(w, min_periods=1).max()
    return (prices / roll_max - 1)

def ewm_vol(returns: pd.DataFrame, span: int = 14) -> pd.DataFrame:
    return returns.ewm(span=span).std()

def portfolio_return(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [c for c in weights if c in returns.columns]
    w = np.array([weights[c] for c in cols])
    w = w / w.sum()
    return (returns[cols] * w).sum(axis=1)

def portfolio_vol(returns: pd.DataFrame, weights: dict[str, float], w: int = 7) -> pd.Series:
    pr = portfolio_return(returns, weights)
    return pr.rolling(w).std()

def make_target(prices: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    return (prices.shift(-horizon) / prices - 1).dropna()

def build_features(
    prices: pd.DataFrame,
    lookback: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr = log_returns(prices)
    r7 = rolling_returns(prices, 7)
    v7 = rolling_vol(lr, 7)
    v14 = rolling_vol(lr, 14)
    sh = rolling_sharpe(lr, 30)
    dd = rolling_drawdown(prices, 30)
    ev = ewm_vol(lr, 14)

    feat = pd.concat({
        "lr": lr, "r7": r7, "v7": v7, "v14": v14,
        "sharpe": sh, "dd": dd, "ewm_v": ev,
    }, axis=1).dropna()

    tgt = make_target(prices, 7)
    idx = feat.index.intersection(tgt.index)
    return feat.loc[idx], tgt.loc[idx]

def train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_pct: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(X)
    split = int(n * (1 - test_pct))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)
