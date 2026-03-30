import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from src.data import fetch_coingecko_prices_cached, pivot_prices, DATA_DIR, load_parquet
from src.features import build_features, make_sequences, log_returns
from src.model import StrategyLSTM, mc_dropout_predict
from src.baseline import NaiveBaseline
from src.evaluate import rmse

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

def load_asset_map() -> dict:
    with open(CONFIGS_DIR / "assets.json") as f:
        return json.load(f)

def resolve_assets(assets: dict[str, float]) -> dict[str, float]:
    amap = load_asset_map()
    sym_to_cg = {v["symbol"].upper(): k for k, v in amap.items()}
    resolved = {}
    for sym, w in assets.items():
        cg_id = sym_to_cg.get(sym.upper())
        if cg_id:
            resolved[cg_id] = w
    return resolved

def predict(
    assets: dict[str, float],
    chain_id: int = 8453,
    seq_len: int = 14,
    model_path: Optional[str] = None,
) -> dict:
    cg_assets = resolve_assets(assets)
    if not cg_assets:
        return {"error": "no matching assets found"}

    coin_ids = list(cg_assets.keys())
    prices_df = fetch_coingecko_prices_cached(coin_ids, days=90)
    if prices_df.empty:
        return {"error": "no price data available"}

    piv = pivot_prices(prices_df)
    avail = [c for c in coin_ids if c in piv.columns]
    if not avail:
        return {"error": "no overlapping price data"}

    piv = piv[avail]
    feat, tgt = build_features(piv)
    if len(feat) < seq_len + 10:
        return {"error": "insufficient data for prediction"}

    X = feat.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = tgt.values.astype(np.float32)

    X_seq, y_seq = make_sequences(X, y[:, 0], seq_len)

    # naive baseline
    naive_pred = np.roll(y_seq, 1)
    naive_pred[0] = 0
    naive_rmse = rmse(y_seq, naive_pred)

    # LSTM
    mp = model_path or str(DATA_DIR / "lstm_best.pt")
    model_rmse_val = naive_rmse
    pred_mean = naive_pred[-1:]
    ci_low = pred_mean - 0.02
    ci_high = pred_mean + 0.02

    if Path(mp).exists():
        input_dim = X_seq.shape[2]
        model = StrategyLSTM(input_dim=input_dim)
        model.load_state_dict(torch.load(mp, weights_only=True))
        mean, low, high = mc_dropout_predict(model, X_seq[-1:])
        pred_mean = mean
        ci_low = low
        ci_high = high
        model_pred = mc_dropout_predict(model, X_seq)[0]
        model_rmse_val = rmse(y_seq, model_pred)

    amap = load_asset_map()
    lr = log_returns(piv)
    weights = {c: cg_assets.get(c, 0) for c in avail}
    total_w = sum(weights.values())

    attribution = {}
    for cid in avail:
        sym = amap.get(cid, {}).get("symbol", cid).upper()
        w = weights[cid] / total_w if total_w > 0 else 0
        r = lr[cid].iloc[-7:].mean() if cid in lr.columns else 0
        v = lr[cid].iloc[-7:].std() if cid in lr.columns else 0
        attribution[sym] = {
            "contribution": round(float(w * r), 6),
            "volatility": round(float(v), 6),
        }

    return {
        "strategy": {k.upper(): v for k, v in assets.items()},
        "chain_id": chain_id,
        "analysis": {
            "predicted_7d_return": round(float(pred_mean[0]), 6),
            "confidence_low": round(float(ci_low[0]), 6),
            "confidence_high": round(float(ci_high[0]), 6),
            "baseline_7d_return": round(float(naive_pred[-1]), 6),
            "model_rmse": round(float(model_rmse_val), 6),
            "baseline_rmse": round(float(naive_rmse), 6),
            "beats_baseline": float(model_rmse_val) < float(naive_rmse),
        },
        "attribution": attribution,
        "data_range": {
            "from": str(piv.index.min().date()),
            "to": str(piv.index.max().date()),
        },
        "model_version": "lstm-v0.1",
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="WETH:0.4,USDC:0.3,AAVE:0.3")
    parser.add_argument("--chain", type=int, default=8453)
    args = parser.parse_args()

    asset_dict = {}
    for pair in args.assets.split(","):
        sym, w = pair.split(":")
        asset_dict[sym.strip()] = float(w.strip())

    result = predict(asset_dict, args.chain)
    print(json.dumps(result, indent=2))
