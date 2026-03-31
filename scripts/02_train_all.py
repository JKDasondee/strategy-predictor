"""Train all models on 365d data including neural models.
Run: python scripts/02_train_all.py
Expected time: ~2-5 min depending on GPU
"""
import sys, time, json
sys.path.insert(0, ".")

import numpy as np
import torch
import pandas as pd
from pathlib import Path

from src.data import load_parquet, pivot_prices, DATA_DIR
from src.features import build_features, train_test_split, make_sequences, log_returns
from src.baseline import NaiveBaseline, MeanReversion, LinearBaseline, BaselineResult
from src.model import StrategyLSTM, train as train_lstm, mc_dropout_predict, TrainConfig
from src.nbeats import NBEATS, train_nbeats, mc_dropout_nbeats
from src.tree import XGBoostModel, LightGBMModel
from src.ensemble import WeightedEnsemble, StackedEnsemble
from src.evaluate import rmse, mape, mae, r_squared, directional_accuracy, comparison_table

SEQ_LEN = 14
ASSETS_TO_EVAL = ["ethereum", "usd-coin", "aave", "chainlink", "uniswap"]

def train_for_asset(piv, target, results_all):
    print(f"\n--- {target} ---")
    feat, tgt = build_features(piv)
    if target not in tgt.columns:
        print(f"  skipped (not in data)")
        return

    Xtr, Xte, ytr, yte = train_test_split(feat, tgt, 0.2)
    print(f"  train: {len(Xtr)}, test: {len(Xte)}, features: {feat.shape[1]}")

    results = []

    # baselines
    for cls in [NaiveBaseline, MeanReversion, LinearBaseline]:
        m = cls()
        r = m.predict(Xtr, Xte, ytr, yte, target)
        results.append(r)

    # trees
    for cls in [XGBoostModel, LightGBMModel]:
        try:
            m = cls()
            r = m.predict(Xtr, Xte, ytr, yte, target)
            results.append(r)
        except Exception as e:
            print(f"  {cls.__name__} failed: {e}")

    # neural
    X_arr = feat.values.astype(np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0)
    y_arr = tgt[target].values.astype(np.float32)
    split = int(len(X_arr) * 0.8)

    X_seq_tr, y_seq_tr = make_sequences(X_arr[:split], y_arr[:split], SEQ_LEN)
    X_seq_te, y_seq_te = make_sequences(X_arr[split:], y_arr[split:], SEQ_LEN)

    if len(X_seq_tr) > 20 and len(X_seq_te) > 3:
        input_dim = X_seq_tr.shape[2]
        bs = min(32, len(X_seq_tr))

        # LSTM
        torch.manual_seed(42)
        lstm = StrategyLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
        cfg = TrainConfig(epochs=200, lr=1e-3, batch_size=bs, patience=25)
        res = train_lstm(lstm, X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, cfg)
        lstm.eval()
        with torch.no_grad():
            pred = lstm(torch.FloatTensor(X_seq_te)).numpy()
        results.append(BaselineResult("lstm", y_seq_te, pred))
        if target == "ethereum":
            torch.save(lstm.state_dict(), str(DATA_DIR / "lstm_best.pt"))

        # N-BEATS
        torch.manual_seed(42)
        nb = NBEATS(input_dim=input_dim, seq_len=SEQ_LEN, n_stacks=2, n_blocks=3, hidden=128)
        cfg_nb = TrainConfig(epochs=200, lr=1e-3, batch_size=bs, patience=25)
        res_nb = train_nbeats(nb, X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, cfg_nb)
        nb.eval()
        with torch.no_grad():
            pred_nb = nb(torch.FloatTensor(X_seq_te)).numpy()
        results.append(BaselineResult("nbeats", y_seq_te, pred_nb))
        if target == "ethereum":
            torch.save(nb.state_dict(), str(DATA_DIR / "nbeats_best.pt"))
    else:
        print(f"  neural skipped (only {len(X_seq_tr)} train sequences)")

    # ensemble
    if len(results) >= 2:
        min_len = min(len(r.y_true) for r in results)
        aligned = [BaselineResult(r.name, r.y_true[-min_len:], r.y_pred[-min_len:]) for r in results]
        ens = WeightedEnsemble()
        ens.fit(aligned)
        r_ens = ens.predict(aligned)
        aligned.append(r_ens)
        print(comparison_table(aligned))
        results_all[target] = {
            r.name: {
                "rmse": round(rmse(r.y_true, r.y_pred), 6),
                "mape": round(mape(r.y_true, r.y_pred), 2),
                "r2": round(r_squared(r.y_true, r.y_pred), 4),
                "dir_acc": round(directional_accuracy(r.y_true, r.y_pred), 1),
            }
            for r in aligned
        }

def main():
    t0 = time.time()
    print("Loading 365d price data...")
    prices = load_parquet("prices_365d")
    piv = pivot_prices(prices)
    avail = [c for c in piv.columns if len(piv[c].dropna()) > 60]
    piv = piv[avail]
    print(f"  {len(avail)} coins, {len(piv)} days")

    results_all = {}
    targets = [a for a in ASSETS_TO_EVAL if a in avail]
    if not targets:
        targets = avail[:5]

    for t in targets:
        train_for_asset(piv, t, results_all)

    # save results
    with open(DATA_DIR / "eval_results.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {DATA_DIR / 'eval_results.json'}")
    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
