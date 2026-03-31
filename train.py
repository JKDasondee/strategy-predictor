import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from src.data import fetch_coingecko_prices_cached, pivot_prices, BASE_ASSETS, DATA_DIR, save_parquet, fetch_defillama_yields
from src.features import build_features, train_test_split, make_sequences, log_returns
from src.baseline import NaiveBaseline, MeanReversion, LinearBaseline, BaselineResult
from src.model import StrategyLSTM, train as train_lstm, mc_dropout_predict, TrainConfig
from src.nbeats import NBEATS, train_nbeats, mc_dropout_nbeats
from src.tree import XGBoostModel, LightGBMModel
from src.ensemble import WeightedEnsemble, StackedEnsemble
from src.evaluate import rmse, mape, mae, r_squared, directional_accuracy, comparison_table

SEQ_LEN = 14
TARGET_ASSET = "ethereum"

def main():
    t0 = time.time()
    print("=" * 60)
    print("STRATEGY PREDICTOR — FULL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Data
    print("\n[1/6] Fetching data...")
    yields = fetch_defillama_yields()
    save_parquet(yields, "yields_base")
    print(f"  yields: {yields.shape[0]} Base pools, {yields['project'].nunique()} projects")

    prices = fetch_coingecko_prices_cached(BASE_ASSETS, days=90, cache_name="prices_full")
    print(f"  prices: {prices.shape[0]} rows, {prices['coin_id'].nunique()} coins")
    print(f"  range: {prices['date'].min().date()} to {prices['date'].max().date()}")

    piv = pivot_prices(prices)
    avail = [c for c in piv.columns if len(piv[c].dropna()) > 30]
    piv = piv[avail]
    print(f"  usable coins: {len(avail)}")

    # 2. Features
    print("\n[2/6] Building features...")
    feat, tgt = build_features(piv)
    Xtr, Xte, ytr, yte = train_test_split(feat, tgt, 0.2)
    print(f"  features: {feat.shape[1]} cols, {len(feat)} rows")
    print(f"  train: {len(Xtr)}, test: {len(Xte)}")

    target = TARGET_ASSET if TARGET_ASSET in tgt.columns else avail[0]
    print(f"  target asset: {target}")

    # 3. Baselines
    print("\n[3/6] Running baselines...")
    results = []

    naive = NaiveBaseline()
    r_naive = naive.predict(Xtr, Xte, ytr, yte, target)
    results.append(r_naive)
    print(f"  naive   RMSE: {rmse(r_naive.y_true, r_naive.y_pred):.6f}")

    mr = MeanReversion()
    r_mr = mr.predict(Xtr, Xte, ytr, yte, target)
    results.append(r_mr)
    print(f"  meanrev RMSE: {rmse(r_mr.y_true, r_mr.y_pred):.6f}")

    ridge = LinearBaseline()
    r_ridge = ridge.predict(Xtr, Xte, ytr, yte, target)
    results.append(r_ridge)
    print(f"  ridge   RMSE: {rmse(r_ridge.y_true, r_ridge.y_pred):.6f}")

    # 4. Tree models
    print("\n[4/6] Training tree models...")
    try:
        xgb_m = XGBoostModel()
        r_xgb = xgb_m.predict(Xtr, Xte, ytr, yte, target)
        results.append(r_xgb)
        print(f"  xgboost RMSE: {rmse(r_xgb.y_true, r_xgb.y_pred):.6f}")
    except Exception as e:
        print(f"  xgboost skipped: {e}")

    try:
        lgb_m = LightGBMModel()
        r_lgb = lgb_m.predict(Xtr, Xte, ytr, yte, target)
        results.append(r_lgb)
        print(f"  lgbm    RMSE: {rmse(r_lgb.y_true, r_lgb.y_pred):.6f}")
    except Exception as e:
        print(f"  lgbm skipped: {e}")

    # 5. Neural models
    print("\n[5/6] Training neural models...")
    X_arr = feat.values.astype(np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0)
    y_arr = tgt[target].values.astype(np.float32)

    split = int(len(X_arr) * 0.8)
    X_seq_tr, y_seq_tr = make_sequences(X_arr[:split], y_arr[:split], SEQ_LEN)
    X_seq_te, y_seq_te = make_sequences(X_arr[split:], y_arr[split:], SEQ_LEN)

    if len(X_seq_tr) > 5 and len(X_seq_te) > 0:
        input_dim = X_seq_tr.shape[2]

        # LSTM
        print("  training LSTM...")
        torch.manual_seed(42)
        lstm = StrategyLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
        cfg = TrainConfig(epochs=150, lr=1e-3, batch_size=min(32, len(X_seq_tr)), patience=20)
        res_lstm = train_lstm(lstm, X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, cfg)
        print(f"  LSTM epochs: {res_lstm['epochs_trained']}, val_loss: {res_lstm['best_val_loss']:.6f}")

        lstm.eval()
        with torch.no_grad():
            lstm_pred = lstm(torch.FloatTensor(X_seq_te)).numpy()
        r_lstm = BaselineResult("lstm", y_seq_te, lstm_pred)
        results.append(r_lstm)
        print(f"  lstm    RMSE: {rmse(r_lstm.y_true, r_lstm.y_pred):.6f}")
        torch.save(lstm.state_dict(), str(DATA_DIR / "lstm_best.pt"))

        # N-BEATS
        print("  training N-BEATS...")
        torch.manual_seed(42)
        nbeats = NBEATS(input_dim=input_dim, seq_len=SEQ_LEN, n_stacks=2, n_blocks=3, hidden=128)
        cfg_nb = TrainConfig(epochs=150, lr=1e-3, batch_size=min(32, len(X_seq_tr)), patience=20)
        res_nb = train_nbeats(nbeats, X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, cfg_nb)
        print(f"  N-BEATS epochs: {res_nb['epochs_trained']}, val_loss: {res_nb['best_val_loss']:.6f}")

        nbeats.eval()
        with torch.no_grad():
            nb_pred = nbeats(torch.FloatTensor(X_seq_te)).numpy()
        r_nb = BaselineResult("nbeats", y_seq_te, nb_pred)
        results.append(r_nb)
        print(f"  nbeats  RMSE: {rmse(r_nb.y_true, r_nb.y_pred):.6f}")
        torch.save(nbeats.state_dict(), str(DATA_DIR / "nbeats_best.pt"))
    else:
        print("  insufficient sequence data for neural models")

    # 6. Ensemble
    print("\n[6/6] Building ensemble...")
    if len(results) >= 2:
        # align all results to same length (neural models have shorter output)
        min_len = min(len(r.y_true) for r in results)
        aligned = [
            BaselineResult(r.name, r.y_true[-min_len:], r.y_pred[-min_len:])
            for r in results
        ]

        ens = WeightedEnsemble()
        ens.fit(aligned)
        r_ens = ens.predict(aligned)
        aligned.append(r_ens)
        print(ens.summary())
        print(f"  ensemble RMSE: {rmse(r_ens.y_true, r_ens.y_pred):.6f}")

        se = StackedEnsemble()
        se.fit(aligned[:-1])  # fit on individual models, not the ensemble
        r_se = se.predict(aligned[:-1])
        aligned.append(r_se)
        print(se.summary())
        print(f"  stacked RMSE: {rmse(r_se.y_true, r_se.y_pred):.6f}")

        # Final table
        print("\n" + "=" * 60)
        print("FINAL MODEL COMPARISON")
        print("=" * 60)
        print(comparison_table(aligned))
    else:
        print("  not enough models for ensemble")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Models saved to {DATA_DIR}/")

if __name__ == "__main__":
    main()
