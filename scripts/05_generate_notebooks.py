"""Generate all 3 Jupyter notebooks programmatically.
Run: python scripts/05_generate_notebooks.py
"""
import json
from pathlib import Path

def cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python", "version": "3.12.0"}},
        "cells": cells,
    }

# Notebook 01: Data Pipeline + EDA
nb01 = nb([
    cell("# 01 - Data Pipeline & EDA\n\nFetch real DeFi data from DeFiLlama and CoinGecko.\nExplore price series, yield distributions, and data quality.", "markdown"),
    cell("import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-v0_8-whitegrid')\n%matplotlib inline"),
    cell("from src.data import (\n    fetch_defillama_yields, fetch_coingecko_prices_cached,\n    save_parquet, load_parquet, pivot_prices, BASE_ASSETS, DATA_DIR\n)"),
    cell("# Fetch yields\nyields = fetch_defillama_yields()\nsave_parquet(yields, 'yields_base')\nprint(f'Shape: {yields.shape}')\nprint(f'Projects: {yields[\"project\"].nunique()}')\nyields.head(10)"),
    cell("# Fetch 365d prices (cached)\nprices = fetch_coingecko_prices_cached(BASE_ASSETS, days=365, cache_name='prices_365d')\nprint(f'Shape: {prices.shape}')\nprint(f'Coins: {prices[\"coin_id\"].nunique()}')\nprint(f'Date range: {prices[\"date\"].min().date()} to {prices[\"date\"].max().date()}')\nprices.head()"),
    cell("# Missing values\nprint('Missing values per column:')\nprint(prices.isna().sum())\nprint(f'\\nTotal rows: {len(prices)}')\nprint(f'Rows with any NaN: {prices.isna().any(axis=1).sum()}')"),
    cell("# Price series: ETH, USDC, AAVE\npiv = pivot_prices(prices)\nfig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)\nfor ax, coin in zip(axes, ['ethereum', 'usd-coin', 'aave']):\n    if coin in piv.columns:\n        ax.plot(piv.index, piv[coin])\n        ax.set_ylabel(f'{coin} (USD)')\n        ax.set_title(coin)\nfig.suptitle('365-Day Price Series (Base Chain Assets)', fontsize=14)\nplt.tight_layout()\nplt.savefig('../data/price_series.png', dpi=150)\nplt.show()"),
    cell("# Yield distribution for Base pools\nfig, ax = plt.subplots(figsize=(10, 5))\nyields_filtered = yields[yields['apy'].between(0, 100)]\nax.hist(yields_filtered['apy'], bins=50, edgecolor='black', alpha=0.7)\nax.set_xlabel('APY (%)')\nax.set_ylabel('Count')\nax.set_title(f'Yield Distribution - Base Chain ({len(yields_filtered)} pools, APY 0-100%)')\nplt.tight_layout()\nplt.savefig('../data/yield_dist.png', dpi=150)\nplt.show()"),
    cell("# Correlation heatmap\nreturns = np.log(piv / piv.shift(1)).dropna()\ntop_coins = returns.std().nlargest(10).index\ncorr = returns[top_coins].corr()\nfig, ax = plt.subplots(figsize=(10, 8))\nim = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)\nax.set_xticks(range(len(top_coins)))\nax.set_yticks(range(len(top_coins)))\nax.set_xticklabels(top_coins, rotation=45, ha='right')\nax.set_yticklabels(top_coins)\nplt.colorbar(im)\nax.set_title('Return Correlation Matrix (Top 10 by Volatility)')\nplt.tight_layout()\nplt.savefig('../data/correlation.png', dpi=150)\nplt.show()"),
    cell("# Summary stats\nprint(f'Assets: {len(piv.columns)}')\nprint(f'Date range: {piv.index.min().date()} to {piv.index.max().date()}')\nprint(f'Days: {len(piv)}')\nprint(f'\\nAnnualized volatility (top 5):')\nvol = returns.std() * np.sqrt(365)\nfor c in vol.nlargest(5).index:\n    print(f'  {c}: {vol[c]:.1%}')"),
])

# Notebook 02: Baseline Models
nb02 = nb([
    cell("# 02 - Baseline Models\n\nEstablish performance floor with naive, mean reversion, and ridge baselines.", "markdown"),
    cell("import sys\nsys.path.insert(0, '..')\n\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-v0_8-whitegrid')\n%matplotlib inline"),
    cell("from src.data import load_parquet, pivot_prices\nfrom src.features import build_features, train_test_split\nfrom src.baseline import NaiveBaseline, MeanReversion, LinearBaseline\nfrom src.evaluate import rmse, mape, mae, r_squared, directional_accuracy, comparison_table"),
    cell("# Load data\nprices = load_parquet('prices_365d')\npiv = pivot_prices(prices)\navail = [c for c in piv.columns if len(piv[c].dropna()) > 60]\npiv = piv[avail]\nprint(f'{len(avail)} coins, {len(piv)} days')"),
    cell("# Build features\nfeat, tgt = build_features(piv)\nXtr, Xte, ytr, yte = train_test_split(feat, tgt, 0.2)\nprint(f'Features: {feat.shape[1]} cols')\nprint(f'Train: {len(Xtr)}, Test: {len(Xte)}')"),
    cell("# Run baselines for ETH\ntarget = 'ethereum'\nresults = []\nfor cls in [NaiveBaseline, MeanReversion, LinearBaseline]:\n    m = cls()\n    r = m.predict(Xtr, Xte, ytr, yte, target)\n    results.append(r)\n\nprint(comparison_table(results))"),
    cell("# Plot predicted vs actual\nfig, axes = plt.subplots(1, 3, figsize=(15, 4))\nfor ax, r in zip(axes, results):\n    ax.scatter(r.y_true, r.y_pred, alpha=0.6, s=20)\n    lims = [min(r.y_true.min(), r.y_pred.min()), max(r.y_true.max(), r.y_pred.max())]\n    ax.plot(lims, lims, 'r--', alpha=0.5)\n    ax.set_xlabel('Actual')\n    ax.set_ylabel('Predicted')\n    ax.set_title(f'{r.name} (RMSE: {rmse(r.y_true, r.y_pred):.4f})')\nplt.suptitle(f'{target} - 7d Return Prediction', fontsize=14)\nplt.tight_layout()\nplt.savefig('../data/baselines.png', dpi=150)\nplt.show()"),
    cell("# Multi-asset baseline comparison\nprint(f'{\"Asset\":<25} {\"Naive\":>10} {\"MeanRev\":>10} {\"Ridge\":>10}')\nprint('-' * 55)\nfor asset in avail[:10]:\n    scores = []\n    for cls in [NaiveBaseline, MeanReversion, LinearBaseline]:\n        m = cls()\n        r = m.predict(Xtr, Xte, ytr, yte, asset)\n        scores.append(rmse(r.y_true, r.y_pred))\n    print(f'{asset:<25} {scores[0]:>10.4f} {scores[1]:>10.4f} {scores[2]:>10.4f}')"),
])

# Notebook 03: LSTM + N-BEATS + Full Comparison
nb03 = nb([
    cell("# 03 - Neural Models & Full Evaluation\n\nTrain LSTM and N-BEATS from scratch. Compare all 7 models + ensemble.", "markdown"),
    cell("import sys\nsys.path.insert(0, '..')\n\nimport numpy as np\nimport torch\nimport pandas as pd\nimport matplotlib.pyplot as plt\nplt.style.use('seaborn-v0_8-whitegrid')\n%matplotlib inline"),
    cell("from src.data import load_parquet, pivot_prices, DATA_DIR\nfrom src.features import build_features, train_test_split, make_sequences\nfrom src.baseline import NaiveBaseline, MeanReversion, LinearBaseline, BaselineResult\nfrom src.model import StrategyLSTM, train as train_lstm, mc_dropout_predict, TrainConfig\nfrom src.nbeats import NBEATS, train_nbeats, mc_dropout_nbeats\nfrom src.tree import XGBoostModel, LightGBMModel\nfrom src.ensemble import WeightedEnsemble\nfrom src.evaluate import rmse, mape, comparison_table"),
    cell("# Load and prepare\nprices = load_parquet('prices_365d')\npiv = pivot_prices(prices)\navail = [c for c in piv.columns if len(piv[c].dropna()) > 60]\npiv = piv[avail]\nfeat, tgt = build_features(piv)\nXtr, Xte, ytr, yte = train_test_split(feat, tgt, 0.2)\ntarget = 'ethereum'\nprint(f'Train: {len(Xtr)}, Test: {len(Xte)}, Features: {feat.shape[1]}')"),
    cell("# Prepare sequences for neural models\nSEQ_LEN = 14\nX_arr = feat.values.astype(np.float32)\nX_arr = np.nan_to_num(X_arr, nan=0.0)\ny_arr = tgt[target].values.astype(np.float32)\nsplit = int(len(X_arr) * 0.8)\nX_seq_tr, y_seq_tr = make_sequences(X_arr[:split], y_arr[:split], SEQ_LEN)\nX_seq_te, y_seq_te = make_sequences(X_arr[split:], y_arr[split:], SEQ_LEN)\nprint(f'Sequences - train: {X_seq_tr.shape}, test: {X_seq_te.shape}')"),
    cell("# Train LSTM\ntorch.manual_seed(42)\ninput_dim = X_seq_tr.shape[2]\nlstm = StrategyLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)\ncfg = TrainConfig(epochs=200, lr=1e-3, batch_size=min(32, len(X_seq_tr)), patience=25)\nres_lstm = train_lstm(lstm, X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, cfg)\nprint(f'Epochs: {res_lstm[\"epochs_trained\"]}, Best val loss: {res_lstm[\"best_val_loss\"]:.6f}')"),
    cell("# Train N-BEATS\ntorch.manual_seed(42)\nnbeats = NBEATS(input_dim=input_dim, seq_len=SEQ_LEN, n_stacks=2, n_blocks=3, hidden=128)\ncfg_nb = TrainConfig(epochs=200, lr=1e-3, batch_size=min(32, len(X_seq_tr)), patience=25)\nres_nb = train_nbeats(nbeats, X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, cfg_nb)\nprint(f'Epochs: {res_nb[\"epochs_trained\"]}, Best val loss: {res_nb[\"best_val_loss\"]:.6f}')"),
    cell("# Training curves\nfig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))\nax1.plot(res_lstm['history']['train_loss'], label='train')\nax1.plot(res_lstm['history']['val_loss'], label='val')\nax1.set_title('LSTM Training')\nax1.set_xlabel('Epoch')\nax1.set_ylabel('MSE Loss')\nax1.legend()\n\nax2.plot(res_nb['history']['train_loss'], label='train')\nax2.plot(res_nb['history']['val_loss'], label='val')\nax2.set_title('N-BEATS Training')\nax2.set_xlabel('Epoch')\nax2.set_ylabel('MSE Loss')\nax2.legend()\nplt.tight_layout()\nplt.savefig('../data/training_curves.png', dpi=150)\nplt.show()"),
    cell("# Collect all predictions\nresults = []\n\n# Baselines\nfor cls in [NaiveBaseline, MeanReversion, LinearBaseline]:\n    r = cls().predict(Xtr, Xte, ytr, yte, target)\n    results.append(r)\n\n# Trees\nfor cls in [XGBoostModel, LightGBMModel]:\n    r = cls().predict(Xtr, Xte, ytr, yte, target)\n    results.append(r)\n\n# Neural\nlstm.eval()\nwith torch.no_grad():\n    lstm_pred = lstm(torch.FloatTensor(X_seq_te)).numpy()\nresults.append(BaselineResult('lstm', y_seq_te, lstm_pred))\n\nnbeats.eval()\nwith torch.no_grad():\n    nb_pred = nbeats(torch.FloatTensor(X_seq_te)).numpy()\nresults.append(BaselineResult('nbeats', y_seq_te, nb_pred))\n\n# Ensemble\nmin_len = min(len(r.y_true) for r in results)\naligned = [BaselineResult(r.name, r.y_true[-min_len:], r.y_pred[-min_len:]) for r in results]\nens = WeightedEnsemble()\nens.fit(aligned)\nr_ens = ens.predict(aligned)\naligned.append(r_ens)\n\nprint(comparison_table(aligned))\nprint()\nprint(ens.summary())"),
    cell("# MC Dropout confidence intervals\nmean, low, high = mc_dropout_predict(lstm, X_seq_te, n_samples=50)\nfig, ax = plt.subplots(figsize=(12, 5))\nx = range(len(y_seq_te))\nax.plot(x, y_seq_te, 'k-', label='Actual', linewidth=1.5)\nax.plot(x, mean, 'b-', label='LSTM Prediction', alpha=0.8)\nax.fill_between(x, low, high, alpha=0.2, color='blue', label='95% CI')\nax.set_xlabel('Test Sample')\nax.set_ylabel('7-day Return')\nax.set_title(f'{target} - LSTM with MC Dropout Confidence Intervals')\nax.legend()\nplt.tight_layout()\nplt.savefig('../data/confidence_intervals.png', dpi=150)\nplt.show()"),
    cell("# Final RMSE bar chart\nnames = [r.name for r in aligned]\nrmses = [rmse(r.y_true, r.y_pred) for r in aligned]\ncolors = ['#d62728' if n in ['naive'] else '#2ca02c' if n == 'ensemble' else '#1f77b4' for n in names]\n\nfig, ax = plt.subplots(figsize=(10, 5))\nbars = ax.bar(names, rmses, color=colors, edgecolor='black', alpha=0.8)\nax.set_ylabel('RMSE')\nax.set_title(f'{target} - Model Comparison (lower is better)')\nplt.xticks(rotation=45, ha='right')\nfor bar, val in zip(bars, rmses):\n    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,\n            f'{val:.4f}', ha='center', va='bottom', fontsize=9)\nplt.tight_layout()\nplt.savefig('../data/model_comparison.png', dpi=150)\nplt.show()"),
])

# Write notebooks
for name, notebook in [("01_data", nb01), ("02_baseline", nb02), ("03_lstm", nb03)]:
    path = Path(f"notebooks/{name}.ipynb")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Written {path}")
