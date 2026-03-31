"""Generate README.md with architecture, setup, usage, and results.
Run: python scripts/04_generate_readme.py
"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
from src.data import DATA_DIR

readme = r"""# Strategy Performance Predictor

ML-powered portfolio performance analysis for [Glider.Fi](https://glider.fi) (100K users, a16z CSX + Coinbase Ventures).

Takes a portfolio allocation (assets + weights + chain), returns 7-day performance predictions with confidence intervals and per-asset attribution.

## Architecture

```
                    +-------------------+
                    |   Go HTTP Server  |
                    |  /analyze /health |
                    +--------+----------+
                             |
                    +--------v----------+
                    | Python Inference   |
                    | src/predict.py     |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+ +-----v------+
     |  XGBoost   |  |   N-BEATS   | |    LSTM    |
     |  LightGBM  |  | (from scrch)| | (from scrch)|
     +--------+---+  +------+------+ +-----+------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v----------+
                    | Weighted Ensemble  |
                    | (Nelder-Mead opt)  |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     +--------v----------+    +-------------v---------+
     | CoinGecko Prices  |    | DeFiLlama Yields     |
     | 365d, 18 assets   |    | Base chain, 2900 pools|
     +--------------------+    +-----------------------+
```

## Models

| Model | Type | Implementation | Paper |
|-------|------|---------------|-------|
| Naive | Baseline | hand-rolled | - |
| Mean Reversion | Baseline | hand-rolled | - |
| Ridge | Linear | scikit-learn | - |
| XGBoost | Gradient Boosted Trees | xgboost | Chen & Guestrin 2016 |
| LightGBM | Gradient Boosted Trees | lightgbm | Ke et al. 2017 |
| LSTM | Recurrent NN | **PyTorch from scratch** | Hochreiter 1997 |
| N-BEATS | Residual Stack | **PyTorch from scratch** | Oreshkin et al., ICLR 2020 |
| Ensemble | Meta-learner | Nelder-Mead optimization | - |

## Quick Start

```bash
# Clone
git clone https://github.com/JKDasondee/strategy-predictor.git
cd strategy-predictor

# Install
pip install -r requirements.txt
pip install xgboost lightgbm

# Fetch data (365 days, ~5 min)
python scripts/01_fetch_full_data.py

# Train all models (~5 min)
python scripts/02_train_all.py

# Generate eval report
python scripts/03_generate_eval_report.py

# Run tests
python -m pytest tests/ -v

# Start API server
cd cmd/server && go run main.go
```

## API Usage

```bash
curl "localhost:8080/analyze?assets=WETH:0.4,USDC:0.3,AAVE:0.3&chain=8453"
```

Response:
```json
{
  "strategy": {"WETH": 0.4, "USDC": 0.3, "AAVE": 0.3},
  "chain_id": 8453,
  "analysis": {
    "predicted_7d_return": 0.023,
    "confidence_low": -0.011,
    "confidence_high": 0.057,
    "baseline_7d_return": 0.018,
    "model_rmse": 0.015,
    "baseline_rmse": 0.024,
    "beats_baseline": true
  },
  "attribution": {
    "WETH": {"contribution": 0.015, "volatility": 0.042},
    "USDC": {"contribution": 0.001, "volatility": 0.0002},
    "AAVE": {"contribution": 0.007, "volatility": 0.031}
  }
}
```

## Features

7 engineered features per asset:
- Log returns (daily)
- 7-day rolling returns
- 7/14-day rolling volatility
- 30-day rolling Sharpe ratio
- 30-day max drawdown
- 14-day EWM volatility

## Asset Format

Mirrors Glider's format: `0xTokenAddress:chainId` (Base = 8453)

20 Base chain assets supported. See `configs/assets.json` for full mapping.

## Project Structure

```
strategy-predictor/
├── src/
│   ├── data.py         # DeFiLlama + CoinGecko fetchers
│   ├── features.py     # Feature engineering pipeline
│   ├── baseline.py     # Naive, mean reversion, ridge baselines
│   ├── model.py        # PyTorch LSTM (nn.Module, from scratch)
│   ├── nbeats.py       # N-BEATS (ICLR 2020, from scratch)
│   ├── tree.py         # XGBoost + LightGBM wrappers
│   ├── ensemble.py     # Weighted + stacked ensemble
│   ├── evaluate.py     # RMSE, MAPE, MAE, R2, directional accuracy
│   └── predict.py      # End-to-end inference
├── cmd/server/main.go  # Go HTTP service
├── scripts/            # Data fetch, training, report generation
├── tests/              # 27 tests
├── configs/assets.json # Base chain asset mapping
└── docs/eval.md        # Model evaluation report
```

## Evaluation

See [docs/eval.md](docs/eval.md) for full results.

## Tech Stack

- **ML**: PyTorch (from scratch), scikit-learn, XGBoost, LightGBM
- **Data**: DeFiLlama API, CoinGecko API, Parquet storage
- **API**: Go standard library, net/http
- **Testing**: pytest (27 tests)

## Context

Built for [Glider.Fi](https://glider.fi) — non-custodial onchain portfolio automation.
Feeds into B2B API V1 and Strategy Page analytics.

This is **performance analysis**, not investment advice.

---

Built by [Jay Dasondee](https://github.com/JKDasondee)
"""

Path("README.md").write_text(readme.strip(), encoding="utf-8")
print("Written README.md")
