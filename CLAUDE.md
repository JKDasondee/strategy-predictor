# Strategy Performance Predictor

## What This Is
ML-powered strategy performance analysis for Glider.Fi (100K users, 30K+ funded portfolios).
Takes a portfolio allocation (assets + weights + chain), returns historical performance analysis
with confidence intervals and attribution. Serves as analytics layer for B2B API and Strategy Page.

## Architecture
- Python ML pipeline (PyTorch LSTM, scikit-learn baselines)
- Go API service (single binary, gRPC-optional)
- Data: DeFiLlama yields API + CoinGecko price series (real on-chain data, no mocks)
- Output: JSON predictions via REST endpoint

## Project Structure
```
strategy-predictor/
├── CLAUDE.md
├── data/               # Parquet files, never committed (gitignored)
├── notebooks/
│   ├── 01_data.ipynb        # Data pipeline + EDA
│   ├── 02_baseline.ipynb    # Naive predictor + metrics
│   └── 03_lstm.ipynb        # PyTorch model + eval
├── src/
│   ├── data.py              # DeFiLlama + CoinGecko fetchers
│   ├── features.py          # Feature engineering
│   ├── model.py             # PyTorch LSTM (nn.Module, from scratch)
│   ├── baseline.py          # Naive + linear regression baselines
│   ├── evaluate.py          # RMSE, MAPE, comparison tables
│   └── predict.py           # Inference pipeline
├── cmd/
│   └── server/
│       └── main.go          # Go HTTP service
├── configs/
│   └── assets.json          # Top 20 Base chain assets with 0xAddress:chainId mapping
├── tests/
├── docs/
│   └── eval.md              # Model evaluation report
├── requirements.txt
├── go.mod
└── README.md
```

## Rules
- NO HuggingFace, NO pre-trained models. All PyTorch is raw nn.Module with custom training loops.
- NO Claude API / LLM wrapper. This is real ML, not prompt engineering.
- NO dashboards or Streamlit UI. Output is JSON via API endpoint. Period.
- Every model MUST be compared against a naive baseline. If it doesn't beat naive, document why.
- All data is REAL (DeFiLlama + CoinGecko). No synthetic/mock data in final version.
- Error metrics (RMSE, MAPE) logged for every training run.
- Code style: minimal comments, short variable names, type hints everywhere.
- Go code uses standard library + chi router. No frameworks.

## Asset Format
Mirrors Glider's format: `0xTokenAddress:chainId` (Base = 8453)
Example: `0x4200000000000000000000000000000000000006:8453` = WETH on Base

## API Contract
```
GET /analyze?assets=WETH:0.4,USDC:0.3,AAVE:0.3&chain=8453

Response:
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
  },
  "data_range": {"from": "2026-01-01", "to": "2026-03-30"},
  "model_version": "lstm-v0.1"
}
```

## Build Sequence (14 days)
Day 1-2: Data pipeline (src/data.py + notebook 01)
Day 3-4: Baseline model (src/baseline.py + notebook 02)
Day 5-7: LSTM model (src/model.py + notebook 03)
Day 8-9: Go service (cmd/server/main.go)
Day 10-11: Glider format mapping + asset resolution
Day 12-13: README + eval report + GIF demo
Day 14: Demo-ready

## Context
- Glider.Fi: non-custodial onchain portfolio automation (a16z CSX, Coinbase Ventures)
- Brian Huang (CEO): MIT EECS, Morgan Stanley → XTX Markets → Anchorage → Glider
- Current P0s: B2B API V1, Strategy Page redesign, campaign targeting
- This project feeds directly into Strategy Page rankings + B2B analytics endpoint
- Frame as "performance analysis" not "predicted returns" (compliance-safe)
- Jay's edge: community ops sees user pain daily via Crisp tickets
