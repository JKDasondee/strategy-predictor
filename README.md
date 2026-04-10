# strategy-predictor

**DeFi strategy performance prediction via LSTM ensemble on real protocol data**

Trains on live DeFi yield and price data (DeFiLlama + CoinGecko) to predict strategy returns. Stacked ensemble: LSTM + N-BEATS + XGBoost + LightGBM. Go REST API serves predictions.

---

### Architecture

```
Data         DeFiLlama yields + CoinGecko prices → Parquet cache
Features     Log returns, 14-step sequences, train/val split
Models       LSTM · N-BEATS · XGBoost · LightGBM → stacked ensemble
API          Go REST (cmd/) serving model predictions
```

### Stack

```
ML        PyTorch · XGBoost · LightGBM
Data      pandas · numpy · DeFiLlama API · CoinGecko API
API       Go
```

### Usage

```bash
pip install -r requirements.txt
python train.py          # fetches data, trains all models, saves ensemble
cd cmd && go run main.go # starts REST API
```
