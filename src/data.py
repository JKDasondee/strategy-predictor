import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_ASSETS: list[str] = [
    "ethereum", "usd-coin", "aave", "chainlink", "uniswap",
    "maker", "compound-governance-token", "lido-dao", "rocket-pool",
    "wrapped-bitcoin", "dai", "tether", "curve-dao-token", "balancer",
    "sushi", "synthetix-network-token", "yearn-finance", "1inch",
    "the-graph", "ens",
]

def fetch_defillama_yields(chain: str = "Base") -> pd.DataFrame:
    r = requests.get("https://yields.llama.fi/pools", timeout=30)
    r.raise_for_status()
    pools = r.json()["data"]
    df = pd.DataFrame(pools)
    df = df[df["chain"] == chain].copy()
    cols = ["pool", "project", "symbol", "tvlUsd", "apy", "apyBase", "apyReward"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].reset_index(drop=True)
    df["timestamp"] = pd.Timestamp.now()
    return df

def fetch_coingecko_prices(
    coin_ids: list[str],
    days: int = 90,
    vs: str = "usd",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base = "https://api.coingecko.com/api/v3/coins"
    for cid in coin_ids:
        url = f"{base}/{cid}/market_chart"
        params = {"vs_currency": vs, "days": days, "interval": "daily"}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(60)
            r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            print(f"skip {cid}: {r.status_code}")
            continue
        j = r.json()
        prices = j.get("prices", [])
        volumes = j.get("total_volumes", [])
        vol_map = {int(v[0]): v[1] for v in volumes}
        rows = []
        for ts, p in prices:
            rows.append({
                "date": pd.Timestamp(ts, unit="ms").normalize(),
                "coin_id": cid,
                "price_usd": p,
                "volume_usd": vol_map.get(int(ts), np.nan),
            })
        frames.append(pd.DataFrame(rows))
        time.sleep(1.5)
    if not frames:
        return pd.DataFrame(columns=["date", "coin_id", "price_usd", "volume_usd"])
    return pd.concat(frames, ignore_index=True)

def fetch_coingecko_prices_cached(
    coin_ids: list[str],
    days: int = 90,
    cache_name: str = "prices",
) -> pd.DataFrame:
    p = DATA_DIR / f"{cache_name}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        age = (pd.Timestamp.now() - pd.Timestamp(df["date"].max())).days
        if age < 2:
            return df[df["coin_id"].isin(coin_ids)]
    df = fetch_coingecko_prices(coin_ids, days)
    if len(df) > 0:
        save_parquet(df, cache_name)
    return df

def save_parquet(df: pd.DataFrame, name: str) -> Path:
    p = DATA_DIR / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p

def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")

def pivot_prices(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot_table(index="date", columns="coin_id", values="price_usd")
    piv = piv.sort_index().ffill()
    return piv

if __name__ == "__main__":
    print("fetching defillama yields...")
    yields = fetch_defillama_yields()
    save_parquet(yields, "yields_base")
    print(f"yields: {yields.shape}")

    print("fetching coingecko prices...")
    prices = fetch_coingecko_prices(BASE_ASSETS[:5], days=90)
    save_parquet(prices, "prices")
    print(f"prices: {prices.shape}")
    print(f"coins: {prices['coin_id'].unique().tolist()}")
    print(f"date range: {prices['date'].min()} → {prices['date'].max()}")
