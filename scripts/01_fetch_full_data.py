"""Fetch 365 days of price data for all 20 assets + DeFiLlama yields.
Run: python scripts/01_fetch_full_data.py
Expected time: ~3-5 min (CoinGecko rate limits)
"""
import sys
sys.path.insert(0, ".")

from src.data import fetch_coingecko_prices, fetch_defillama_yields, save_parquet, BASE_ASSETS, DATA_DIR

print("Fetching DeFiLlama yields...")
yields = fetch_defillama_yields()
save_parquet(yields, "yields_base")
print(f"  {yields.shape[0]} pools, {yields['project'].nunique()} projects")

print(f"\nFetching 365d prices for {len(BASE_ASSETS)} assets...")
print("  (rate limited, ~1.5s per coin)")
prices = fetch_coingecko_prices(BASE_ASSETS, days=365)
save_parquet(prices, "prices_365d")
print(f"  {prices.shape[0]} rows, {prices['coin_id'].nunique()} coins")
print(f"  {prices['date'].min().date()} to {prices['date'].max().date()}")
print(f"\nSaved to {DATA_DIR}")
