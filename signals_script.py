import os
import sys
import requests
import pandas as pd
import numpy as np
import polars as pl
import yfinance as yf
import numerapi
import gc

# ==========================================
# CONFIGURATION
# ==========================================
FRED_API_KEY = "3699988d98d460d752a241f85df9532f"
MODEL_SLOT_NAME = "dawityimer"

# ==========================================
# TICKER CONVERTER
# ==========================================
class TickerConverter:
    def __init__(self):
        # Map exchange codes to Yahoo suffixes
        self.iso_map = {"US": "", "JP": ".T", "GB": ".L", "CA": ".TO", "AU": ".AX", "FR": ".PA"}
        self.inv_map = {v.replace(".",""): k for k, v in self.iso_map.items()}

    def to_yahoo(self, t):
        parts = t.split(" ")
        if len(parts) == 2:
            ticker, exch = parts[0].replace("/", "-"), parts[1]
            return f"{ticker}{self.iso_map.get(exch, '.'+exch.lower())}"
        return t

    def to_numerai(self, yt, target_col):
        if "." in yt:
            t, s = yt.split(".", 1)
            return f"{t.replace('-', '/')} {self.inv_map.get(s.upper(), s.upper())}"
        return f"{yt.replace('-', '/')} US"

# ==========================================
# DATA & ALPHA ENGINE
# ==========================================
def get_yield_shield():
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&api_key={FRED_API_KEY}&file_type=json"
    try:
        data = requests.get(url, timeout=10).json()
        val = data["observations"][-1]["value"]
        return float(val) if val != "." else 1.0
    except: return 1.0

def main():
    sapi = numerapi.SignalsAPI(os.getenv('NUMERAI_PUBLIC_KEY'), os.getenv('NUMERAI_SECRET_KEY'))
    conv = TickerConverter()

    # 1. ADAPTIVE UNIVERSE HANDSHAKE
    potential_paths = ["live.parquet", "signals/v2.0/live.parquet", "signals/v1.0/live.parquet"]
    universe = None
    
    print("Initiating Adaptive Universe Handshake...")
    for path in potential_paths:
        try:
            print(f"Checking path: {path}")
            sapi.download_dataset(path, "live.parquet")
            universe = pl.read_parquet("live.parquet")
            print(f"Successfully loaded universe from {path}")
            break
        except Exception as e:
            continue

    if universe is None:
        print("CRITICAL: Failed to download universe.")
        sys.exit(1)

    # DYNAMIC COLUMN DISCOVERY: Find which name Numerai is using today
    possible_names = ["numerai_ticker", "bloomberg_ticker", "ticker"]
    ticker_col = next((c for c in possible_names if c in universe.columns), None)
    
    if not ticker_col:
        print(f"CRITICAL: Could not find ticker column. Available: {universe.columns}")
        sys.exit(1)
        
    print(f"Using identified ticker column: '{ticker_col}'")
    all_tickers = universe[ticker_col].unique().to_list()
    
    # 2. TARGETED SAMPLING (Top 250 US)
    us_universe = [t for t in all_tickers if t is not None and t.endswith(" US")][:250]
    yahoo_list = [conv.to_yahoo(t) for t in us_universe]

    # 3. DOWNLOAD MARKET DATA
    print(f"Fetching data for {len(yahoo_list)} tickers...")
    raw_data = yf.download(yahoo_list, period="7mo", interval="1d", threads=False, progress=False)
    
    # Robust conversion from yfinance to Polars
    prices_pd = raw_data['Adj Close'].stack().reset_index()
    prices_pd.columns = ['date', 'yahoo_ticker', 'close']
    df = pl.from_pandas(prices_pd).sort(["yahoo_ticker", "date"]).drop_nulls()

    # 4. ALPHA CALCULATION (Risk-Adjusted Mean Reversion)
    df = df.with_columns([
        pl.col("close").rolling_mean(20).over("yahoo_ticker").alias("ma20"),
        pl.col("close").rolling_std(20).over("yahoo_ticker").alias("std20"),
        (pl.col("close").pct_change().rolling_std(20).over("yahoo_ticker") * np.sqrt(252)).alias("ann_vol")
    ]).filter(pl.col("std20") > 0)

    # Apply LdP Logic: Inverse Z-Score / Annual Volatility
    df = df.with_columns(
        (((pl.col("close") - pl.col("ma20")) / pl.col("std20")) * -1.0 / pl.col("ann_vol")).alias("raw_signal")
    )

    latest = df.group_by("yahoo_ticker").last().drop_nulls()
    latest = latest.with_columns(
        (pl.col("raw_signal").rank() / (latest.height + 1)).alias("signal")
    )

    # 5. MACRO SHIELD
    if get_yield_shield() < 0:
        print("Yield Inversion! Setting neutral signals.")
        latest = latest.with_columns(pl.lit(0.5).alias("signal"))

    # 6. MAPPING & SUBMISSION
    latest = latest.with_columns(
        pl.col("yahoo_ticker").map_elements(lambda x: conv.to_numerai(x, ticker_col), return_dtype=pl.String).alias(ticker_col)
    )

    final_sub = universe.select(ticker_col).join(latest.select([ticker_col, "signal"]), on=ticker_col, how="left")
    final_sub = final_sub.with_columns(pl.col("signal").fill_null(0.5).clip(0.01, 0.99))
    
    final_sub.to_pandas().to_csv("submission.csv", index=False)
    
    # Model ID Lookup
    models = sapi.get_models()
    model_id = models.get(MODEL_SLOT_NAME) or next(iter(models.values()))
    
    sapi.upload_predictions("submission.csv", model_id=model_id)
    print("SIGNALS ENGINE: MISSION SUCCESSFUL.")

if __name__ == "__main__":
    main()
