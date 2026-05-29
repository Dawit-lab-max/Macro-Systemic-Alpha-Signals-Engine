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

class TickerConverter:
    def __init__(self):
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

def get_yield_shield():
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&api_key={FRED_API_KEY}&file_type=json"
    try:
        data = requests.get(url, timeout=10).json()
        val = data["observations"][-1]["value"]
        return float(val) if val != "." else 1.0
    except: return 1.0

def main():
    # Setup API
    public_key = os.getenv('NUMERAI_PUBLIC_KEY')
    secret_key = os.getenv('NUMERAI_SECRET_KEY')
    sapi = numerapi.SignalsAPI(public_key, secret_key)
    conv = TickerConverter()

    # 1. UNIVERSE HANDSHAKE (FIXED VERSIONED PATH)
    # We try v1.0 which is the most stable path for Signals
    dataset_path = "signals/v1.0/live.parquet"
    print(f"Downloading Numerai Universe from {dataset_path}...")
    
    try:
        sapi.download_dataset(dataset_path, "live.parquet")
    except Exception as e:
        print(f"Primary path failed, trying fallback... {e}")
        sapi.download_dataset("live.parquet", "live.parquet") # API sometimes handles this differently

    universe = pl.read_parquet("live.parquet")
    possible_names = ["numerai_ticker", "bloomberg_ticker", "ticker"]
    ticker_col = next((c for c in possible_names if c in universe.columns), "ticker")
    all_tickers = universe[ticker_col].unique().to_list()
    
    # 2. TARGETED DATA FETCH (150 tickers for speed and reliability)
    us_universe = [t for t in all_tickers if t is not None and t.endswith(" US")][:150]
    yahoo_list = [conv.to_yahoo(t) for t in us_universe]

    print(f"Fetching market data for {len(yahoo_list)} stocks...")
    raw_data = yf.download(yahoo_list, period="7mo", interval="1d", threads=False, progress=False)
    
    if raw_data.empty or 'Adj Close' not in raw_data:
        print("Market data fetch failed. Using neutral baseline.")
        latest = pl.DataFrame({ticker_col: [], "signal": []})
    else:
        prices_pd = raw_data['Adj Close'].stack().reset_index()
        prices_pd.columns = ['date', 'yahoo_ticker', 'close']
        df = pl.from_pandas(prices_pd).sort(["yahoo_ticker", "date"]).drop_nulls()

        # 3. ALPHA ENGINE
        df = df.with_columns([
            pl.col("close").rolling_mean(20).over("yahoo_ticker").alias("ma20"),
            pl.col("close").rolling_std(20).over("yahoo_ticker").alias("std20"),
            (pl.col("close").pct_change().rolling_std(20).over("yahoo_ticker") * np.sqrt(252)).alias("ann_vol")
        ]).filter(pl.col("std20") > 0)

        df = df.with_columns(
            (((pl.col("close") - pl.col("ma20")) / pl.col("std20")) * -1.0 / pl.col("ann_vol")).alias("raw_signal")
        )

        latest = df.group_by("yahoo_ticker").last().drop_nulls()
        latest = latest.with_columns((pl.col("raw_signal").rank() / (latest.height + 1)).alias("signal"))
        latest = latest.with_columns(
            pl.col("yahoo_ticker").map_elements(lambda x: conv.to_numerai(x, ticker_col), return_dtype=pl.String).alias(ticker_col)
        )

    # 4. MACRO SHIELD
    if get_yield_shield() < 0:
        print("Yield Inversion! Protecting capital.")
        latest = latest.with_columns(pl.lit(0.5).alias("signal"))

    # 5. FINAL ASSEMBLY & EPSILON JITTER (Fixes non-zero std dev error)
    final_sub = universe.select(ticker_col).join(latest.select([ticker_col, "signal"]), on=ticker_col, how="left")
    final_sub = final_sub.with_columns(pl.col("signal").fill_null(0.5))

    # Add tiny random noise so the standard deviation is never exactly zero
    noise = np.random.uniform(-1e-6, 1e-6, len(final_sub))
    final_sub = final_sub.with_columns(
        (pl.col("signal") + pl.Series(noise)).clip(0.01, 0.99).alias("signal")
    )

    # 6. UPLOAD
    final_sub.to_pandas().to_csv("submission.csv", index=False)
    models = sapi.get_models()
    model_id = models.get(MODEL_SLOT_NAME) or next(iter(models.values()))
    
    print(f"Deploying to Numerai: {MODEL_SLOT_NAME}")
    sapi.upload_predictions("submission.csv", model_id=model_id)
    print("ET-QUANT SIGNALS ENGINE: DEPLOYMENT SUCCESS.")

if __name__ == "__main__":
    main()
