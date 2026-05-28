#!/usr/bin/env python3
"""
Numerai Signals Submission Pipeline
Author: MScFE Team / Dawit Yimer Production Setup
Description: Low-memory, crash-resistant production pipeline using Polars.
             Dynamically retrieves the official universe via live.parquet,
             decouples completely from yfinance, and fetches historical prices
             from Stooq using parallel I/O requests. Maps signals and applies
             FRED macro risk shields.
"""
import os
import sys
import requests
import io
import random
import concurrent.futures
import pandas as pd
import polars as pl
import numerapi

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
FRED_API_KEY = "3699988d98d460d752a241f85df9532f"
MODEL_SLOT_NAME = "dawityimer"

# ==========================================
# UNIVERSE HANDSHAKE
# ==========================================
def fetch_official_universe(sapi: numerapi.SignalsAPI) -> tuple[list[str], str]:
    """
    Downloads the official live.parquet file and extracts the active universe tickers.
    Tries successive active data versions (v2.1, v2.0, v1.0) for redundancy.
    Returns (list_of_tickers, ticker_column_name).
    """
    print("[Universe Handshake] Querying active ticker list via live.parquet...")
    versions = ["signals/v2.1/live.parquet", "signals/v2.0/live.parquet", "signals/v1.0/live.parquet"]
    
    for version in versions:
        try:
            print(f"[Universe Handshake] Attempting download of '{version}'...")
            sapi.download_dataset(version, "live.parquet")
            if os.path.exists("live.parquet"):
                print(f"[Universe Handshake] Successfully downloaded '{version}'")
                
                # Load with Polars to inspect the active column names
                df = pl.read_parquet("live.parquet")
                for col in ["numerai_ticker", "bloomberg_ticker", "ticker"]:
                    if col in df.columns:
                        tickers = df[col].drop_nulls().unique().to_list()
                        if tickers and len(tickers) > 100:
                            print(f"[Universe Handshake] Extracted {len(tickers)} tickers from column '{col}'")
                            return tickers, col
        except Exception as e:
            print(f"[Universe Handshake] Download/Read failed for '{version}': {e}")
            continue
            
    # Legacy fallbacks in case API dataset catalog is entirely unresponsive
    print("[Universe Handshake] Dataset download failed. Trying legacy fallback methods...")
    for method_name in ["get_eligible_tickers", "ticker_universe"]:
        if hasattr(sapi, method_name):
            try:
                method = getattr(sapi, method_name)
                tickers = method()
                if tickers and len(tickers) > 100:
                    print(f"[Universe Handshake] Fallback retrieved {len(tickers)} tickers via legacy {method_name}")
                    return tickers, "bloomberg_ticker"
            except Exception as e:
                print(f"[Universe Handshake] Legacy fallback '{method_name}' failed: {e}")
                
    raise RuntimeError("Critical: Failed to resolve the official universe across all channels.")

# ==========================================
# STOOQ HISTORICAL DATA DOWNLOADER
# ==========================================
def download_stooq_ticker(symbol: str) -> pl.DataFrame:
    """
    Downloads historical EOD CSV daily data from Stooq for a single symbol.
    Returns a Polars DataFrame with columns: date, ticker, close.
    """
    parts = symbol.split(" ")
    if len(parts) != 2:
        return None
    ticker_name, country = parts[0], parts[1].upper()
    
    # Align countries with Stooq's suffixes
    if country == "GB":
        stooq_symbol = f"{ticker_name}.UK"
    else:
        stooq_symbol = f"{ticker_name}.{country}"
        
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200 or len(response.text.strip()) < 100:
            return None
        
        # Load CSV using pandas and convert to Polars
        df_pd = pd.read_csv(io.StringIO(response.text))
        if df_pd.empty or "Close" not in df_pd.columns or "Date" not in df_pd.columns:
            return None
            
        df = pl.from_pandas(df_pd)
        df = df.select([
            pl.col("Date").alias("date"),
            pl.col("Close").alias("close")
        ])
        df = df.with_columns(pl.lit(symbol).alias("ticker"))
        return df
    except Exception:
        return None

def download_historical_prices_stooq(symbols: list) -> pl.DataFrame:
    """
    Downloads historical daily data from Stooq for multiple symbols concurrently.
    """
    print(f"[Data Downloader] Concurrently downloading {len(symbols)} tickers from Stooq...")
    dfs = []
    
    # Utilize threaded workers for concurrency under I/O bounds
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_symbol = {executor.submit(download_stooq_ticker, sym): sym for sym in symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            res = future.result()
            if res is not None:
                dfs.append(res)
                
    if not dfs:
        raise ValueError("Critical Error: All market data downloads failed from Stooq.")
        
    return pl.concat(dfs)

# ==========================================
# RISK SHIELD (FRED T10Y2Y)
# ==========================================
def get_latest_yield_curve_spread() -> float:
    """
    Fetches the latest T10Y2Y spread from FRED.
    Returns 1.0 (positive) as default fallback to prevent erroneous neutralization.
    """
    print("[Risk Shield] Fetching T10Y2Y spread from FRED...")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=T10Y2Y&api_key={FRED_API_KEY}&file_type=json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        observations = data.get("observations", [])
        
        for obs in reversed(observations):
            val_str = obs.get("value", "").strip()
            if val_str and val_str != ".":
                return float(val_str)
                
        print("[Risk Shield] Warning: No numeric yield spread value found in response.")
        return 1.0
    except Exception as e:
        print(f"[Risk Shield] Request failed: {e}. Defaulting spread to positive (safe mode).")
        return 1.0

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def main():
    # Setup credentials
    public_key = os.getenv("NUMERAI_PUBLIC_KEY")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    
    if not public_key or not secret_key:
        print("Error: Missing NUMERAI_PUBLIC_KEY or NUMERAI_SECRET_KEY environment variables.")
        sys.exit(1)
        
    sapi = numerapi.SignalsAPI(public_key, secret_key)
    
    # 1. Official Universe Handshake
    eligible_tickers, target_col = fetch_official_universe(sapi)
    print(f"Retrieved official universe of {len(eligible_tickers)} active tickers using target column: '{target_col}'")
    
    # Filter US liquid segment to preserve rate limits and ensure maximum data density
    us_tickers = [t for t in eligible_tickers if t.endswith(" US")]
    
    # Deterministic sampling (800 stocks provides a highly diverse baseline portfolio)
    random.seed(42)
    tickers_to_fetch = random.sample(us_tickers, min(len(us_tickers), 800))
    print(f"Filtered universe to {len(tickers_to_fetch)} liquid US assets for target download.")
    
    # 2. Parallel Stooq Historical Data Retrieval (Bypasses Yahoo rate-limits)
    prices_df = download_historical_prices_stooq(tickers_to_fetch)
    print(f"[Diagnostics] prices_df raw download shape: {prices_df.shape}")
    
    # 3. Alpha Calculation: MScFE 20-Day Mean Reversion
    print("Computing rolling mean reversion z-scores...")
    prices_df = prices_df.sort(["ticker", "date"])
    
    # Generate 20-day MA and Std Dev using Polars expressions
    prices_df = prices_df.with_columns([
        pl.col("close").rolling_mean(window_size=20).over("ticker").alias("ma_20"),
        pl.col("close").rolling_std(window_size=20).over("ticker").alias("std_20")
    ])
    
    # Drop rows without enough history
    prices_df = prices_df.filter(
        (pl.col("std_20").is_not_null()) & (pl.col("std_20") > 1e-8)
    )
    print(f"[Diagnostics] prices_df post-std_20 filtering shape: {prices_df.shape}")
    
    if prices_df.height == 0:
        raise ValueError("Critical Abort: prices_df was emptied after filtering. Check Stooq output formats.")

    # Calculate Z-score
    prices_df = prices_df.with_columns(
        ((pl.col("close") - pl.col("ma_20")) / pl.col("std_20")).alias("z_score")
    )
    
    # Extract latest signals for each ticker
    latest_signals = prices_df.group_by("ticker").last()
    print(f"[Diagnostics] latest_signals pre-mapping shape: {latest_signals.shape}")
    
    # Apply Mean Reversion: Inverse of Z-score
    latest_signals = latest_signals.with_columns(
        (pl.col("z_score") * -1.0).alias("raw_signal")
    )
    
    # Uniform rank generation (scaling to 0-1)
    latest_signals = latest_signals.with_columns(
        pl.col("raw_signal").rank(method="average").alias("rank")
    )
    num_signals = latest_signals.height
    latest_signals = latest_signals.with_columns(
        (pl.col("rank") / (num_signals + 1)).alias("signal")
    )
    
    # Clean up column mapping names to align join keys
    latest_signals = latest_signals.rename({"ticker": target_col})
    
    # 4. Risk Shield Check (FRED Macro Overlay)
    spread = get_latest_yield_curve_spread()
    print(f"FRED Yield Spread: {spread}%")
    
    if spread < 0.0:
        print("[Risk Shield] Negative spread detected. Overriding all signals to 0.5 (Neutral).")
        latest_signals = latest_signals.with_columns(
            pl.lit(0.5).alias("signal")
        )
        
    # 5. Complete Universe Align & Safe-Fill
    # Left-join computed signals to the downloaded active universe to ensure complete coverage.
    universe_df = pl.DataFrame({target_col: eligible_tickers})
    submission_df = universe_df.join(
        latest_signals.select([target_col, "signal"]),
        on=target_col,
        how="left"
    )
    
    # Calculate and log mapping statistics
    successfully_joined = submission_df.filter(pl.col("signal").is_not_null()).height
    print(f"[Sanity Check] Successfully mapped and calculated signals for {successfully_joined} out of {universe_df.height} tickers.")
    
    # Ensure any missed tickers default to neutral 0.5 to satisfy the submission volume requirement
    submission_df = submission_df.with_columns(
        pl.col("signal").fill_null(0.5)
    )
    
    # Ensure strict adherence to (0, 1) exclusive boundaries
    submission_df = submission_df.with_columns(
        pl.col("signal").clip(lower_bound=0.01, upper_bound=0.99)
    )
    
    # 6. Pre-flight Validation (Explicitly enforce non-zero Standard Deviation)
    signal_std = submission_df["signal"].std()
    print(f"[Sanity Check] Signal vector standard deviation: {signal_std}")
    
    if signal_std is None or signal_std < 1e-6:
        raise ValueError(
            f"Pre-flight Abort: Calculated submission standard deviation is {signal_std}. "
            f"This indicates a formatting mismatch between model outputs and the target universe. "
            f"Aborting upload to preserve submission quota."
        )
        
    # Write to local file
    final_output = submission_df.select([target_col, "signal"])
    output_path = "submission.csv"
    final_output.write_csv(output_path)
    print(f"Generated submission file with {final_output.height} valid tickers.")
    
    # Clean up local file to free disk space
    if os.path.exists("live.parquet"):
        os.remove("live.parquet")
    
    # 7. Model ID Validation & Upload
    print("Finding model ID programmatically...")
    try:
        models = sapi.get_models()
        model_id = models.get(MODEL_SLOT_NAME)
        
        # Self-healing case-insensitive search if mismatch exists
        if not model_id:
            for m_name, m_uuid in models.items():
                if m_name.lower().strip() == MODEL_SLOT_NAME:
                    model_id = m_uuid
                    print(f"Matched slot programmatically: '{m_name}' -> {model_id}")
                    break
                    
        if not model_id:
            raise ValueError(f"Could not map slot '{MODEL_SLOT_NAME}' in active models: {list(models.keys())}")
            
        print(f"Uploading submission file to model ID: {model_id}...")
        sapi.upload_predictions(output_path, model_id=model_id)
        print("Pipeline execution completed successfully.")
        
    except Exception as e:
        print(f"Submission failed during upload step: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
