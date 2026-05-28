#!/usr/bin/env python3
"""
Numerai Signals Submission Pipeline
Author: MScFE Team / Dawit Yimer Production Setup
Description: Low-memory, crash-resistant production pipeline using Polars.
             Dynamically retrieves the official universe via live.parquet,
             handles ticker mapping, and applies mean reversion and risk shields.
"""
import os
import sys
import requests
import pandas as pd
import polars as pl
import yfinance as yf
import numerapi

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
FRED_API_KEY = "3699988d98d460d752a241f85df9532f"
TICKER_MAP_URL = "https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv"
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
# TICKER CONVERTER
# ==========================================
class TickerConverter:
    """
    Maps Bloomberg & Numerai tickers to Yahoo Finance tickers, and vice-versa.
    Utilizes the official S3 mapping database with a programmatic fallback heuristic.
    """
    def __init__(self, ticker_map_url: str):
        print("[Ticker Map] Downloading mapping table from S3...")
        self.to_yahoo_map = {}
        self.to_target_map = {}  # yahoo -> target_ticker (supports multiple formats)
        
        try:
            df = pl.read_csv(ticker_map_url)
        except Exception as e:
            print(f"[Ticker Map] Polars read failed, using Pandas fallback: {e}")
            df = pl.from_pandas(pd.read_csv(ticker_map_url))
            
        for r in df.iter_rows(named=True):
            bbg = r.get("bloomberg_ticker")
            num_t = r.get("ticker")  # in the map file, 'ticker' represents the numerai_ticker
            yahoo = r.get("yahoo")
            
            if yahoo and str(yahoo).strip() != "" and str(yahoo) != "nan":
                yahoo_str = str(yahoo).strip()
                
                if bbg and str(bbg).strip() != "" and str(bbg) != "nan":
                    self.to_yahoo_map[str(bbg).strip()] = yahoo_str
                    
                if num_t and str(num_t).strip() != "" and str(num_t) != "nan":
                    self.to_yahoo_map[str(num_t).strip()] = yahoo_str
                
                self.to_target_map[yahoo_str] = {
                    "bloomberg_ticker": str(bbg).strip() if bbg else None,
                    "numerai_ticker": str(num_t).strip() if num_t else None,
                    "ticker": str(num_t).strip() if num_t else None
                }

    def to_yahoo(self, source_ticker: str) -> str:
        src_clean = source_ticker.strip()
        if src_clean in self.to_yahoo_map:
            return self.to_yahoo_map[src_clean]
        
        # Rule-based fallback heuristic if ticker is missing from the S3 database
        parts = src_clean.split(" ")
        if len(parts) == 2:
            ticker, exchange = parts[0], parts[1]
            ticker = ticker.replace("/", "-")
            
            exchange_map = {
                "US": "", "JP": ".T", "JT": ".T", "KS": ".KS", "LN": ".L",
                "CN": ".TO", "AU": ".AX", "FP": ".PA", "GY": ".DE", "HK": ".HK",
                "SS": ".SS", "SZ": ".SZ", "ID": ".JK", "IM": ".MI", "NA": ".AS",
                "SP": ".MC", "SW": ".SW", "TA": ".TA"
            }
            suffix = exchange_map.get(exchange, f".{exchange}")
            return f"{ticker}{suffix}" if suffix else ticker
            
        return src_clean

    def to_target(self, yahoo_ticker: str, target_col: str) -> str:
        yahoo_clean = yahoo_ticker.strip()
        if yahoo_clean in self.to_target_map:
            mapped_val = self.to_target_map[yahoo_clean].get(target_col)
            if mapped_val:
                return mapped_val
        
        # Heuristic fallback to translate Yahoo back to Bloomberg or Numerai formats
        if target_col in ["numerai_ticker", "ticker"]:
            if "." in yahoo_clean:
                ticker, suffix = yahoo_clean.split(".", 1)
                ticker = ticker.replace("-", "/")
                return f"{ticker} {suffix.upper()}"
            else:
                return f"{yahoo_clean} US"
        else:
            if "." in yahoo_clean:
                ticker, suffix = yahoo_clean.split(".", 1)
                ticker = ticker.replace("-", "/")
                inv_map = {
                    "T": "JP", "KS": "KS", "L": "LN", "TO": "CN", "AX": "AU",
                    "PA": "FP", "DE": "GY", "HK": "HK", "SS": "SS", "SZ": "SZ",
                    "JK": "ID", "MI": "IM", "AS": "NA", "MC": "SP", "SW": "SW", "TA": "TA"
                }
                exchange = inv_map.get(suffix, suffix.upper())
                return f"{ticker} {exchange}"
            else:
                return f"{yahoo_clean} US"

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
# DATA DOWNLOADER
# ==========================================
def download_historical_prices(yahoo_tickers: list, period: str = "60d") -> pl.DataFrame:
    """
    Downloads historical data from Yahoo Finance in optimal chunks.
    Melted to long format immediately to prevent RAM blowouts under 7GB limit.
    """
    chunk_size = 400
    dfs = []
    
    for i in range(0, len(yahoo_tickers), chunk_size):
        chunk = yahoo_tickers[i:i + chunk_size]
        print(f"[Data Downloader] Batch {i // chunk_size + 1}/{(len(yahoo_tickers) - 1) // chunk_size + 1}...")
        try:
            chunk_data = yf.download(chunk, period=period, interval="1d", progress=False)
            if chunk_data.empty:
                continue
                
            # Safely extract Adjusted Close
            if "Adj Close" in chunk_data.columns:
                adj_close = chunk_data["Adj Close"]
            elif isinstance(chunk_data.columns, pd.MultiIndex):
                adj_close = chunk_data.xs("Adj Close", axis=1, level=1)
            else:
                adj_close = chunk_data
                
            # Reshape into long-format Pandas DataFrame
            melted = adj_close.reset_index().melt(
                id_vars="Date", 
                value_vars=adj_close.columns, 
                var_name="ticker", 
                value_name="close"
            )
            
            # Instantly cast to Polars to save memory
            pl_df = pl.from_pandas(melted).rename({"Date": "date"}).drop_nulls()
            dfs.append(pl_df)
            
        except Exception as e:
            print(f"[Data Downloader] Warning: Failed downloading batch starting at index {i}: {e}")
            continue
            
    if not dfs:
        raise ValueError("Critical: All market data downloads failed.")
        
    return pl.concat(dfs)

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
    
    # 1. Official Universe Handshake (Retrieves exact, active column name and tickers)
    eligible_tickers, target_col = fetch_official_universe(sapi)
    print(f"Retrieved official universe of {len(eligible_tickers)} active tickers using target column: '{target_col}'")
    
    # 2. Ticker Conversion Setup
    converter = TickerConverter(TICKER_MAP_URL)
    
    # Convert active tickers to Yahoo Tickers for downloading
    yahoo_tickers_to_fetch = list(set([converter.to_yahoo(t) for t in eligible_tickers]))
    print(f"Mapped {len(eligible_tickers)} source tickers to {len(yahoo_tickers_to_fetch)} unique Yahoo tickers.")
    
    # 3. Memory-Safe Market Data Fetch (Polars/Vectorized chunking)
    prices_df = download_historical_prices(yahoo_tickers_to_fetch, period="60d")
    
    # 4. Alpha Calculation: MScFE 20-Day Mean Reversion
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
    
    # Calculate Z-score
    prices_df = prices_df.with_columns(
        ((pl.col("close") - pl.col("ma_20")) / pl.col("std_20")).alias("z_score")
    )
    
    # Extract latest signals for each ticker
    latest_signals = prices_df.group_by("ticker").last()
    
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
    
    # Map back to the active identifier column format
    latest_signals = latest_signals.with_columns(
        pl.col("ticker").map_elements(lambda x: converter.to_target(x, target_col), return_dtype=pl.String).alias(target_col)
    )
    
    # 5. Risk Shield Check (FRED Macro Overlay)
    spread = get_latest_yield_curve_spread()
    print(f"FRED Yield Spread: {spread}%")
    
    if spread < 0.0:
        print("[Risk Shield] Negative spread detected. Overriding all signals to 0.5 (Neutral).")
        latest_signals = latest_signals.with_columns(
            pl.lit(0.5).alias("signal")
        )
        
    # 6. Complete Universe Align & Safe-Fill
    # Left-join computed signals to the downloaded active universe to ensure complete coverage.
    universe_df = pl.DataFrame({target_col: eligible_tickers})
    submission_df = universe_df.join(
        latest_signals.select([target_col, "signal"]),
        on=target_col,
        how="left"
    )
    
    # Ensure any missed tickers default to neutral 0.5 to satisfy the submission volume requirement
    submission_df = submission_df.with_columns(
        pl.col("signal").fill_null(0.5)
    )
    
    # Ensure strict adherence to (0, 1) exclusive boundaries
    submission_df = submission_df.with_columns(
        pl.col("signal").clip(lower_bound=0.01, upper_bound=0.99)
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
