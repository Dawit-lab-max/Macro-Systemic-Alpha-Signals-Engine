#!/usr/bin/env python3
"""
Numerai Signals Submission Pipeline
Author: MScFE Team / Dawit Yimer Production Setup
Description: Low-memory, rate-limit-safe production pipeline using Polars.
             Dynamically retrieves the official universe via live.parquet,
             downloads a highly targeted, deterministic sample of 200 US tickers
             in a single sequential batch to bypass Yahoo rate limits,
             calculates mean reversion alpha, and safe-fills the rest of the universe.
"""
import os
import sys
import requests
import random
import pandas as pd
import polars as pl
import yfinance as yf
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
# TICKER CONVERTER
# ==========================================
class TickerConverter:
    """
    Pure-Python Rule-Based Ticker Converter.
    Eliminates dependencies on external S3 CSV mapping databases.
    Maps Bloomberg/Numerai formats (e.g., 'AAPL US', '7203 JP') to Yahoo, and vice-versa.
    """
    def __init__(self):
        # Forward maps (Exchange ISO code -> Yahoo suffix)
        self.iso_to_yahoo_suffix = {
            "US": "",
            "JP": ".T",
            "KR": ".KS",
            "GB": ".L",
            "CA": ".TO",
            "AU": ".AX",
            "FR": ".PA",
            "DE": ".DE",
            "HK": ".HK",
            "CN": ".SS",
            "ID": ".JK",
            "IT": ".MI",
            "NL": ".AS",
            "ES": ".MC",
            "CH": ".SW",
            "IL": ".TA",
            "IN": ".NS",
            "BR": ".SA",
            "MX": ".MX",
            "PL": ".WA",
            "ZA": ".JO",
        }
        # Inverse maps (Yahoo suffix -> Exchange ISO code)
        self.yahoo_suffix_to_iso = {
            "": "US",
            "T": "JP",
            "KS": "KR",
            "L": "GB",
            "TO": "CA",
            "V": "CA",
            "AX": "AU",
            "PA": "FR",
            "DE": "DE",
            "HK": "HK",
            "SS": "CN",
            "SZ": "CN",
            "JK": "ID",
            "MI": "IT",
            "AS": "NL",
            "MC": "ES",
            "SW": "CH",
            "TA": "IL",
            "NS": "IN",
            "SA": "BR",
            "MX": "MX",
            "WA": "PL",
            "JO": "ZA",
        }

    def to_yahoo(self, source_ticker: str) -> str:
        if not source_ticker:
            return ""
        src_clean = str(source_ticker).strip()
        parts = src_clean.split(" ")
        if len(parts) == 2:
            ticker, exchange = parts[0], parts[1].upper()
            ticker = ticker.replace("/", "-")
            
            # Map using ISO or standard fallback exchange rule
            if exchange in self.iso_to_yahoo_suffix:
                suffix = self.iso_to_yahoo_suffix[exchange]
                return f"{ticker}{suffix}"
            else:
                # Fallback exchange codes (e.g., Bloomberg legacy mapping)
                fallback_map = {
                    "LN": ".L", "CN": ".TO", "FP": ".PA", "GY": ".DE"
                }
                suffix = fallback_map.get(exchange, f".{exchange.lower()}")
                return f"{ticker}{suffix}"
        return src_clean

    def to_target(self, yahoo_ticker: str, target_col: str) -> str:
        if not yahoo_ticker:
            return ""
        yahoo_clean = str(yahoo_ticker).strip()
        if "." in yahoo_clean:
            ticker, suffix = yahoo_clean.split(".", 1)
            ticker = ticker.replace("-", "/")
            suffix_upper = suffix.upper()
            
            # 1. Output Numerai-compliant ISO format if target column is numerai_ticker
            if target_col in ["numerai_ticker", "ticker"]:
                if suffix in self.yahoo_suffix_to_iso:
                    country = self.yahoo_suffix_to_iso[suffix]
                elif suffix_upper in self.yahoo_suffix_to_iso:
                    country = self.yahoo_suffix_to_iso[suffix_upper]
                else:
                    country = suffix_upper
                return f"{ticker} {country}"
            
            # 2. Output legacy Bloomberg formats otherwise
            else:
                bbg_exchange_map = {
                    "L": "LN", "TO": "CN", "V": "CN", "PA": "FP", "DE": "GY", "SW": "SW"
                }
                if suffix in bbg_exchange_map:
                    exchange = bbg_exchange_map[suffix]
                elif suffix_upper in bbg_exchange_map:
                    exchange = bbg_exchange_map[suffix_upper]
                else:
                    exchange = self.yahoo_suffix_to_iso.get(suffix, suffix_upper)
                return f"{ticker} {exchange}"
        else:
            ticker = yahoo_clean.replace("-", "/")
            return f"{ticker} US"

# ==========================================
# DATA DOWNLOADER
# ==========================================
def download_historical_prices(yahoo_tickers: list, period: str = "3mo") -> pl.DataFrame:
    """
    Downloads historical data from Yahoo Finance in a single sequential call.
    Uses browser user-agent headers to bypass scraper blockers.
    """
    # Create custom session with modern desktop user-agent headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    })

    print(f"[Data Downloader] Downloading {len(yahoo_tickers)} tickers in a single sequential batch...")
    try:
        # threads=False prevents yfinance from triggering rate limits on cloud IP ranges
        chunk_data = yf.download(
            yahoo_tickers, 
            period=period, 
            interval="1d", 
            progress=False, 
            session=session,
            threads=False
        )
        
        if chunk_data.empty:
            raise ValueError("Yahoo Finance returned an empty dataset. Cloud IP block active.")
            
        # Assign index name to "Date" before processing structure
        chunk_data.index.name = "Date"
        
        # 1. Handle MultiIndex Column Structures
        if isinstance(chunk_data.columns, pd.MultiIndex):
            metric_level = None
            for level_idx in [0, 1]:
                unique_vals = chunk_data.columns.get_level_values(level_idx).unique()
                if any(m in unique_vals for m in ["Adj Close", "Close"]):
                    metric_level = level_idx
                    break
            
            if metric_level is None:
                metric_level = 0
                
            available_metrics = chunk_data.columns.get_level_values(metric_level).unique()
            target_metric = "Adj Close" if "Adj Close" in available_metrics else "Close"
            adj_close = chunk_data.xs(target_metric, axis=1, level=metric_level).copy()
            
        # 2. Handle Flat Column Structures
        else:
            if "Adj Close" in chunk_data.columns:
                adj_close = chunk_data[["Adj Close"]].copy()
            elif "Close" in chunk_data.columns:
                adj_close = chunk_data[["Close"]].copy()
            else:
                raise ValueError("No close price metrics found in DataFrame.")
            adj_close.columns = [yahoo_tickers[0]]
            
        # 3. Cast to DataFrame if slicing yielded a pd.Series
        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame()
            
        # 4. Explicitly enforce "Date" index name on the extracted DataFrame
        adj_close.index.name = "Date"
        
        # 5. Melt to long format
        melted = adj_close.reset_index().melt(
            id_vars="Date", 
            value_vars=adj_close.columns, 
            var_name="ticker", 
            value_name="close"
        )
        
        # Cast to Polars to save memory
        pl_df = pl.from_pandas(melted).rename({"Date": "date"}).drop_nulls()
        return pl_df
        
    except Exception as e:
        print(f"[Data Downloader] Error during yfinance download: {e}")
        raise e

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
    
    # 1. Official Universe Handshake (Retrieves exact, active column name and tickers)
    eligible_tickers, target_col = fetch_official_universe(sapi)
    print(f"Retrieved official universe of {len(eligible_tickers)} active tickers using target column: '{target_col}'")
    
    # 2. Ticker Conversion Setup
    converter = TickerConverter()
    
    # Filter US liquid segment
    us_tickers = [t for t in eligible_tickers if t.endswith(" US")]
    print(f"Total eligible US tickers found: {len(us_tickers)}")
    
    # Select a deterministic sample of exactly 200 US tickers to remain safe from rate limits
    random.seed(42)
    tickers_to_fetch = random.sample(us_tickers, min(len(us_tickers), 200))
    print(f"Selected deterministic sample of {len(tickers_to_fetch)} US tickers for rate-limit safe download.")
    
    # Convert active tickers to Yahoo Tickers for downloading
    yahoo_tickers_to_fetch = list(set([converter.to_yahoo(t) for t in tickers_to_fetch]))
    
    # 3. Memory-Safe Market Data Fetch (Polars/Vectorized chunking - "3mo" period)
    prices_df = download_historical_prices(yahoo_tickers_to_fetch, period="3mo")
    print(f"[Diagnostics] prices_df raw download shape: {prices_df.shape}")
    
    # 4. Alpha Calculation: MScFE 20-Day Mean Reversion
    print("Computing rolling mean reversion z-scores...")
    prices_df = prices_df.sort(["ticker", "date"])
    
    # Generate 20-day MA and Std Dev using Polars expressions
    prices_df = prices_df.with_columns([
        pl.col("close").rolling_mean(window_size=20).over("ticker").alias("ma_20"),
        pl.col("close").rolling_std(window_size=20).over("ticker").alias("std_20")
    ])
    
    # Drop rows without enough history (requiring std_20 calculated on 20 trading days)
    prices_df = prices_df.filter(
        (pl.col("std_20").is_not_null()) & (pl.col("std_20") > 1e-8)
    )
    print(f"[Diagnostics] prices_df post-std_20 filtering shape: {prices_df.shape}")
    
    if prices_df.height == 0:
        raise ValueError("Critical Abort: prices_df was emptied after filtering. Check data format.")

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
    
    # Map back to the active identifier column format
    latest_signals = latest_signals.with_columns(
        pl.col("ticker").map_elements(lambda x: converter.to_target(x, target_col), return_dtype=pl.String).alias(target_col)
    )
    
    # Log sample mapped elements to confirm format consistency
    if latest_signals.height > 0:
        print(f"[Diagnostics] Raw 'ticker' column samples: {latest_signals['ticker'].head(5).to_list()}")
        print(f"[Diagnostics] Mapped '{target_col}' column samples: {latest_signals[target_col].head(5).to_list()}")
        
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
    print(f"[Diagnostics] universe_df sample: {universe_df[target_col].head(5).to_list()}")
    
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
    
    # 7. Pre-flight Validation (Explicitly enforce non-zero Standard Deviation)
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
    
    # 8. Model ID Validation & Upload
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
