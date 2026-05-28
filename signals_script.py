import os
import sys
import pandas as pd
import polars as pl
import yfinance as yf
import numerapi
import numpy as np
import datetime

# ==========================================
# INSTITUTIONAL CONFIGURATION
# ==========================================
MODEL_SLOT_NAME = "dawityimer"

# ==========================================
# ALPHA ENGINE (MScFE 610/632 Logic)
# ==========================================
def main():
    # 1. Handshake & Universe
    public_key = os.getenv("NUMERAI_PUBLIC_KEY")
    secret_key = os.getenv("NUMERAI_SECRET_KEY")
    sapi = numerapi.SignalsAPI(public_key, secret_key)
    
    print("[1/5] Extracting official eligible universe...")
    eligible_tickers = sapi.get_eligible_tickers()
    
    # MLOps Move: We prioritize the top 1200 tickers to prevent Yahoo Throttling
    # This ensures high 'Data Density' so our math doesn't crash
    target_tickers = eligible_tickers[:1200] 
    
    # Map to Yahoo format
    yahoo_map = [t.replace(".", "-").replace(" US", "") for t in target_tickers]

    # 2. High-Performance Data Acquisition
    print(f"[2/5] Downloading 120 days of history for {len(yahoo_map)} assets...")
    # Use auto_adjust and small chunks to avoid 'HTTP 403' blocks
    raw_data = yf.download(yahoo_map, period="120d", interval="1d", auto_adjust=True, progress=False, threads=True)
    
    if raw_data.empty:
        print("CRITICAL: Yahoo Finance returned no data. Aborting to save quota.")
        sys.exit(1)

    prices = raw_data['Close'].ffill().tail(60) # Keep 60 days for stable math
    
    # 3. MScFE Forensic Math: Mean Reversion Z-Score
    print("[3/5] Executing Stochastic Signal Extraction...")
    # Calculate Log Returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Signal = (Price - 20d Mean) / 20d StdDev
    m_avg = prices.rolling(window=20).mean()
    m_std = prices.rolling(window=20).std()
    z_score = (prices - m_avg) / m_std
    
    # The Sword: Invert Z-Score for Mean Reversion
    # The Shield: Weight by Inverse Volatility (GWP 610 Logic)
    volatility = log_returns.rolling(window=20).std()
    raw_alpha = (-z_score / volatility).iloc[-1] # Most recent day

    # 4. Clean and Rank (The 'Cope Up' Strategy)
    alpha_clean = raw_alpha.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Data Density: {len(alpha_clean)} stocks survived the MScFE Audit.")
    
    if len(alpha_clean) < 100:
        print("ERROR: Not enough data density for a valid signal. Check yfinance connection.")
        sys.exit(1)

    # Normalize to 0-1 for the Hedge Fund
    predictions = alpha_clean.rank(pct=True)

    # 5. Final Handshake & Delivery
    print("[4/5] Formatting for institutional delivery...")
    submission_df = predictions.reset_index()
    submission_df.columns = ["ticker", "signal"]
    
    # Convert back to Numerai format (AAPL US)
    submission_df["ticker"] = submission_df["ticker"].apply(lambda x: f"{x.replace('-', '.')} US")

    # Final Align: Ensure 100% of the universe is represented (Fill others with 0.5)
    universe_df = pd.DataFrame(eligible_tickers, columns=["ticker"])
    final_output = universe_df.merge(submission_df, on="ticker", how="left")
    final_output["signal"] = final_output["signal"].fillna(0.5)

    final_output.to_csv("submission.csv", index=False)

    # 6. Pushing to Fund
    print("[5/5] Locating Model ID programmatically...")
    try:
        models = sapi.get_models()
        model_id = next((uuid for name, uuid in models.items() if name.lower() == MODEL_SLOT_NAME), None)
        
        if not model_id:
            raise ValueError(f"Model {MODEL_SLOT_NAME} not found.")

        sapi.upload_predictions("submission.csv", model_id=model_id)
        print(f"SUCCESS: Systemic Alpha captured and delivered to {MODEL_SLOT_NAME}.")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
