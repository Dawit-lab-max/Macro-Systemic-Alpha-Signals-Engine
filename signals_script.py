import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime
import gc

# 1. INSTITUTIONAL AUTHENTICATION
# Ensuring the FRED key is visible to the library environment
os.environ["FRED_API_KEY"] = os.getenv('FRED_API_KEY', '3699988d98d460d752a241f85df9532f').strip()

PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. MACRO REGIME FILTER (GWP 610 Logic)
def get_risk_multiplier():
    try:
        print("Connecting to St. Louis Fed (FRED)...")
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        # Pulling the 10Y-2Y Treasury Spread (The Systemic Risk Compass)
        spread_data = web.get_data_fred('T10Y2Y', start)
        latest_spread = spread_data.iloc[-1].values[0]
        print(f"Institutional Signal: 10Y-2Y Spread is {latest_spread}")
        return 0.5 if latest_spread < 0 else 1.0
    except Exception as e:
        print(f"FRED Auth Bypass: {e}. Defaulting to safe risk (1.0).")
        return 1.0

# 3. ALPHA GENERATION (MScFE 632 Mean Reversion)
def generate_signals(ticker_data, risk_multiplier):
    # Normalized Z-Score calculation
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    # Mean Reversion: Bet against the extreme movement
    raw_alpha = -z_score.iloc[-1] 
    return raw_alpha * risk_multiplier

# 4. UNIVERSE SELECTION (Liquid S&P Assets)
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", 
    "PG", "HD", "DIS", "BAC", "VZ", "ADBE", "NFLX", "INTC", "PFE", "T", "XOM"
]

if __name__ == "__main__":
    print("Executing V5.0 Macro-Systemic Signals Pipeline...")
    
    # Process Filter
    multiplier = get_risk_multiplier()
    
    # Download Equity Data
    # We disable threading to keep RAM usage extremely low for the 7GB cloud server
    data = yf.download(tickers, period="1y", interval="1d", progress=False, threads=False)['Close']
    data = data.ffill().fillna(0)

    # Process Alpha
    final_alpha = generate_signals(data, multiplier)
    predictions = final_alpha.rank(pct=True)

    # 5. DYNAMIC MODEL HANDSHAKE (The 'Model Not Found' Fix)
    try:
        print("Scanning Signals Dashboard for verified handle...")
        my_models = sapi.get_models()
        print(f"Visible Model Slots: {my_models}")
        
        target_handle = "dawityimer"
        target_uuid = None
        
        # Case-insensitive search for the model UUID
        for name, uuid in my_models.items():
            if name.lower() == target_handle.lower():
                target_uuid = uuid
                break
        
        if not target_uuid:
            raise ValueError(f"Model '{target_handle}' not found. Please create it at signals.numer.ai/dashboard")

        print(f"Handshaking with UUID: {target_uuid}")

        # 6. FORMATTING & DELIVERY
        submission = predictions.reset_index()
        submission.columns = ["ticker", "signal"]
        submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")
        submission.to_csv("signals_upload.csv", index=False)
        
        sapi.upload_predictions("signals_upload.csv", model_id=target_uuid)
        print("SUCCESS: V5.0 MACRO-SYSTEMIC ALPHA DELIVERED.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise e
