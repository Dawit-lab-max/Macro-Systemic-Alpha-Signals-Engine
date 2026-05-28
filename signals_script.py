import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime

# 1. INSTITUTIONAL AUTHENTICATION
# My authorized FRED Key
FRED_API_KEY = "3699988d98d460d752a241f85df9532f"

# Numerai Keys from GitHub Secrets
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. THE "SHIELD": MACRO REGIME FILTER (GWP 610 Logic)
def get_risk_multiplier():
    """
    Polls the Federal Reserve for the 10Y-2Y Treasury Spread.
    Determines if we are in a 'Risk-On' or 'Risk-Off' regime.
    """
    try:
        print("Connecting to St. Louis Fed Terminal...")
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        # Authenticated pull using your provided key
        spread_data = web.get_data_fred('T10Y2Y', start, datetime.datetime.now(), api_key=FRED_API_KEY)
        latest_spread = spread_data.iloc[-1].values[0]
        print(f"Institutional Signal: 10Y-2Y Spread at {latest_spread}")
        
        # Inverted Yield Curve (<0) = Systemic Danger. Shrink signals to protect capital.
        return 0.5 if latest_spread < 0 else 1.0
    except Exception as e:
        print(f"FRED Link Warning: {e}. Defaulting to Neutral Risk.")
        return 1.0

# 3. THE "SWORD": ALPHA SIGNAL EXTRACTION (GWP 632 Logic)
def generate_mean_reversion_signals(ticker_data, multiplier):
    """
    Stochastic Mean Reversion: Identifies price-action 'glitches'.
    """
    # 20-day Z-Score Calculation
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    
    # Reverse the Z-score (Mean Reversion Alpha)
    alpha = -z_score.iloc[-1] 
    return alpha * multiplier

# 4. UNIVERSE DEFINITION (Liquid S&P 100)
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", 
    "PG", "HD", "DIS", "BAC", "VZ", "ADBE", "NFLX", "INTC", "PFE", "T", "XOM"
]

if __name__ == "__main__":
    print("Commencing V4.5 Production Run: Macro-Systemic-Alpha-Signals-Engine")
    
    # Trigger Macro Shield
    risk_multiplier = get_risk_multiplier()
    
    # Download Equity Stream (Yahoo Finance)
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    data = data.ffill().fillna(0)

    # Process Alpha
    raw_alpha = generate_mean_reversion_signals(data, risk_multiplier)
    predictions = raw_alpha.rank(pct=True)

    # 5. DYNAMIC HANDSHAKE (Finding UUID for Handle 'dawityimer')
    try:
        all_models = sapi.get_models()
        model_uuid = all_models.get('dawityimer')
        
        if not model_uuid:
            print("ERROR: Model 'dawityimer' not found in this account.")
            exit(1)
            
        print(f"Targeting Model UUID: {model_uuid}")

        # 6. FORMATTING (Standardizing for Bloomberg/Reuters format)
        submission = predictions.reset_index()
        submission.columns = ["ticker", "signal"]
        submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")

        # 7. FINAL DELIVERY
        submission.to_csv("signals_upload.csv", index=False)
        sapi.upload_predictions("signals_upload.csv", model_id=model_uuid)
        print("SUCCESS: SYSTEMIC ALPHA DELIVERED TO HEDGE FUND.")
        
    except Exception as e:
        print(f"CRITICAL SYSTEM FAILURE: {e}")
        raise e
