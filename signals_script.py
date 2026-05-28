import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime
import gc

# 1. SECURE CREDENTIALS
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. THE "SHIELD": FRED MACRO FILTER
def get_risk_multiplier():
    try:
        print("Polling FRED for Systemic Risk...")
        start = datetime.datetime.now() - datetime.timedelta(days=60)
        # Pulling the 10Y-2Y Treasury Spread
        spread_data = web.DataReader('T10Y2Y', 'fred', start)
        latest_spread = spread_data.iloc[-1].values[0]
        return 0.5 if latest_spread < 0 else 1.0
    except Exception as e:
        print(f"FRED link skipped: {e}")
        return 1.0

# 3. THE "SWORD": ALPHA GENERATION
def generate_final_signals(ticker_data, risk_multiplier):
    # Calculate Z-Score Mean Reversion
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    
    # Identify the 'Glitch' (The Alpha)
    raw_alpha = -z_score.iloc[-1] 
    return raw_alpha * risk_multiplier

# 4. UNIVERSE SELECTION
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", 
    "PG", "HD", "DIS", "BAC", "VZ", "ADBE", "NFLX", "INTC", "PFE", "T", "XOM"
]

if __name__ == "__main__":
    print("Executing Macro-Systemic Pipeline...")
    
    # 1. Get Macro Filter
    multiplier = get_risk_multiplier()
    
    # 2. Get Equity Data
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    data = data.ffill().fillna(0)

    # 3. Process Signals (FIXED FUNCTION CALL)
    final_alpha = generate_final_signals(data, multiplier)
    
    # 4. Normalize to [0, 1]
    predictions = final_alpha.rank(pct=True)

    # 5. FORMATTING
    submission = predictions.reset_index()
    submission.columns = ["ticker", "signal"]
    # Ticker format must be 'TICKER US'
    submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")

    # 6. DELIVERY
    submission.to_csv("signals_upload.csv", index=False)
    try:
        sapi.upload_predictions("signals_upload.csv", model_id="DAWITYIMER")
        print("SUCCESS: SYSTEMIC ALPHA DELIVERED.")
    except Exception as e:
        print(f"Hedge Fund API Error: {e}")
        raise e
