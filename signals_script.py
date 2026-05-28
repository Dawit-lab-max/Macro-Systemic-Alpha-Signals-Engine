import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime

# 1. INSTITUTIONAL AUTHENTICATION
# Setting FRED key globally to fix the 'unexpected keyword' bug
os.environ["FRED_API_KEY"] = "3699988d98d460d752a241f85df9532f"

PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. THE "SHIELD": FRED MACRO FILTER
def get_risk_multiplier():
    try:
        print("Polling Federal Reserve (FRED) for Systemic Risk...")
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        # Pulling the 10Y-2Y Treasury Spread
        spread_data = web.get_data_fred('T10Y2Y', start)
        latest_spread = spread_data.iloc[-1].values[0]
        print(f"Institutional Signal: 10Y-2Y Spread is {latest_spread}")
        return 0.5 if latest_spread < 0 else 1.0
    except Exception as e:
        print(f"FRED Auth Bypass: {e}. Defaulting to safe risk (1.0).")
        return 1.0

# 3. THE "SWORD": ALPHA EXTRACTION (MScFE 632 logic)
def generate_signals(ticker_data, risk_multiplier):
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    # Statistical Mean Reversion
    raw_alpha = -z_score.iloc[-1] 
    return raw_alpha * risk_multiplier

# 4. UNIVERSE SELECTION
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", 
    "PG", "HD", "DIS", "BAC", "VZ", "ADBE", "NFLX", "INTC", "PFE", "T", "XOM"
]

if __name__ == "__main__":
    print("Executing V4.7 Final Production Logic...")
    
    multiplier = get_risk_multiplier()
    
    # yfinance cache fix for cloud environments
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    data = data.ffill().fillna(0)

    final_alpha = generate_signals(data, multiplier)
    predictions = final_alpha.rank(pct=True)

    # 5. DYNAMIC MODEL SCANNER (The 'Model Not Found' Fix)
    try:
        # Fetching every model in your Signals account
        print("Scanning Signals Dashboard for model slots...")
        my_models = sapi.get_models()
        print(f"Models found in your account: {my_models}")
        
        # Searching for 'dawityimer' (Case-Insensitive)
        target_handle = "dawityimer"
        target_uuid = None
        
        for name, uuid in my_models.items():
            if name.lower() == target_handle.lower():
                target_uuid = uuid
                break
        
        if not target_uuid:
            print(f"CRITICAL ERROR: No model named '{target_handle}' found on the SIGNALS dashboard.")
            print("Action Required: Go to numer.ai/signals and create a model named 'dawityimer'.")
            exit(1)

        print(f"Verified UUID for {target_handle}: {target_uuid}")

        # 6. FORMATTING & DELIVERY
        submission = predictions.reset_index()
        submission.columns = ["ticker", "signal"]
        submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")
        submission.to_csv("signals_upload.csv", index=False)
        
        sapi.upload_predictions("signals_upload.csv", model_id=target_uuid)
        print(f"SUCCESS: V4.7 Signals delivered to UUID {target_uuid}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise e
