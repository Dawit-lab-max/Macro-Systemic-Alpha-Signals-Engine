import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime

# 1. INSTITUTIONAL AUTHENTICATION
# Setting FRED key globally to fix the library compatibility bug
os.environ["FRED_API_KEY"] = "3699988d98d460d752a241f85df9532f"

# Numerai Credentials from your GitHub Secrets vault
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. MACRO REGIME FILTER (FRED Data)
def get_risk_multiplier():
    """
    Uses the 10Y-2Y Treasury Spread to determine market stress.
    An inverted curve indicates high systemic risk.
    """
    try:
        print("Accessing Federal Reserve (FRED) for Systemic Risk levels...")
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        # Direct pull using the pinned pandas-datareader logic
        spread_data = web.get_data_fred('T10Y2Y', start)
        latest_spread = spread_data.iloc[-1].values[0]
        print(f"Institutional Signal: 10Y-2Y Spread is {latest_spread}")
        # MScFE Logic: If spread < 0 (Inverted), reduce conviction to preserve capital
        return 0.5 if latest_spread < 0 else 1.0
    except Exception as e:
        print(f"FRED Auth Bypass: {e}. Defaulting to safe risk (1.0).")
        return 1.0

# 3. ALPHA GENERATION (MScFE Mean Reversion Thesis)
def generate_signals(ticker_data, risk_multiplier):
    """
    Calculates 20-day Mean Reversion Z-Scores.
    """
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    
    # Identify the 'Glitch' (The Alpha): Invert the Z-score
    raw_alpha = -z_score.iloc[-1] 
    return raw_alpha * risk_multiplier

# 4. UNIVERSE SELECTION (Liquid S&P 100 Assets)
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", 
    "PG", "HD", "DIS", "BAC", "VZ", "ADBE", "NFLX", "INTC", "PFE", "T", "XOM"
]

if __name__ == "__main__":
    print("Executing Macro-Systemic Signals Pipeline...")
    
    # Get Macro Multiplier
    multiplier = get_risk_multiplier()
    
    # Download Market Data (Yahoo Finance)
    # progress=False prevents log clutter
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    data = data.ffill().fillna(0)

    # Process Alpha
    final_alpha = generate_signals(data, multiplier)
    
    # Rank between 0 and 1 for the Hedge Fund
    predictions = final_alpha.rank(pct=True)

    # 5. DYNAMIC UUID HANDSHAKE (Case-Insensitive Fix)
    try:
        print("Scanning Signals Dashboard for model slots...")
        my_models = sapi.get_models()
        
        # Searching for model handle 'dawityimer'
        target_handle = "dawityimer"
        target_uuid = None
        
        for name, uuid in my_models.items():
            if name.lower() == target_handle.lower():
                target_uuid = uuid
                break
        
        if not target_uuid:
            print(f"ERROR: No slot named '{target_handle}' found at numer.ai/signals.")
            exit(1)

        print(f"Model UUID Verified: {target_uuid}")

        # 6. FORMATTING FOR SUBMISSION
        submission = predictions.reset_index()
        submission.columns = ["ticker", "signal"]
        # Convert to Bloomberg/Reuters ticker format
        submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")
        
        # Save and Upload
        submission.to_csv("signals_upload.csv", index=False)
        sapi.upload_predictions("signals_upload.csv", model_id=target_uuid)
        print("SUCCESS: MACRO-SYSTEMIC ALPHA DELIVERED TO HEDGE FUND.")
        
    except Exception as e:
        print(f"CRITICAL SYSTEM ERROR: {e}")
        raise e
