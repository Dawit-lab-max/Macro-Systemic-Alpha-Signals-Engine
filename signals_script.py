import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime
import gc

# 1. INSTITUTIONAL CREDENTIALS 
# .strip() handles any hidden spaces from the GitHub Secret vault
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. MACRO REGIME FILTER (FRED Integration)
def get_systemic_risk_filter():
    try:
        print("Connecting to St. Louis Fed (FRED)...")
        start = datetime.datetime.now() - datetime.timedelta(days=60)
        # Pulling the 10Y-2Y Treasury Spread
        spread_data = web.DataReader('T10Y2Y', 'fred', start)
        current_spread = spread_data.iloc[-1].values[0]
        return 0.5 if current_spread < 0 else 1.0
    except Exception as e:
        print(f"FRED link unstable: {e}. Reverting to Neutral Risk.")
        return 1.0

# 3. ALPHA GENERATION (MScFE 610 Mean Reversion Logic)
def generate_signals(ticker_data, risk_multiplier):
    # Log-return normalization
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    
    # The 'Sword': Buy low Z, Sell high Z
    raw_alpha = -z_score.iloc[-1] 
    return raw_alpha * risk_multiplier

# 4. UNIVERSE SELECTION (Liquid Blue Chips)
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", 
    "PG", "HD", "DIS", "BAC", "VZ", "ADBE", "NFLX", "INTC", "PFE", "T", "XOM"
]

if __name__ == "__main__":
    print("Executing V4 Macro-Systemic Pipeline...")
    
    # Trigger Macro Shield
    multiplier = get_systemic_risk_filter()
    
    # Download Equity Data
    # progress=False prevents log clutter in GitHub Actions
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    data = data.ffill().fillna(0)

    # Process Signals
    final_alpha = generate_signals(data, multiplier)
    
    # Normalize to [0, 1] as required by the fund
    predictions = final_alpha.rank(pct=True)

    # 5. FORMATTING FOR NUMERAI SIGNALS
    submission = predictions.reset_index()
    submission.columns = ["ticker", "signal"]
    # Ticker format must be 'TICKER US'
    submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")

    submission.to_csv("signals_upload.csv", index=False)
    
    try:
        # Push to the DAWITYIMER model slot
        sapi.upload_predictions("signals_upload.csv", model_id="DAWITYIMER")
        print("TRANSMISSION SUCCESSFUL: SYSTEMIC ALPHA CAPTURED.")
    except Exception as e:
        print(f"Hedge Fund API Error: {e}")
        raise e
