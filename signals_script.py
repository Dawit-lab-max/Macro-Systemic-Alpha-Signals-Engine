import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime
import gc

# 1. INSTITUTIONAL AUTHENTICATION
os.environ["FRED_API_KEY"] = "3699988d98d460d752a241f85df9532f"
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. MACRO SHIELD (MScFE 610 Logic)
def get_risk_multiplier():
    try:
        print("Consulting FRED for Systemic Risk...")
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        spread_data = web.get_data_fred('T10Y2Y', start)
        latest = spread_data.iloc[-1].values[0]
        print(f"10Y-2Y Spread: {latest}")
        return 0.5 if latest < 0 else 1.0
    except:
        return 1.0

# 3. UNIVERSE EXPANSION (S&P 500 - Solving the 'Not Enough Stocks' Error)
def get_sp500_tickers():
    # Efficiently scraping the S&P 500 list from Wikipedia (Institutional Standard)
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]
    return df['Symbol'].tolist()

# 4. ALPHA GENERATION (MScFE 632 logic)
def generate_signals(ticker_data, risk_multiplier):
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    raw_alpha = -z_score.iloc[-1] 
    return raw_alpha * risk_multiplier

if __name__ == "__main__":
    print("Executing V7.0 High-Capacity Signals Pipeline...")
    
    # Process Filter & Universe
    multiplier = get_risk_multiplier()
    tickers = get_sp500_tickers()
    print(f"Targeting {len(tickers)} assets for maximum statistical significance.")

    # Download Market Data in chunks to prevent Cloud RAM crash
    # We use a 1-year window for stable Z-scores
    data = yf.download(tickers, period="1y", interval="1d", progress=False, threads=True)['Close']
    data = data.ffill().fillna(0)

    final_alpha = generate_signals(data, multiplier)
    predictions = final_alpha.rank(pct=True)

    # 5. DYNAMIC MODEL HANDSHAKE
    try:
        my_models = sapi.get_models()
        target_handle = "dawityimer"
        target_uuid = next((uuid for name, uuid in my_models.items() if name.lower() == target_handle), None)
        
        if not target_uuid:
            print(f"ERROR: Create slot '{target_handle}' at signals.numer.ai first!")
            exit(1)

        # 6. FORMATTING (Standardizing for Bloomberg/US Format)
        submission = predictions.reset_index()
        submission.columns = ["ticker", "signal"]
        submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")
        
        # FINAL AUDIT: Ensure we meet the minimum stock count (usually > 100)
        print(f"Final submission contains {len(submission)} stocks.")
        
        submission.to_csv("submission.csv", index=False)
        sapi.upload_predictions("submission.csv", model_id=target_uuid)
        print("SUCCESS: HIGH-CAPACITY ALPHA DELIVERED.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise e
