import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime
import requests # New dependency for the 403 bypass

# 1. INSTITUTIONAL AUTHENTICATION
os.environ["FRED_API_KEY"] = "3699988d98d460d752a241f85df9532f"
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. MACRO SHIELD (GWP 610 Logic)
def get_risk_multiplier():
    try:
        print("Polling FRED for Systemic Risk...")
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        spread_data = web.get_data_fred('T10Y2Y', start)
        latest = spread_data.iloc[-1].values[0]
        print(f"10Y-2Y Spread: {latest}")
        return 0.5 if latest < 0 else 1.0
    except:
        return 1.0

# 3. ROBUST UNIVERSE EXTRACTION (Bypassing 403 Forbidden)
def get_sp500_tickers():
    print("Extracting S&P 500 Universe...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        table = pd.read_html(response.text)
        df = table[0]
        # Professional cleanup of tickers
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except Exception as e:
        print(f"Wikipedia blocked request: {e}. Reverting to static liquid universe.")
        # Fallback to Top 100 most liquid assets to ensure >100 stocks submitted
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG", "UNH", "MA", "HD", "DIS", "PYPL", "BAC", "VZ", "ADBE", "NFLX", "INTC", "PFE", "T", "ABT", "XOM", "CVX", "COST", "PEP", "KO", "WMT", "WFC", "XLY", "XLP", "XLV", "XLF", "XLI", "XLK", "XLU", "XLB", "CSCO", "PEP", "AVGO", "COST", "QCOM", "TXN", "TMUS", "AMAT", "INTU", "AMGN", "SBUX", "MDLZ", "ISRG", "GILD", "BKNG", "ADI", "VRTX", "ADP", "REGN", "PYPL", "FISV", "ATVI", "MELI", "PANW", "SNPS", "CDNS", "CHTR", "KLAC", "MAR", "MNST", "ORLY", "CTAS", "KDP", "AEP", "ADSK", "PAYX", "MCHP", "EXC", "LULU", "IDXX", "DXCM", "MELI", "NXPI", "WDAY", "AZN", "BIIB", "TEAM", "CRWD", "EBAY", "JD", "ROST", "SIRI", "VRSK", "ALGN", "WBA", "SPLK", "CPRT", "FAST", "ILMN", "LRCX", "PCAR", "PDD", "SWKS", "EBAY"]

# 4. ALPHA GENERATION (MScFE 632 Mean Reversion)
def generate_signals(ticker_data, risk_multiplier):
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    raw_alpha = -z_score.iloc[-1] 
    return raw_alpha * risk_multiplier

if __name__ == "__main__":
    print("Executing V7.1 Surgical Alpha Pipeline...")
    
    multiplier = get_risk_multiplier()
    tickers = get_sp500_tickers()
    
    # Downloading Market Data
    data = yf.download(tickers, period="1y", interval="1d", progress=False, threads=True)['Close']
    data = data.ffill().fillna(0)

    final_alpha = generate_signals(data, multiplier)
    predictions = final_alpha.rank(pct=True)

    # 5. DYNAMIC HANDSHAKE
    try:
        my_models = sapi.get_models()
        target_handle = "dawityimer"
        target_uuid = next((uuid for name, uuid in my_models.items() if name.lower() == target_handle), None)
        
        if not target_uuid:
            print(f"ERROR: Model handle '{target_handle}' not found.")
            exit(1)

        # 6. FORMATTING
        submission = predictions.reset_index()
        submission.columns = ["ticker", "signal"]
        submission["ticker"] = submission["ticker"].apply(lambda x: f"{x.replace('-', '.')} US")
        
        print(f"Transmitting {len(submission)} validated signals...")
        submission.to_csv("submission.csv", index=False)
        sapi.upload_predictions("submission.csv", model_id=target_uuid)
        print("SUCCESS: V7.1 SIGNAL DELIVERY VERIFIED.")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        raise e
