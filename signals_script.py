import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime
import gc

# 1. INSTITUTIONAL CREDENTIALS
# These pull from your GitHub Secrets vault
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. THE "SHIELD": FRED MACRO REGIME FILTER
def get_systemic_risk_filter():
    """
    Uses the Treasury Yield Spread (10Y minus 2Y) as a proxy for systemic risk.
    An inverted curve signals a recessionary regime.
    """
    try:
        print("Polling St. Louis Fed (FRED) for Systemic Risk levels...")
        start = datetime.datetime.now() - datetime.timedelta(days=60)
        # T10Y2Y is the primary recession indicator used by firms like Pharo
        spread = web.DataReader('T10Y2Y', 'fred', start)
        latest_spread = spread.iloc[-1].values[0]
        
        # If spread is negative, reduce signal conviction to preserve capital
        return 0.5 if latest_spread < 0 else 1.0
    except Exception as e:
        print(f"FRED Connection Error: {e}. Defaulting to Neutral Risk.")
        return 1.0

# 3. THE "SWORD": MULTI-FACTOR EQUITY ALPHA
def generate_alpha_signals(ticker_data, risk_multiplier):
    """
    Calculates a Mean-Reversion Alpha normalized by volatility.
    Rooted in MScFE 610 Stochastic Modeling.
    """
    # Calculate 20-day Mean Reversion (Z-Score)
    m_avg = ticker_data.rolling(window=20).mean()
    m_std = ticker_data.rolling(window=20).std()
    z_score = (ticker_data - m_avg) / m_std
    
    # Identify the 'Glitch' (The Alpha)
    # We invert the Z-score: if it's too high, we sell; if too low, we buy.
    alpha = -z_score.iloc[-1] 
    
    # Apply the Macro Shield
    return alpha * risk_multiplier

# 4. UNIVERSE DEFINITION
# We use the 'Liquid 50' - High-volume US Equities
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V", 
    "UNH", "MA", "PG", "HD", "DIS", "PYPL", "BAC", "VZ", "ADBE", "CMCSA",
    "NFLX", "INTC", "PFE", "T", "ABT", "XOM", "CVX", "COST", "PEP", "KO",
    "WMT", "WFC", "XLY", "XLP", "XLV", "XLF", "XLI", "XLK", "XLU", "XLB"
]

# 5. EXECUTION PIPELINE
if __name__ == "__main__":
    print(f"Commencing Daily Run for: Macro-Systemic-Alpha-Signals-Engine")
    
    # Pull Macro Filter
    risk_shield = get_systemic_risk_filter()
    print(f"Systemic Risk Multiplier: {risk_shield}")

    # Pull Equity Data
    # Map for Yahoo Finance
    yf_map = [t.replace("BRK-B", "BRK-B") for t in tickers]
    data = yf.download(yf_map, period="1y", interval="1d", progress=False)['Close']
    
    # Handle any missing data (MScFE Standard)
    data = data.ffill().fillna(0)

    # Extract Alpha
    raw_alpha = generate_alpha_signals(data, risk_shield)
    
    # 6. NORMALIZATION (Ranking for Numerai)
    # Convert raw math into the 0.0 to 1.0 rank the fund expects
    final_ranks = raw_alpha.rank(pct=True)

    # 7. FORMATTING & DELIVERY
    submission = final_ranks.reset_index()
    submission.columns = ["ticker", "signal"]
    
    # Ensure Ticker format is 'SYMBOL US' for US stocks
    submission["ticker"] = submission["ticker"].apply(lambda x: f"{x.replace('BRK-B', 'BRK/B')} US")

    submission.to_csv("signals_upload.csv", index=False)
    
    try:
        sapi.upload_predictions("signals_upload.csv", model_id="DAWITYIMER")
        print("TRANSMISSION SUCCESSFUL: ALPHA CAPTURED.")
    except Exception as e:
        print(f"Transmission Failed: {e}")
