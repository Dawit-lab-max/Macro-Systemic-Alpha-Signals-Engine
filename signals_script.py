import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from numerapi import SignalsAPI
import datetime

# 1. AUTHENTICATION (FRED & NUMERAI)
os.environ["FRED_API_KEY"] = "3699988d98d460d752a241f85df9532f"
PUBLIC_KEY = os.getenv('NUMERAI_PUBLIC_KEY').strip()
SECRET_KEY = os.getenv('NUMERAI_SECRET_KEY').strip()
sapi = SignalsAPI(PUBLIC_KEY, SECRET_KEY)

# 2. MACRO SHIELD (GWP 610 Logic)
def get_risk_multiplier():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=90)
        spread = web.get_data_fred('T10Y2Y', start)
        return 0.5 if spread.iloc[-1].values[0] < 0 else 1.0
    except:
        return 1.0

# 3. UNIVERSE & ALPHA
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG"]
multiplier = get_risk_multiplier()
data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
data = data.ffill().fillna(0)

# Mean Reversion Logic (MScFE 632)
z_score = (data - data.rolling(20).mean()) / data.rolling(20).std()
final_alpha = -z_score.iloc[-1] * multiplier
predictions = final_alpha.rank(pct=True)

# 4. DYNAMIC HANDSHAKE (UUID Fix)
try:
    my_models = sapi.get_models()
    target_uuid = None
    for name, uuid in my_models.items():
        if name.lower() == "dawityimer":
            target_uuid = uuid
            break
    
    if not target_uuid:
        print("CRITICAL: Create model handle 'dawityimer' at numer.ai/signals first!")
        exit(1)

    # 5. DELIVERY
    submission = predictions.reset_index()
    submission.columns = ["ticker", "signal"]
    submission["ticker"] = submission["ticker"].apply(lambda x: f"{x} US")
    submission.to_csv("signals_upload.csv", index=False)
    sapi.upload_predictions("signals_upload.csv", model_id=target_uuid)
    print("SIGNALS V4.7 SUCCESSFUL.")
except Exception as e:
    print(f"Error: {e}")
    raise e
