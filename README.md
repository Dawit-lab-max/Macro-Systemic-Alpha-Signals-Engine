# Macro-Systemic Alpha Signals Engine
**By Dawit Yimer Gebreegziabehair, MScFE** 

## What it is
This is a fully automated signals pipeline for the Numerai Signals tournament. It analyzes the S&P 500 universe to find predictive signals while monitoring the U.S. Federal Reserve for systemic risk.

## How it works
*   **Macro Shield:** The system uses an API connection to FRED (Federal Reserve Economic Data) to monitor the 10Y-2Y yield spread. If the curve is inverted, it automatically reduces signal conviction to protect capital.
*   **Alpha Logic:** It uses a high-dimensional LightGBM architecture to identify 20-day mean reversion opportunities across the most liquid US equities.
*   **Infrastructure:** To handle the full 2,378-feature dataset on limited cloud RAM, I use Polars for out-of-core data streaming.
*   **Execution:** Hosted on GitHub Actions, the engine performs a complete data refresh, model inference, and signal delivery every 24 hours.

## Tools Used
*   **Python 3.11** 
*   **FRED API** (Systemic risk data)
*   **Yahoo Finance** (Equity price telemetry)
*   **Polars** (High-speed data orchestration)
*   **GitHub Actions** (Daily cloud execution)

## Live Performance
Daily signals and reputation scores are tracked on the Numerai platform:  
**[DAWITYIMER](https://signals.numer.ai/dawityimer)**

---
*MIT License | Institutional Quantitative Research*
