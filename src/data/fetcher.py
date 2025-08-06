# src/data/fetcher.py
# Downloads SPY daily OHLC data from Yahoo Finance
# Provides clean data for volatility modeling and simulation
# RELEVANT FILES: preprocessor.py, main.py

import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_spy_data(end_date='2024-12-31', years_back=10):
    start_date = pd.to_datetime(end_date) - pd.DateOffset(years=years_back)
    
    spy = yf.Ticker('SPY')
    data = spy.history(start=start_date, end=end_date)
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()
    
    return data

def get_returns(prices):
    return prices['Close'].pct_change().dropna()

def get_log_returns(prices):
    import numpy as np
    return (prices['Close'] / prices['Close'].shift(1)).apply(np.log).dropna()