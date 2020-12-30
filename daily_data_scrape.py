"""
# Created by ashish1610dhiman at 30/12/20
Contact at ashish1610dhiman@gmail.com
"""

import yfinance as yf
import pandas as pd
import traceback

# Get component tickers
component_df = pd.read_csv("./input/hang_seng_components_2015.csv")
component_tickers = list(component_df["symbol"])

ticker_dict = {"ad_index": "HSI"}
for i in range(1, len(component_tickers)):
    ticker_dict[f"security_{i}"] = component_tickers[i]

# Scrape historical data for the tickers
final_data_weekly = pd.DataFrame()
final_data_daily = pd.DataFrame()

for ad_name, symbol in ticker_dict.items():
    tick = yf.Ticker(symbol)
    try:
        hist_daily = tick.history(period="max",interval="1d")
        hist_daily=hist_daily[["Close"]]
        hist_daily.columns=[ad_name]
    except:
        traceback.TracebackException()
    if symbol=="HSI":
        final_data_daily=hist_daily
    else:
        final_data_daily=final_data_daily.merge(hist_daily,"outer",left_index=True,right_index=True)

