"""
# Created by ashish1610dhiman at 30/12/20
Contact at ashish1610dhiman@gmail.com
"""


import pandas as pd
import numpy as np

stock_data=pd.read_csv("./input/all_stocks_5yr.csv")
stocks=stock_data["Name"].unique()
stock_data.rename(columns={"date": "Date"},inplace=True)

index_data=pd.read_csv("./input/^GSPC.csv")

## Aggregate data
final_data=pd.DataFrame()
final_data=index_data[["Date","Close"]].rename(columns={"Close": "index"})


for counter,stock in enumerate(stocks,1):
    name = f"security_{counter}"
    subset = stock_data.loc[stock_data["Name"]==stock][["Date","close"]].rename(columns={"close": name})
    final_data=final_data.merge(subset,on="Date",how="outer")

final_data.to_csv("./input/S&P500_daily_step0.csv",index=False)

#Get null stats
null_stats=final_data.isnull().mean()
null_stats.sort_values(inplace=True,ascending=False)

high_null_stocks=null_stats.loc[null_stats>0.05].rename(index=str).index


final_data=final_data.drop(labels=high_null_stocks,axis=1)
final_data.dropna(axis=0,inplace=True)

final_data.columns=[["Date","index"]+[f"security_{counter}" for counter in range(1,final_data.shape[1]-1)]]

final_data[["index"]].plot()

#Gen daily data as prev format
final_data.drop(columns=["Date"]).to_csv("./input/index-weekly-data/index_daily9.csv",index=False)


#Gen weekly data as prev format
final_data.drop(columns=["Date"]).to_csv("./input/index-weekly-data/index_daily9.csv",index=False)

final_weekly = final_data.drop(columns=["Date"]).iloc[1::5]
final_weekly[["index"]].plot()
final_weekly.to_csv("./input/index-weekly-data/index_9.csv",index=False)