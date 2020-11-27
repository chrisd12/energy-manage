# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:30:06 2020

@author: chris
"""


import pandas as pd

df_raw = pd.read_csv("data/household_data.csv")

df = pd.DataFrame()
df["import"] = df_raw['DE_KN_residential3_grid_import']
df["export"] = df_raw['DE_KN_residential3_grid_export']
df["pv"] = df_raw['DE_KN_residential3_pv']
df.index = pd.to_datetime(df_raw["cet_cest_timestamp"],yearfirst=True,utc=True)
df.index = df.index.tz_localize(None)

ts = pd.DataFrame()
ts["import"] = df["import"].diff().iloc[43000:80000]
ts["export"] = df["export"].diff().iloc[43000:80000]
ts["pv"] = df["pv"].diff().iloc[43000:80000]
ts.index = df.index[43000:80000]
ts["load"] = ts["import"]
for i in range(len(ts)):
    ts["load"][i] = max(ts["import"][i] - (ts["export"][i] - ts["pv"][i]), 0.)

df_price = pd.read_csv("data/DE_AT_LU_price.csv", sep=";")
df_price.index = pd.to_datetime(df_price["Datetime"], dayfirst=True)
df_price.drop(columns = ["Datetime"], inplace = True)


ts_sync = ts["2016-03-04 00:00":"2017-03-24 00:00"]
df_price_sync = df_price["2016-03-04 00:00":"2017-03-24 00:00"]


data_15m = pd.DataFrame()
data_15m["load"] = ts_sync["load"].round(3)
data_15m["pv"] = ts_sync["pv"].round(3)
data_15m["price"] = (df_price_sync.resample("15T").mean().interpolate(method="linear")/1000).round(5)
data_15m.index.rename("Datetime", inplace=True)

data_60m = pd.DataFrame()
data_60m["load"] = ts_sync["load"].resample("60T").sum().round(3)
data_60m["pv"] = ts_sync["pv"].resample("60T").sum().round(3)
data_60m["price"] = (df_price_sync.resample("60T").mean()/1000).round(5)
data_60m.index.rename("Datetime", inplace=True)

data_15m.to_csv("data/data_15m.csv")
data_60m.to_csv("data/data_60m.csv")



