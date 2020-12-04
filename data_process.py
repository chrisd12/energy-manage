# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:30:06 2020

@author: chris
"""

import pandas as pd

import plotly.graph_objects as go
from plotly.offline import plot
import datetime


df_raw = pd.read_csv("data/household_data.csv")

df = pd.DataFrame()
df["import"] = df_raw['DE_KN_industrial1_grid_import']
df["export"] = df_raw['DE_KN_public2_grid_import']
df["pv"] = df_raw['DE_KN_industrial1_pv_1']
df.index = pd.to_datetime(df_raw["cet_cest_timestamp"],yearfirst=True,utc=True)
df.index = df.index.tz_localize(None)


fig = go.Figure()
fig.add_trace(go.Scatter(x=df["import"].index, y=df["import"].values, mode='lines', name='import'))
fig.add_trace(go.Scatter(x=df["export"].index, y=df["export"].values, mode='lines', name='export'))
fig.add_trace(go.Scatter(x=df["pv"].index, y=df["pv"].values, mode='lines', name='pv'))

plot(fig)


ts = pd.DataFrame()
ts["import"] = df["import"].diff().iloc[43000:80000]
ts["export"] = df["export"].diff().iloc[43000:80000]
ts["pv"] = df["pv"].diff().iloc[43000:80000]
ts.index = df.index[43000:80000]
ts["load"] = ts["import"]
# for i in range(len(ts)):
    # ts["load"][i] = max(ts["import"][i] - (ts["export"][i] - ts["pv"][i]), 0.)

df_price = pd.read_csv("data/DE_AT_LU_price.csv", sep=";")
df_price.index = pd.to_datetime(df_price["Datetime"], dayfirst=True)
df_price.drop(columns = ["Datetime"], inplace = True)

ts_sync = ts["2016-03-01 00:00":"2017-03-01 00:00"]
df_price_sync = df_price["2016-03-01 00:00":"2017-03-01 00:00"]



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


fig = go.Figure()
fig.add_trace(go.Scatter(x=data_60m["load"].index, y=data_60m["load"].values, mode='lines', name='load'))
fig.add_trace(go.Scatter(x=data_60m["pv"].index, y=data_60m["pv"].values, mode='lines', name='pv'))
# fig.add_trace(go.Scatter(x=df["pv"].index, y=df["pv"].values, mode='lines', name='pv'))

plot(fig)

# data_15m.to_csv("data/data_15m.csv")
# data_60m.to_csv("data/data_60m.csv")


df_columbia_raw = pd.read_csv("data/columbia/Residential_10.csv", parse_dates=[["date","hour"]])
df_columbia = pd.DataFrame()
df_columbia["H10"] = df_columbia_raw["energy_kWh"]
df_columbia.index = df_columbia_raw["date_hour"]


node_list = [10,11,12,13,14,15,16]
fig = go.Figure()
for i in node_list:
    df_h = pd.DataFrame()
    df_h = pd.read_csv("data/columbia/Residential_"+str(i)+".csv", parse_dates=[["date","hour"]])
    df_h.index = df_h["date_hour"]
    df_columbia["H"+str(i)] = df_h["energy_kWh"]
    
    fig.add_trace(go.Scatter(x=df_h.index, y=df_h["energy_kWh"].values, mode='lines', name=str(i)))

plot(fig)
