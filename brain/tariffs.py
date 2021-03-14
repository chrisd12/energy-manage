# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 21:12:50 2020

@author: chris
"""

import numpy as np
from brain.utils import Memory
from brain.policy import epsilon_decay
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd


class RetailPrice:
    def __init__(self, mode=0, constant = 20, index = None):
        self.mode = mode
        self.constant = constant
        self.datetime = index
        if self.mode == 1 :
            self.load_predefined()

    def fixed_rate(self):
        return self.constant
    
    def load_predefined(self):
        self.retail_ts = pd.read_csv("./data/tariffs/TimeofUse/ToU1.csv",delimiter=";")
        self.retail_ts.index = self.retail_ts["Time"]
        self.retail_ts.drop(columns=["Time"], inplace = True)
        
    def predefined(self, datetime):
        return self.retail_ts.loc[datetime]
    
    
    def step(self,datetime):
        if self.mode == 0:
            self.price = self.fixed_rate()
    
        if self.mode == 1:
            self.price = self.predefined(datetime)
            
        return self.price
        
    def generate_array(self):
        self.retail_df = pd.DataFrame(index = self.datetime, columns = ["Price"])
        for i in self.datetime:
            self.retail_df["Price"].loc[i] = self.step(i)
        
        return self.retail_df
    

class TimeofUse:
    def __init__(self, mode=0, index = None):
        self.max_import = 0.
        self.max_export = 0.
        self.datetime = index
        self.mode = mode # 0: predefined ToU
        if self.mode == 0:
            self.load_predefined()
            
    def load_predefined(self):
        self.ToU = pd.read_csv("./data/tariffs/TimeofUse/ToU1.csv",delimiter=";")
        self.ToU["Time"]=pd.to_datetime(self.ToU["Time"])
        self.ToU.index = self.ToU["Time"].dt.time
        self.ToU.drop(columns=["Time"], inplace = True)


    def step(self,time):
        if self.mode == 0:
            self.grid_fee = self.ToU["GridFee [€/kWh]"].loc[time]
            
        return self.grid_fee

    def generate_array(self):
        self.grid_fee_df = pd.DataFrame(index = self.datetime, columns = ["Price"])
        time = self.datetime.dt.time
        for i in self.datetime:
            self.grid_fee_df["Price"].loc[i] = self.step(time.loc[str(i)])
        
        return self.grid_fee_df


