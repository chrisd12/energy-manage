# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:23:50 2020

@author: chris
"""
import numpy as np

class Storage:
    def __init__(self, scaled_capacity, scaled_p_max, eff, min_SOC, timestep, action_size, peak = None):
        self.capacity = scaled_capacity
        self.p_max = scaled_p_max
        self.eff = eff
        self.ini_SOC = 0.
        self.min_SOC = min_SOC
        self.action_size = action_size
        self.action_list = np.linspace(-1, 1, num = action_size, endpoint = True)
        self.peak = peak

    def penalties_check(self, SOC, next_SOC, load, pv):
        penalty = 0.
        balance = load - pv + (next_SOC - SOC) * self.capacity
        
        if next_SOC > 1.:
            next_SOC = 1.
            penalty += 10.

        if next_SOC < 0.:
            next_SOC = 0.
            penalty += 10.
            
        # if pv - load - (next_SOC - SOC) * self.capacity > 0 :
        #     penalty += 10    
            
        # if self.peak is not None:
        #     if abs(balance) > self.peak:
        #         penalty += (balance - self.peak) * 10

        return penalty, next_SOC

    def perform_action(self, state, action, resolution):
        load = state[0]
        pv = state[1]
        SOC = state[2]
        price = state[3]
        avg_price = state[4]
        
        delta_SOC = self.action_list[action] * self.p_max * (resolution / 60) / self.capacity # relative SOC in %
        next_SOC = SOC + delta_SOC
        penalty, next_SOC = self.penalties_check(SOC, next_SOC, load, pv)
        delta_SOC = next_SOC - SOC

        balance = load - pv + (delta_SOC * self.capacity)
        
        if balance < 0:
            current_gain = (price - avg_price) * abs(balance)
        else:
            current_gain = (avg_price - price) * balance
        # current_gain = (avg_price - price) * balance

        reward = current_gain - penalty
        return reward, next_SOC, balance

