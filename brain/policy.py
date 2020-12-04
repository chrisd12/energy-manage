# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:45:15 2020

@author: chris
"""
import numpy as np

def epsilon_decay(model, explore_start, explore_stop, decay_rate, decay_step, action_size,state):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
      action = np.random.randint(0, action_size)
    else:
        q_values=model.predict(np.expand_dims(state, axis = 0))
        action = np.argmax(q_values)
    return action, explore_probability
    
