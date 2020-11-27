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


class Agent:
    def __init__(self, battery_class, memory_size, state_size):
        self.action_size = battery_class.action_size
        self.action_list = np.linspace(-1, 1, num = self.action_size, endpoint = True)
        self.bat_p_max = battery_class.p_max
        self.state_size = state_size
        self.explore_start = 1.
        self.explore_stop = 0.01
        self.decay_rate = 0.0001
        self.decay_step = 0
        self.gamma = 0.95
        self.memory = Memory(memory_size)
        self.batch_size = 64
        self.model = self.neural_network(state_size)
        self.target_model = self.neural_network(state_size)
                
    def random_action(self):
        action = np.random.randint(0 , len(self.action_list))
        return action
    
    def save_exp(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        self.memory.store(experience)
        
    
    def neural_network(self, state_size):
        model = Sequential()
        model.add(Dense(40, input_dim=state_size, activation="elu"))
        model.add(Dense(160, activation="elu"))
        model.add(Dense(40, activation="linear"))
        model.add(Dense(len(self.action_list)))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
        model.summary()
        return model
    
    
    def smart_action(self, state):
        action, explore_probability = epsilon_decay(self.model, self.explore_start, self.explore_stop, self.decay_rate, self.decay_step, len(self.action_list), state)  
        return action, explore_probability
    
    def memorize(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    
    def train(self):
        tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)
       
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
            
        target = self.model.predict(state)
        target_old = np.array(target)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])


        # if self.decay_step % 120 == 0:
          # self.target_train()

        indices = np.arange(self.batch_size, dtype=np.int32)
        absolute_errors = np.abs(target_old[indices, np.array(action)]-target[indices, np.array(action)])
        self.memory.batch_update(tree_idx, absolute_errors)

        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        self.decay_step += 1


    def target_train(self):
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-0.1) + q_weight *0.1
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)
            
            
class Template:
    def __init__(self, battery_class, memory_size, state_size):
        self.action_size = battery_class.action_size
        self.action_list = np.linspace(-1, 1, num = self.action_size, endpoint = True)
        self.bat_p_max = battery_class.p_max
        self.bat_cap = battery_class.capacity
        
    def smart_action(self, state):
        load = state[0]
        pv = state[1]
        SOC = state[2]
        
        balance = pv - load
        
        if balance < 0:
            balance = max(balance, -SOC*self.bat_cap)
            action_ratio = max(balance / self.bat_p_max, -1.)
            
        else:
            balance = min(balance, (1-SOC)*self.bat_cap)
            action_ratio = min(balance / self.bat_p_max, 1.)

        action = np.abs(self.action_list - action_ratio).argmin()

        return action