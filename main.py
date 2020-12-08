# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:24:51 2020

@author: chris
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from envs.Storage import Storage
from brain.agents import Agent, Template

# df = pd.read_csv("data/data_60m.csv")
# df = pd.read_csv("data/data.csv")
df = pd.read_csv("data/columbia/columbia_final.csv")

energy_scaler = StandardScaler(with_mean = False)
pv = energy_scaler.fit_transform(df.iloc[:, 2:3].values)
load = energy_scaler.transform(df.iloc[:, 1:2].values)

price_scaler = StandardScaler(with_mean = False)
price = price_scaler.fit_transform(df.iloc[:, 3:4].values)

data_scaled = np.concatenate([load, pv, price], axis = -1)

capacity = 150
p_max = 30
eff = 1.
min_SOC = 0. # TODO integrate this in Storage class
timestep = 60  # in minutes
action_size = 11
state_size = 5

peak = 1.7

battery = Storage(scaled_capacity = energy_scaler.transform(np.array([[capacity]]))[0], 
                  scaled_p_max = energy_scaler.transform([[p_max]])[0], 
                  eff = 1., 
                  min_SOC = 0., 
                  timestep = 60, 
                  action_size = action_size,
                  # peak = energy_scaler.transform(np.array([[peak]]))[0]
                  )

agent = Agent(battery_class = battery, memory_size = 10000, state_size = state_size)


template_battery = Storage(scaled_capacity = energy_scaler.transform(np.array([[capacity]]))[0], 
                  scaled_p_max = energy_scaler.transform([[p_max]])[0], eff = 1., 
                  min_SOC = 0., timestep = 60, action_size = 100000)

template_agent = Template(battery_class = template_battery, memory_size = 20000, state_size = state_size)


train_ratio = 0.99
train_size = round(len(data_scaled)*train_ratio)



for step in range(train_size):
    if step == 0:
        avg_price = np.zeros(train_size+1)
        day = 0
        SOC = np.zeros(train_size+1)
        SOC[step] = battery.ini_SOC
        battery_cmd = np.zeros(train_size)
        balance = np.zeros(train_size)
        done = False
        SOC_list = []
        action_list = []
        reward_list = []
        av_price_list = []
        balance_list = []
        
    # if step & (7*24)==0:
    #     done=True
    # else : 
    #     done = False
    if step>=24:
        avg_price[step] = np.mean(price[step-24:step])
        avg_price[step+1] = np.mean(price[step-24+1:step+1])
    else:
        avg_price[step] = np.mean(price[0:step+1])
        avg_price[step+1] = np.mean(price[1:step+2])
    state = np.concatenate([load[step], pv[step], np.array([SOC[step]]), price[step], np.array([avg_price[step]])])

    action = agent.random_action()
    reward, SOC[step+1],balance[step] = battery.perform_action(state = state, action = action, resolution = timestep)
    

    if step < train_size-1:
        next_state = np.concatenate([load[step+1], pv[step+1], np.array([SOC[step+1]]), price[step+1], np.array([avg_price[step+1]])])
        
        agent.save_exp(state, action, reward, next_state, done)


    else:
        break
       

# fig = plt.figure(figsize=(18,7))
# plt.step(range(0, 24 * 7), load[-24 * 7:], "bo-", label = "Load",where="post")
# plt.step(range(0, 24 * 7), pv[-24 * 7:], "ro-", label = "PV",where="post")
# plt.step(range(0, 24 * 7), SOC[-24 * 7-1:-1], "go-", label = "SOC",where="post")
# plt.step(range(0, 24 * 7), battery_cmd[-24 * 7:], "orange", label = "Storage Command",where="post")
# plt.legend()
# plt.show()

n_epoch = 5

for epoch in range(n_epoch):
    for step in range(train_size):
        if step == 0:
            avg_price = np.zeros(train_size+1)
            day = 0
            SOC = np.zeros(train_size+1)
            balance = np.zeros(train_size)
            SOC[step] = battery.ini_SOC
            done = False 
            SOC_list = []
            action_list = []
            reward_list = []
            av_price_list = []  
            balance_list = []
    
    
        # if step & (7*24)==0:
        #     done=True
        # else : 
        #     done = False
            
        if step>=24:
            avg_price[step] = np.mean(price[step-24:step])
            avg_price[step+1] = np.mean(price[step-24+1:step+1])
        else:
            avg_price[step] = np.mean(price[0:step+1])
            avg_price[step+1] = np.mean(price[1:step+2])   
            
        state = np.concatenate([load[step], pv[step], np.array([SOC[step]]), price[step], np.array([avg_price[step]])])
    
        action, explore_probability = agent.smart_action(state)
        reward, SOC[step+1], balance[step] = battery.perform_action(state = state, action = action, resolution = timestep) 
        
        SOC_list.append(SOC[step+1])
        reward_list.append(reward)
        action_list.append(action)
        av_price_list.append(avg_price[step])  
        if step % (24 *7 * (60 / timestep)) == 0:
            print("Epoch : ", epoch, " | Week : ", int(step/(24*7*(60/timestep))), " | Mean Reward : ", round(np.mean(reward_list[(int(step-24*7*(60/timestep))):step]),2), " | Eps : ", round(explore_probability,2))
            # try:
            #     print("Loss : ", agent.model_loss[-1])
            # except :
            #     pass
            
        if step < train_size-1:
            next_state = np.concatenate([load[step+1], pv[step+1], np.array([SOC[step+1]]), price[step+1], np.array([avg_price[step+1]])])
            
            agent.save_exp(state, action, reward, next_state, done)      
            agent.train()
                
        else:
            break

agent.model.save_weights("brain/saved_agents/DQN/columbia_v0.hdf5")

test_size = len(data_scaled)
agent.model.load_weights("brain/saved_agents/DQN/columbia_v0.hdf5")



for step in range(test_size):
    if step == 0:
        avg_price = np.zeros(test_size)
        day = 0
        SOC = np.zeros(test_size+1)
        balance = np.zeros(test_size)
        balance_template = np.zeros(test_size)
        SOC[step] = battery.ini_SOC
        done = False  
        SOC_list = []
        SOC_template_list = []
        action_list = []
        reward_list = []
        av_price_list = []
        balance_list = []
        balance_template_list = []
        action_template_list = []
        SOC_template = np.zeros(test_size+1)
        SOC_template[step] = battery.ini_SOC
    
    # if step & (7*24)==0:
    #     done=True
    # else : 
    #     done = False

    if step>=24:
        avg_price[step] = np.mean(price[step-24:step])
    else:
        avg_price[step] = np.mean(price[0:step+1])        
    
    state = np.concatenate([load[step], pv[step], np.array([SOC[step]]), price[step], np.array([avg_price[step]])])
    action = np.argmax(agent.model.predict(np.expand_dims(state, axis = 0)))

    state_template = np.concatenate([load[step], pv[step], np.array([SOC_template[step]]), price[step], np.array([avg_price[step]])]) 
    action_template = template_agent.smart_action(state_template)

    reward, SOC[step+1], balance[step] = battery.perform_action(state = state, action = action, resolution = timestep) 
    
    template_reward, SOC_template[step+1], balance_template[step] = template_battery.perform_action(state = state_template, action = action_template, resolution = timestep)

    SOC_template_list.append(SOC_template[step+1])

    SOC_list.append(SOC[step+1])
    reward_list.append(reward)
    action_list.append(action)
    action_template_list.append(template_battery.action_list[action_template]*template_battery.p_max)

    av_price_list.append(avg_price[step])  
    balance_list.append(np.array([balance[step]]))
    balance_template_list.append(np.array([balance_template[step]]))



price_real = price_scaler.inverse_transform(price)/100
load_real = energy_scaler.inverse_transform(load)
pv_real = energy_scaler.inverse_transform(pv)
p_max_real = energy_scaler.inverse_transform(battery.p_max)
balance_real = energy_scaler.inverse_transform(balance_list)
balance_template_real = energy_scaler.inverse_transform(balance_template_list)

cost_wo_ESS = (load_real - pv_real) * price_real
cost_template_ESS = balance_template_real * price_real
cost_ESS = balance_real * price_real

ESS_template_profit = cost_wo_ESS - cost_template_ESS
ESS_profit = cost_wo_ESS - cost_ESS

print("Cost without storage : ", np.sum(cost_wo_ESS[-24*7:]))
print("Cost with template storage : ",np.sum(cost_template_ESS[-24*7:]))
print("Cost with smart storage : ",np.sum(cost_ESS[-24*7:]))
print("Template Storage Profit : ",np.sum(ESS_template_profit[-24*7:]))
print("Smart Storage Profit : ",np.sum(ESS_profit[-24*7:]))


plt.style.use('seaborn-muted')
x = data_scaled
i = 150
j = 143
plt.figure(figsize = (25,5))
# plt.step(range(0, 24 * (i-j)), SOC_template_list[-24 * i:-24*j], "b-", label = "SOC template")
plt.step(range(0, 24 * (i-j)), x[-24 * i:-24*j, 0], label = "load")
plt.step(range(0, 24 * (i-j)), x[-24 * i:-24*j, 1], label = "pv")
plt.step(range(0, 24 * (i-j)), SOC_list[-24 * i:-24*j], label = "SOC")

plt.step(range(0, 24 * (i-j)), av_price_list[-24 * i:-24*j], label = "av_price")
plt.step(range(0, 24 * (i-j)), x[-24 * i:-24*j, 2], "b", label = "price")
# plt.step(range(0, 24 * (i-j)), reward_list[-24*i:-24*j], "r", label = "Reward")

# plt.step(range(0, 24 * (i-j)), balance_list[-24 * i:-24*j], "orange", label = "net_meter")
# plt.step(range(0, 24 * (i-j)), balance_template_list[-24 * i:-24*j], "blue", label = "net_meter template")

plt.bar(range(0, 24 * (i-j)), agent.action_list[action_list[-24 * i:-24*j]], facecolor = "w", edgecolor = "k", label = "action")
plt.ylabel("SOC/ Normalized Price")
plt.xlabel("Hour")
plt.legend()
plt.show()


plt.step(range(0, len(agent.model_loss)), agent.model_loss, "b", label = "price")
