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
from brain.tariffs import RetailPrice, TimeofUse

# df = pd.read_csv("data/data_60m.csv")
# df = pd.read_csv("data/data.csv")
df = pd.read_csv("data/columbia/columbia_final.csv")
df["date_hour"] = pd.to_datetime(df["date_hour"], format="%Y-%m-%d %H:%M:%S")
df.index = df["date_hour"]
df = df.loc[:"2017-09-01 00:00:00"]
energy_scaler = StandardScaler(with_mean = False)
pv = energy_scaler.fit_transform(df.iloc[:, 2:3].values)
load = energy_scaler.transform(df.iloc[:, 1:2].values)

smart_price = False

if smart_price:
    retail = RetailPrice(mode=0,constant=20, index = df.index)
    price_retail = retail.generate_array()
    ToU = TimeofUse(index=df["date_hour"]) 
    grid_fee = ToU.generate_array()
    df["smart_price"] = price_retail + grid_fee


price_scaler = StandardScaler(with_mean = False)

buy_price = price_scaler.fit_transform(df.iloc[:, -1:].values)
sell_price = price_scaler.fit_transform(df.iloc[:, -1:].values)

data_scaled = np.concatenate([load, pv, buy_price], axis = -1)

capacity = 100
p_max = 30
eff = 1.
min_SOC = 0. # TODO integrate this in Storage class
timestep = 60  # in minutes
action_size = 11
state_size = 7

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


train_ratio = 0.8
train_size = round(len(data_scaled)*train_ratio)


    
for step in range(train_size):
    if step == 0:
        avg_buy_price = np.zeros(train_size+1)
        avg_sell_price = np.zeros(train_size+1)

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
        avg_buy_price[step] = np.mean(buy_price[step-24:step])
        avg_buy_price[step+1] = np.mean(buy_price[step-24+1:step+1])
        avg_sell_price[step] = np.mean(sell_price[step-24:step])
        avg_sell_price[step+1] = np.mean(sell_price[step-24+1:step+1])
    else:
        avg_buy_price[step] = np.mean(buy_price[0:step+1])
        avg_buy_price[step+1] = np.mean(buy_price[1:step+2])
        avg_sell_price[step] = np.mean(sell_price[0:step+1])
        avg_sell_price[step+1] = np.mean(sell_price[1:step+2])
    state = np.concatenate([load[step], pv[step], np.array([SOC[step]]), buy_price[step], np.array([avg_buy_price[step]]), 
                            sell_price[step], np.array([avg_sell_price[step]])])

    action = agent.random_action()
    reward, SOC[step+1],balance[step] = battery.perform_action(state = state, action = action, resolution = timestep)
    

    if step < train_size-1:
        next_state = np.concatenate([load[step+1], pv[step+1], np.array([SOC[step+1]]), buy_price[step+1], np.array([avg_buy_price[step+1]]),
                                      sell_price[step+1], np.array([avg_sell_price[step+1]])])
        
        agent.save_exp(state, action, reward, next_state, done)
        agent.target_train()

    else:
        break
       

# fig = plt.figure(figsize=(18,7))
# plt.step(range(0, 24 * 7), load[-24 * 7:], "bo-", label = "Load",where="post")
# plt.step(range(0, 24 * 7), pv[-24 * 7:], "ro-", label = "PV",where="post")
# plt.step(range(0, 24 * 7), SOC[-24 * 7-1:-1], "go-", label = "SOC",where="post")
# plt.step(range(0, 24 * 7), battery_cmd[-24 * 7:], "orange", label = "Storage Command",where="post")
# plt.legend()
# plt.show()

n_epoch = 6

for epoch in range(n_epoch):
    for step in range(train_size):
        if step == 0:
            avg_buy_price = np.zeros(train_size+1)
            avg_sell_price = np.zeros(train_size+1)

            day = 0
            SOC = np.zeros(train_size+1)
            balance = np.zeros(train_size)
            SOC[step] = battery.ini_SOC
            done = False 
            SOC_list = []
            action_list = []
            reward_list = []
            av_buy_price_list = [] 
            av_sell_price_list = []  
            balance_list = []
    
    
        if step & (7*24)==0:
            agent.train()
        else : 
            pass
            
        if step>=24:
            avg_buy_price[step] = np.mean(buy_price[step-24:step])
            avg_buy_price[step+1] = np.mean(buy_price[step-24+1:step+1])
            avg_sell_price[step] = np.mean(sell_price[step-24:step])
            avg_sell_price[step+1] = np.mean(sell_price[step-24+1:step+1])
        else:
            avg_buy_price[step] = np.mean(buy_price[0:step+1])
            avg_buy_price[step+1] = np.mean(buy_price[1:step+2])   
            avg_sell_price[step] = np.mean(sell_price[0:step+1])
            avg_sell_price[step+1] = np.mean(sell_price[1:step+2]) 
            
        state = np.concatenate([load[step], pv[step], np.array([SOC[step]]), buy_price[step], np.array([avg_buy_price[step]]),
                                 sell_price[step], np.array([avg_sell_price[step]])])
    
        action, explore_probability = agent.smart_action(state)
        reward, SOC[step+1], balance[step] = battery.perform_action(state = state, action = action, resolution = timestep) 
        
        SOC_list.append(SOC[step+1])
        reward_list.append(reward)
        action_list.append(action)
        av_buy_price_list.append(avg_buy_price[step])
        av_sell_price_list.append(avg_sell_price[step])  
        
        if step % (24 *7 * (60 / timestep)) == 0:
            print("Epoch : ", epoch, " | Week : ", int(step/(24*7*(60/timestep))), " | Mean Reward : ", round(np.mean(reward_list[(int(step-24*7*(60/timestep))):step]),2), " | Eps : ", round(explore_probability,2))
            # try:
            #     print("Loss : ", agent.model_loss[-1])
            # except :
            #     pass
            
        if step < train_size-1:
            next_state = np.concatenate([load[step+1], pv[step+1], np.array([SOC[step+1]]), buy_price[step+1], np.array([avg_buy_price[step+1]]),
                                        sell_price[step+1], np.array([avg_sell_price[step+1]])])
            
            agent.save_exp(state, action, reward, next_state, done)      
            agent.train()
                
        else:
            break

agent.model.save_weights("brain/saved_agents/DQN/columbia_v4.hdf5")



test_size = len(data_scaled)
agent.model.load_weights("brain/saved_agents/DQN/columbia_v4.hdf5")



for step in range(test_size):
    if step == 0:
        avg_buy_price = np.zeros(test_size)
        avg_sell_price = np.zeros(test_size)
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
        av_buy_price_list = []
        av_sell_price_list = []
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
        avg_buy_price[step] = np.mean(buy_price[step-24:step])
        avg_sell_price[step] = np.mean(sell_price[step-24:step])
    else:
        avg_buy_price[step] = np.mean(buy_price[0:step+1])       
        avg_sell_price[step] = np.mean(sell_price[0:step+1])  
    
    state = np.concatenate([load[step], pv[step], np.array([SOC[step]]), buy_price[step], np.array([avg_buy_price[step]]),
                            sell_price[step], np.array([avg_sell_price[step]])])
    action = np.argmax(agent.model.predict(np.expand_dims(state, axis = 0)))

     
    state_template = np.concatenate([load[step], pv[step], np.array([SOC_template[step]]), buy_price[step], np.array([avg_buy_price[step]]),
                            sell_price[step], np.array([avg_sell_price[step]])])
    action_template = template_agent.smart_action(state_template)

    reward, SOC[step+1], balance[step] = battery.perform_action(state = state, action = action, resolution = timestep) 
    
    template_reward, SOC_template[step+1], balance_template[step] = template_battery.perform_action(state = state_template, action = action_template, resolution = timestep)

    SOC_template_list.append(SOC_template[step+1])

    SOC_list.append(SOC[step+1])
    reward_list.append(reward)
    action_list.append(action)
    action_template_list.append(template_battery.action_list[action_template]*template_battery.p_max)

    av_buy_price_list.append(avg_buy_price[step])  
    av_sell_price_list.append(avg_sell_price[step]) 
    balance_list.append(np.array([balance[step]]))
    balance_template_list.append(np.array([balance_template[step]]))



buy_price_real = price_scaler.inverse_transform(buy_price)/100
sell_price_real = price_scaler.inverse_transform(sell_price)/100
avg_buy_price_real = price_scaler.inverse_transform(av_buy_price_list)/100
load_real = energy_scaler.inverse_transform(load)
pv_real = energy_scaler.inverse_transform(pv)
p_max_real = energy_scaler.inverse_transform(battery.p_max)
balance_real = energy_scaler.inverse_transform(balance_list)
balance_template_real = energy_scaler.inverse_transform(balance_template_list)


balance_wo_ES_real = load_real - pv_real


cost_wo_ESS = np.clip(balance_wo_ES_real,0,None)*buy_price_real + np.clip(balance_wo_ES_real,None,0)*sell_price_real


cost_template_ESS = np.clip(balance_template_real,0,None)*buy_price_real + np.clip(balance_template_real,None,0)*sell_price_real
cost_ESS = np.clip(balance_real,0,None)*buy_price_real + np.clip(balance_real,None,0)*sell_price_real

ESS_template_profit = cost_wo_ESS - cost_template_ESS
ESS_profit = cost_wo_ESS - cost_ESS

print("Cost without storage : ", np.sum(cost_wo_ESS[-24*14:]))
print("Cost with template storage : ",np.sum(cost_template_ESS[-24*14:]))
print("Cost with smart storage : ",np.sum(cost_ESS[-24*14:]))
print("Template Storage Profit : ",np.sum(ESS_template_profit[-24*14:]))
print("Smart Storage Profit : ",np.sum(ESS_profit[-24*14:]))
print("Max import template : ",np.max(balance_template_real[-24*14:]))
print("Max import smart : ",np.max(balance_real[-24*14:]))
print("Max export template : ",np.min(balance_template_real[-24*14:]))
print("Max export smart : ",np.min(balance_real[-24*14:]))


plt.style.use('seaborn-muted')
x = data_scaled
i = 7
j = 1

plt.figure(figsize = (20,5))
# plt.step(range(0, 24 * (i-j)), SOC_template_list[-24 * i:-24*j], "b-", label = "SOC template")
# plt.step(df["date_hour"].values[-24 * i:-24*j], load_real[-24 * i:-24*j, 0], label = "load", color='blue')
# plt.step(df["date_hour"].values[-24 * i:-24*j], pv_real[-24 * i:-24*j, 0], label = "pv", color='orange')


plt.step(df["date_hour"].values[-24 * i:-24*j], avg_buy_price_real[-24 * i:-24*j]*100, label = "av_price", color='grey')
plt.step(df["date_hour"].values[-24 * i:-24*j], buy_price_real[-24 * i:-24*j, 0]*100, "b", label = "price", color='black')
plt.step(df["date_hour"].values[-24 * i:-24*j], buy_price_real[-24 * i:-24*j, 0]*100, "b", label = "price", color='purple')


# plt.step(range(0, 24 * (i-j)), reward_list[-24*i:-24*j], "r", label = "Reward")
# plt.step(df["date_hour"].values[-24 * i:-24*j], balance_wo_ES_real[-24 * i:-24*j], "yellow", label = "net_meter w/o ESS")
# plt.step(df["date_hour"].values[-24 * i:-24*j], balance_real[-24 * i:-24*j], "orange", label = "net_meter")
# plt.step(df["date_hour"].values[-24 * i:-24*j], balance_template_real[-24 * i:-24*j], "blue", label = "net_meter template")

plt.ylabel("Power [kW] / Price [€/MWh]")
plt.legend(loc='upper left')

plt.twinx()
plt.step(df["date_hour"].values[-24 * i:-24*j], SOC_list[-24 * i:-24*j], label = "SOC", color='red')
plt.step(df["date_hour"].values[-24 * i:-24*j], SOC_template_list[-24 * i:-24*j], label = "SOC template", color='pink')

plt.bar(df["date_hour"].values[-24 * i:-24*j],agent.action_list[action_list[-24 * i:-24*j]],width = 0.04, facecolor = "w", edgecolor = "k", label = "action")
plt.ylabel("Storage/action [-]")
plt.xlabel("Time")
plt.legend(loc='upper right')
plt.show()


plt.step(range(0, len(agent.model_loss)), agent.model_loss, "b", label = "price")
