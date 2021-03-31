# ========================= import ===================================

import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns
import random

# ======================== CFG ======================================

class CFG:
    height = 5
    width = 12
    start = [4, 0]
    goal = [4, 11]
    action_Up = [-1, 0]
    action_Down = [1, 0]
    action_Left = [0, -1]
    action_Right = [0, 1]
    actions = [action_Up, action_Down, action_Left, action_Right]
    epsilon = 0.1
    episode = 500
    alpha = 0.5
    runs = 300
    
# ========================= init =======================================

q_Qlearning = np.zeros((CFG.height, CFG.width, 4))
q_Sarsa = np.zeros((CFG.height, CFG.width, 4))
reward_Qlearning = np.zeros((CFG.episode))
reward_Sarsa = np.zeros((CFG.episode))
# ========================= function ===================================

def step(state, action):
    [curX, curY] = state
    newX = curX + action[0]
    newY = curY + action[1] 
    newX = min(max(0, newX), CFG.height - 1)
    newY = min(max(0, newY), CFG.width  -1)
    if (newX == CFG.height -1 and 0 < newY < CFG.width -1):
        return CFG.start,-100
    return [newX, newY], -1

def Qlearning(Q):
    [curX,curY] = CFG.start
    total = 0
    
    while([curX, curY] != CFG.goal):
        if (np.random.random()<CFG.epsilon):
            action = np.random.choice(np.arange(4))
        else:
            action = np.argmax(Q[curX, curY, :])

        [nextX, nextY], reward = step([curX, curY], CFG.actions[action])
        total += reward
        Q[curX, curY, action] += CFG.alpha * (reward + np.max(Q[nextX, nextY, :]) - Q[curX, curY, action]) 
        curX = nextX
        curY = nextY
    return total

def Sarsa(Q):
    [curX,curY] = CFG.start
    total = 0
    if (np.random.random()<CFG.epsilon):
        action = np.random.choice(np.arange(4))
    else:
        action = np.argmax(Q[curX, curY, :])
    
    while([curX, curY] != CFG.goal):
        [nextX, nextY], reward = step([curX, curY], CFG.actions[action])
        total += reward
        if (np.random.random()<CFG.epsilon):
            Next_action = np.random.choice(np.arange(4))
        else:
            Next_action = np.argmax(Q[nextX, nextY, :])

        Q[curX, curY, action] += CFG.alpha * (reward + Q[nextX, nextY, Next_action] - Q[curX, curY, action]) 
        curX = nextX
        curY = nextY
        action = Next_action
    return total

# ========================= main ========================================

for r in trange(CFG.runs):
    Sarsa_tmp = []
    Qlearning_tmp = []
    for ep in range(CFG.episode):
        Sarsa_tmp.append(Sarsa(q_Sarsa))
        Qlearning_tmp.append(Qlearning(q_Qlearning))
    reward_Sarsa += Sarsa_tmp
    reward_Qlearning += Qlearning_tmp
reward_Sarsa/= CFG.runs
reward_Qlearning/= CFG.runs

# print(reward_Sarsa)
plt.plot(reward_Sarsa, label='Sarsa')
plt.plot(reward_Qlearning, label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.ylim([-100, 0])
plt.legend()
plt.savefig('figure_6_13/figure_6_13.png')
plt.show()