# ========================= import ===================================

import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns
import random

# ======================== CFG ======================================

class CFG:
    height = 7
    width = 10
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    action_Up = [-1, 0]
    action_Down = [1, 0]
    action_Left = [0, -1]
    action_Right = [0, 1]
    actions = [action_Up, action_Down, action_Left, action_Right]
    epsilon = 0.1
    alpha = 0.5
    pend = -1
    start = [3, 0]
    goal = [3,7]
    episode = 1000

# ==================== init ==========================================

Q = np.zeros((CFG.height, CFG.width, 4))
steps = []

# ==================== function ======================================

def step(state, action):
    [curX, curY] = state
    newX = curX + action[0] - CFG.wind[curY]
    newY = curY + action[1] 
    newX = min(max(0, newX), CFG.height - 1)
    newY = min(max(0, newY), CFG.width  -1)
    return [newX, newY]


def play(Q):
    [curX,curY] = CFG.start
    time = 0
    if (np.random.random()<CFG.epsilon):
        action = np.random.choice(np.arange(4))
    else:
        action = np.argmax(Q[curX, curY, :])
    
    while([curX, curY] != CFG.goal):
        [nextX, nextY] = step([curX, curY], CFG.actions[action])

        if (np.random.random()<CFG.epsilon):
            Next_action = np.random.choice(np.arange(4))
        else:
            Next_action = np.argmax(Q[nextX, nextY, :])

        Q[curX, curY, action] += CFG.alpha * (CFG.pend + Q[nextX, nextY, Next_action] - Q[curX, curY, action]) 
        curX = nextX
        curY = nextY
        action = Next_action
        time += 1
    return time

# =================== main ============================================

for ep in range(CFG.episode):
    steps.append(play(Q))

# print(steps)

steps = np.add.accumulate(steps)
plt.plot(steps, np.arange(CFG.episode))
plt.xlabel('Times steps')
plt.ylabel('Episodes')
plt.savefig('figure_6_11/figure_6_11.png')