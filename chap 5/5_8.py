# ========================= import ===================================

import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import random

# ========================= CFG ======================================

class CFG:
    action_back = 0
    action_end = 1
    actions = [action_back, action_end]
    n_run = 10
    n_episode = 50000000

# ========================= define policy ============================

def target_policy():
    return CFG.action_back

def behavier_policy():
    return random.choice(CFG.actions)

def random_back():
    return random.choice(np.arange(10)) == 0

# ========================= function ==================================

def MonteCarlo_ordinary_sampling(n_episode, n_r):
    rewards = []
    isr = []
    for ep in tqdm(range(n_episode), desc = f'try {n_r}'):
        num = 1.0
        dem = 1.0
        cnt = 0
        while(True):
            action = behavier_policy()
            if (action == CFG.action_end):
                num = 0
                isr.append(num/dem)
                rewards.append(0)
                break
            else:
                cnt += 1
                dem *= 0.5
                if (random_back()):
                    isr.append(num/dem)
                    rewards.append(1)
                    break
    
    rewards = np.asarray(rewards)
    isr = np.asarray(isr)
    rewards_isr = rewards * isr
    rewards_isr = np.add.accumulate(rewards_isr)
    ordinary = rewards_isr/np.arange(1, n_episode + 1)
    return ordinary
                

            


# ========================= main ======================================

for r in range(CFG.n_run):
    ordinary = MonteCarlo_ordinary_sampling(CFG.n_episode, r)
    plt.plot(ordinary, label = f'{r}')
    plt.xscale('log')
    plt.xlabel('episode')
    plt.ylabel('state value')
    plt.savefig(f'figure_5_8_{r}.png')
    plt.close()
