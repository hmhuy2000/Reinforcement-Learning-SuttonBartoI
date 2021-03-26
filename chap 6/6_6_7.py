# ========================= import ===================================

import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns
import random

# ======================== CFG ======================================

class CFG:

    episode = 100
    MC_alpha = [0.01, 0.02, 0.03, 0.04]
    TD_alpha = [0.05, 0.1, 0.15]
    target = [1/6, 2/6, 3/6, 4/6, 5/6]
    run = 100
    garma = 1.0

# ======================== init ======================================

class game():
    def __init__(self, alpha, TypeAlgo):
        self.value_function = np.full((7), 0.5)
        self.policy = np.full((7,2), 0.5)
        self.Left = -1
        self.Right = 1
        self.actions = [self.Left, self.Right]
        self.FristEnd = 0
        self.LastEnd = 6
        self.value_function[self.FristEnd] = 0
        self.value_function[self.LastEnd] = 1
        self.alpha = alpha
        self.TypeAlgo = TypeAlgo

    def reset(self):
        pass

    def step(self):
        if (self.TypeAlgo == 'MC'):
            return self.step_MC()
        if (self.TypeAlgo == 'TD'):
            return self.step_TD()

    def step_TD(self):
        his = []
        cur = 3
        reward = 0

        while(cur != self.FristEnd and cur != self.LastEnd):
            action = np.random.choice(self.actions, p=self.policy[cur, :])
            # if (cur + action == self.LastEnd):
            #     state_reward = 1
            # else:
            #     state_reward = 0
            state_reward = 0
            his.append((cur, action, state_reward))
            cur += action
            reward += state_reward

        return his, reward

    def step_MC(self):
        his = []
        cur = 3
        reward = 0

        while(cur != self.FristEnd and cur != self.LastEnd):
            action = np.random.choice(self.actions, p=self.policy[cur, :])
            if (cur + action == self.LastEnd):
                state_reward = 1
            else:
                state_reward = 0
            his.append((cur, action, state_reward))
            cur += action
            reward += state_reward

        return his, reward

    def update(self, state, action, state_reward, reward):
        if (self.TypeAlgo == 'MC'):
            return self.update_MC(state, reward)
        if (self.TypeAlgo == 'TD'):
            return self.update_TD(state, state + action, state_reward)

    def update_TD(self, state, Nstate, state_reward):
        up = self.alpha * (state_reward + CFG.garma * self.value_function[Nstate] - self.value_function[state])
        self.value_function[state] += up
        return up
    
    def update_MC(self, state, reward):
        up = self.alpha * (reward - self.value_function[state])
        self.value_function[state] += up
        return up

    def get_value_function(self):
        return self.value_function

# ======================= function ===================================

def MC_evaluate(episode, alpha):
    total_error = np.zeros((episode), dtype = float)
    for _ in trange(CFG.run, desc = f'MC - {alpha}'):
        env = game(alpha = alpha, TypeAlgo = 'MC')
        error = []
        for ep in range(episode):
            env.reset()
            his, reward = env.step()
            for i in range(len(his)):
                state, action, state_reward = his[i]
                env.update(state, action, state_reward, reward)
            error.append(np.sqrt(np.sum(np.power(env.get_value_function()[1:-1] - CFG.target, 2))/5.0))
        error = np.asarray(error)
        total_error += error
    return total_error/CFG.run

def TD_evaluate(episode, alpha, TDplot = False):
    total_error = np.zeros((episode), dtype = float)
    for r in trange(CFG.run, desc = f'TD - {alpha}'):
        env = game(alpha = alpha, TypeAlgo = 'TD')
        error = []
        for ep in range(episode):
            env.reset()
            his, reward = env.step()
            update = np.zeros((7), dtype = float)
            for i in range(len(his)):
                state, action, state_reward = his[i]
                env.update(state, action, state_reward, reward)

            env.value_function += update
            error.append(np.sqrt(np.sum(np.power(env.get_value_function()[1:-1] - CFG.target, 2))/5.0))
            if (r == 1 and TDplot):
                if (ep == 0 or ep == 1 or ep == 10 or ep == 99):
                    plt.plot(env.get_value_function()[1:-1], label = f'{ep}')
        error = np.asarray(error)
        total_error += error
        if (r == 1 and TDplot):
            plt.plot(CFG.target,label = 'true values')
            plt.xticks([0,1,2,3,4],['A','B','C','D','E'])
            plt.legend()
            plt.savefig('figure_6_6/6_6.png')
            plt.close()
    
    return total_error/CFG.run


# ======================= main ========================================

for alpha in CFG.MC_alpha:
    error = MC_evaluate(CFG.episode, alpha)
    plt.plot(error,'--', label = f'MC - {alpha}')

for alpha in CFG.TD_alpha:
    error = TD_evaluate(CFG.episode, alpha)
    plt.plot(error, label = f'TD - {alpha}')
plt.legend()
plt.savefig('figure_6_7/6_7.png')
plt.close()

TD_evaluate(CFG.episode, 0.1, True)