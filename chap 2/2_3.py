import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from numba import jit, cuda 

import warnings
warnings.filterwarnings("ignore")


class CFG:
    n = 10
    mean = 0.0
    variance = 1.0
    t = 3000
    esp = [0, 0.01, 0.05, 0.1, 0.15, 0.2]
    c_param = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3]
    grad_param = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3]
    n_try = 2000
    
  
class bandit():
    def __init__(self, m, v, alpha = 0):
        self.m = m
        self.v = v
        self.mean = 0.0
        self.cnt = 0
        self.H = 0
        self.alpha = alpha

    def get_UCB_score(self, t, c):
        if self.cnt == 0:
            sq = 10000
        else:
            sq = c * np.sqrt(np.log(t)/self.cnt)
        return self.mean + sq

    def get_reward(self):
        reward = self.v * np.random.randn() + self.m 
        return reward

    def update(self, reward):
        self.cnt += 1
        self.mean = self.mean + 1/self.cnt * (reward - self.mean)

    def update_grad(self, reward, avg_reward, prob, is_choose):
        self.cnt += 1
        self.H = self.H + self.alpha * (reward - avg_reward) * (is_choose - prob)
    
def get_result_grad(alpha):
    bandits = [bandit(np.random.randn(),CFG.variance, alpha) for i in range(CFG.n)]
    best_choice = np.argmax([ban.m for ban in bandits])
    res = []
    optimal = []
    avg_reward = 0
    for t in range(CFG.t):
        sumH = np.sum([np.exp(ban.H) for ban in bandits])
        probs = [np.exp(ban.H)/sumH for ban in bandits]
        choose = np.random.choice(range(CFG.n), p = probs)
        # print(choose, [ban.H for ban in bandits], probs)
        val = bandits[choose].get_reward()
        avg_reward += (val - avg_reward)/(t+1)
        res.append(val)
        if (choose == best_choice):
            optimal.append(1)
        else:
            optimal.append(0)

        bandits[choose].update_grad(val, avg_reward, probs[choose], 1)
        for i in range(CFG.n):
            if i == choose:
                continue
            bandits[i].update_grad(val, avg_reward, probs[choose], 0)
    # print(res)
    return res, optimal

plt.figure(figsize=(20, 10))

def get_result_eps(e):
    bandits = [bandit(np.random.randn(),CFG.variance) for i in range(CFG.n)]
    best_choice = np.argmax([ban.m for ban in bandits])
    res = []
    optimal = []
    for _ in range(CFG.t):
        if (np.random.random()<e):
            choose = np.random.choice(CFG.n)
        else:
            choose = np.argmax([ban.mean for ban in bandits])
        
        val = bandits[choose].get_reward()
        res.append(val)
        if (choose == best_choice):
            optimal.append(1)
        else:
            optimal.append(0)
        bandits[choose].update(val)
    # print(res)
    return res, optimal
    
def get_result_UCB(c):
    bandits = [bandit(np.random.randn(),CFG.variance) for i in range(CFG.n)]
    best_choice = np.argmax([ban.m for ban in bandits])
    res = []
    optimal = []
    for t in range(CFG.t):
        choose = np.argmax([ban.get_UCB_score(t, c) for ban in bandits])

        val = bandits[choose].get_reward()
        res.append(val)
        if (choose == best_choice):
            optimal.append(1)
        else:
            optimal.append(0)
        bandits[choose].update(val)
    # print(res)
    return res, optimal


for e in CFG.esp:
    res = np.zeros(CFG.t)
    optimal = np.zeros(CFG.t)
    for tr in trange(CFG.n_try, desc=f'epsilon e = {e}'):
        res_tmp, optimal_tmp = get_result_eps(e)
        res += res_tmp
        optimal += optimal_tmp
    res /= CFG.n_try
    optimal /= CFG.n_try
    plt.subplot(1,2,1)
    plt.plot(res, label = f'epsilon e = {e}')
    plt.subplot(1,2,2)
    plt.plot(optimal, label = f'epsilon e = {e}')

for c in CFG.c_param:
    res = np.zeros(CFG.t)
    optimal = np.zeros(CFG.t)
    for tr in trange(CFG.n_try, desc=f'UCB c = {c}'):
        res_tmp, optimal_tmp = get_result_UCB(c)
        res += res_tmp
        optimal += optimal_tmp
    res /= CFG.n_try
    optimal /= CFG.n_try
    plt.subplot(1,2,1)
    plt.plot(res, label = f'UCB c = {c}')
    plt.subplot(1,2,2)
    plt.plot(optimal, label = f'UCB c = {c}')

for alpha in CFG.grad_param:
    res = np.zeros(CFG.t)
    optimal = np.zeros(CFG.t)
    for tr in trange(CFG.n_try, desc=f'gradient alpha = {alpha}'):
        res_tmp, optimal_tmp = get_result_grad(alpha)
        res += res_tmp
        optimal += optimal_tmp
    res /= CFG.n_try
    optimal /= CFG.n_try
    plt.subplot(1,2,1)
    plt.plot(res, label = f'gradient alpha = {alpha}')
    plt.subplot(1,2,2)
    plt.plot(optimal, label = f'gradient alpha = {alpha}')

plt.subplot(1,2,1)
plt.xlabel('step')
plt.ylabel('average reward')
plt.subplot(1,2,2)
plt.xlabel('step')
plt.ylabel('optimal action')
plt.legend()
plt.savefig('tmp.png')
plt.show()


# ==============================================================================

