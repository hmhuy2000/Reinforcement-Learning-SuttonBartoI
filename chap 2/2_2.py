import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class CFG:
    n = 10
    mean = 0.0
    variance = 1.0
    t = 1000
    esp = [0, 0.01, 0.05, 0.1, 0.15, 0.2]
    c_param = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3]
    n_try = 2000
    

class bandit():
    def __init__(self, m, v):
        self.m = m
        self.v = v
        self.mean = 0.0
        self.cnt = 0

    def reset(self):
        self.mean = 0.0
        self.cnt = 0

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

def get_result_eps(e):
    bandits = [bandit(np.random.randn(),CFG.variance) for i in range(CFG.n)]
    res = []
    for _ in range(CFG.t):
        if (np.random.random()<e):
            choose = np.random.choice(CFG.n)
        else:
            choose = np.argmax([ban.mean for ban in bandits])
        
        val = bandits[choose].get_reward()
        res.append(val)
        bandits[choose].update(val)
    # print(res)
    return res
    
def get_result_UCB(c):
    bandits = [bandit(np.random.randn(),CFG.variance) for i in range(CFG.n)]
    res = []
    for t in range(CFG.t):
        # if (np.random.random()<e):
        #     choose = np.random.choice(CFG.n)
        # else:
        choose = np.argmax([ban.get_UCB_score(t, c) for ban in bandits])
        
        val = bandits[choose].get_reward()
        res.append(val)
        bandits[choose].update(val)
    # print(res)
    return res

plt.figure(figsize=(20, 10))

# for e in CFG.esp:
#     res = np.zeros(CFG.t)
#     for tr in trange(CFG.n_try):
#         res += get_result_eps(e)
#     res /= CFG.n_try
#     plt.plot(res, label = f'epsilon e = {e}')

for c in CFG.c_param:
    res = np.zeros(CFG.t)
    for tr in trange(CFG.n_try, desc=f'UCB c = {c}'):
        res += get_result_UCB(c)
    res /= CFG.n_try
    plt.plot(res, label = f'UCB c = {c}')

plt.xlabel('step')
plt.ylabel('average reward')
plt.legend()
plt.savefig('figure_2_2_a.png')
plt.show()
