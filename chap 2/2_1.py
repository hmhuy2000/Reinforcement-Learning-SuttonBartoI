import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


class CFG:
    n = 10
    mean = 0.0
    variance = 1.0
    t = 1000
    esp = [0, 0.01, 0.05, 0.1, 0.15, 0.2]
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

    def get_reward(self):
        reward = self.v * np.random.randn() + self.m 
        return reward

    def update(self, reward):
        self.cnt += 1
        self.mean = self.mean + 1/self.cnt * (reward - self.mean)

def get_result(e):
    bandits = [bandit(np.random.randn(),CFG.variance) for i in range(CFG.n)]
    res = []
    global cnt 
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
    

plt.figure(figsize=(20, 10))

for e in CFG.esp:
    res = np.zeros(CFG.t)
    for tr in trange(CFG.n_try):
        res += get_result(e)
    print(res.shape)
    res /= CFG.n_try
    # print(res)
    plt.plot(res, label = e)
    print(f'done {e}')

plt.xlabel('step')
plt.ylabel('average reward')
plt.legend()
plt.savefig('figure_2_1.png')
plt.show()
