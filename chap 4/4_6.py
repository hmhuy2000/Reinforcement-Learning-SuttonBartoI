import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns

# ======================== CFG ===============================

class CFG:
    n_state = 100
    p = 0.4
    garma = 1.0
    actions = np.arange(1, n_state)
    esp = 1e-8

# ======================== init ==============================

policy = np.zeros((CFG.n_state + 1))
value = np.zeros((CFG.n_state + 1))
value[CFG.n_state] = 1.0
history = []

# ======================= function ===========================

def value_of_action(state, action, value):
    win_state = state + action
    lose_state = state - action
    res = 0.0

    res += CFG.p * (CFG.garma * value[win_state])
    res += (1 - CFG.p) * (CFG.garma * value[lose_state])
    return res

def optimize_value_function(policy, value, history):
    diff = 0.0
    history.append(value.copy())
    new = np.zeros((CFG.n_state +1))
    new += -np.inf
    new[0] = 0
    new[CFG.n_state] = 1
    for i in range(1, CFG.n_state):
        for action in CFG.actions:
            if (action in np.arange(min(i, CFG.n_state - i) + 1)):
                new[i] = max(new[i], value_of_action(i, action, value))
        diff = max(diff, np.abs(new[i] - value[i]))

    value = new
    return value, diff

# ====================== main ==================================

step = 0
while(True):
    step += 1
    value, diff = optimize_value_function(policy, value, history)
    print(f'diff at step {step} is: {diff}')
    if (diff < CFG.esp):
        break

for i in range(1, CFG.n_state):
    returns = []
    for action in CFG.actions:
        if (action in np.arange(min(i, CFG.n_state - i) + 1)):
            returns.append(value_of_action(i, action, value))
        else:
            returns.append(-np.inf)
    policy[i] = CFG.actions[np.argmax(returns)]


plt.title('value function')
plt.xlabel('capital')
plt.ylabel('value')
for i in range(len(history)):
    plt.plot(history[i], label = f'step {i+1}')

plt.legend()
plt.savefig('figure_4_6_a.png')
plt.show()

plt.plot(policy)
plt.title('policy')
plt.xlabel('capital')
plt.ylabel('policy')
plt.savefig('figure_4_6_b.png')
plt.show()
