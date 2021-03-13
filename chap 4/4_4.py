import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns

# ================================== config =====================================================

class CFG:
    max_move_car = 5
    max_car = 20
    expect_request_1 = 3
    expect_request_2 = 4
    expect_return_1 = 3
    expect_return_2 = 2
    actions = np.arange(-max_move_car, max_move_car+1)
    garma = 0.9
    price = 10.0
    cost = 2.0
# =================================== init ========================================================
from scipy.stats import poisson
policy = np.zeros((CFG.max_car + 1, CFG.max_car + 1), dtype = int)
value = np.zeros((CFG.max_car + 1, CFG.max_car + 1))

# =================================== function ====================================================
poisson_cache = dict()
def poisson_prob(n, lamb):
    global poisson_cache
    key = n * 100 + lamb
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lamb)
    return poisson_cache[key]

def get_value_state(state, action, policy, value, constant = True):
    res = 0.0
    res -= CFG.cost * np.abs(action)
    (car_1, car_2) = state
    car_1 -= action
    car_2 += action

    for request_1 in range(11):
        for request_2 in range(11):
            prob_request = poisson_prob(request_1, CFG.expect_request_1) * poisson_prob(request_2, CFG.expect_request_2)
            valid_request_1 = min(car_1, request_1)
            valid_request_2 = min(car_2, request_2)
            reward = CFG.price * (valid_request_1 + valid_request_2)
            tmp_car_1 = car_1 - valid_request_1
            tmp_car_2 = car_2 - valid_request_2
            
            if (constant):
                tmp_car_1 = min(tmp_car_1 + CFG.expect_return_1, CFG.max_car)
                tmp_car_2 = min(tmp_car_2 + CFG.expect_return_2, CFG.max_car)
                # print(state, action, prob_request, reward, value[tmp_car_1, tmp_car_2])
                res += prob_request * (reward + CFG.garma * value[tmp_car_1, tmp_car_2])
                # print(res)
                continue

            for return_1 in range(CFG.max_car + 1):
                for return_2 in range(CFG.max_car + 1):
                    prob_return = poisson_prob(return_1, CFG.expect_return_1) * poisson_prob(return_2, CFG.expect_return_2)
                    tmp_car_1 = min(tmp_car_1 + return_1, CFG.max_car)
                    tmp_car_2 = min(tmp_car_2 + return_2, CFG.max_car)
                    prob = prob_request * prob_return
                    res += prob * (reward + CFG.garma * value[tmp_car_1, tmp_car_2])
    
    return res

def get_optimal_value_map(policy, value):
    while(True):
        new = np.zeros((CFG.max_car + 1, CFG.max_car + 1))
        for i in range(CFG.max_car + 1):
            for j in range(CFG.max_car + 1):
                new[i,j] = get_value_state((i, j), policy[i, j], policy, value)

        print(f'value difference :{np.max(np.abs(value - new))}')
        if(np.max(np.abs(value - new))<1e-4):
            break
        value = new


    return value
                



# =================================== main ========================================================

step = 0
while(True):
    step += 1
    value = get_optimal_value_map(policy, value)
    print(f'Done value estimate for step {step}')
    done = True

    for i in range(CFG.max_car + 1):
        for j in range(CFG.max_car +1):
            old = policy[i,j]
            action_reward = []
            for action in CFG.actions:
                if (0 <= action <= i and j + action <= 20) or (-j <= action <= 0 and i - action <= 20):
                    action_reward.append(get_value_state((i, j), action, policy, value))
                else:
                    action_reward.append(-np.inf)

            new = CFG.actions[np.argmax(action_reward)]
            policy[i, j] = new
            if (new != old):
                done = False
    
    print(f'Result of step number {step} is: {done}')
    if (done):
        fig, ax = plt.subplots()
        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False)
        ax.table(cellText=policy,loc='center')
        plt.plot()
        plt.savefig('figure_4_4_a.png')
        plt.show()

        ax = sns.heatmap(value, cmap="YlGnBu")
        plt.savefig('figure_4_4_b.png')
        plt.show()


        break