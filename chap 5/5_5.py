import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns
import random

# ========================== CFG =======================

class CFG:
    HIT = 1
    STOP = 0
    actions = [STOP, HIT]
    WIN = 1
    DRAW = 0
    LOSE = -1


# ======================== function ======================

def random_card():
    card = np.random.randint(13) + 1
    card = min(card, 10)
    return card

def value_card(card):
    if (card == 1):
        return 11
    else:
        return card

def random_play(policy_player, policy_dealer, init_state = None, debug = False):
    player_ace = 0
    player_ace_1 = 0
    dealer_ace = 0
    dealer_ace_1 = 0
    player_sum = 0
    dealer_sum = 0
    dealer_show = 0
    his = []
    if (init_state):
        (player_ace, dealer_show, player_sum, action) = init_state
        if (debug):
            print(f'player init {player_sum} dealer show {dealer_show} action {action}')

        if (dealer_show == 1):
            dealer_ace += 1
        dealer_sum += value_card(dealer_show)

        card = random_card()
        if (card == 1):
            dealer_ace += 1
        dealer_sum += value_card(card)
        if (dealer_sum > 21):
            dealer_sum -= 10
            dealer_ace_1 += 1

        his.append((player_ace > player_ace_1, player_sum, dealer_show, action))
        if (action == CFG.HIT):
            card = random_card()
            if (debug):
                print(f'player {player_sum} {card}')
            if (card == 1):
                player_ace += 1
            player_sum += value_card(card)
            if (player_sum > 21 and player_ace > player_ace_1):
                player_sum -= 10
                player_ace_1 += 1
        

    else:
        while(player_sum <12):
            card = random_card()
            if (card == 1):
                player_ace += 1
            player_sum += value_card(card)
            if (player_sum > 21):
                player_sum -= 10
                player_ace_1 += 1
        
        if (True):
            card = random_card()
            dealer_show = card
            if (card == 1):
                dealer_ace += 1
            dealer_sum += value_card(card)

            card = random_card()
            if (card == 1):
                dealer_ace += 1
            dealer_sum += value_card(card)
            if (dealer_sum > 21):
                dealer_sum -= 10
                dealer_ace_1 += 1

    while(True):
        if (player_sum > 21):
            if (debug):
                print(f'quát {player_sum}')
            return his, -1
        action = policy_player[int(player_ace > player_ace_1), player_sum, dealer_show]
        his.append((player_ace > player_ace_1, player_sum, dealer_show, action))
        if (action == CFG.STOP):
            break
        card = random_card()
        if (debug):
            print(f'player {player_sum} {card}')
        if (card == 1):
            player_ace += 1
        player_sum += value_card(card)
        if (player_sum > 21 and player_ace > player_ace_1):
            player_sum -= 10
            player_ace_1 += 1
    
    while(True):
        if (dealer_sum == 21):
            if(debug):
                print(f'player {player_sum} dealer {dealer_sum}')
            if (player_sum == 21):
                return his, 0
            else:
                return his, -1
        if (dealer_sum > 21):
            return his, 1
        action = policy_dealer[dealer_sum]
        if (action == CFG.STOP):
            break
        card = random_card()
        if(debug):
            print(f'dealer {dealer_sum} {card}')
        if (card == 1):
            dealer_ace += 1
        dealer_sum += value_card(card)
        if(dealer_sum > 21 and dealer_ace > dealer_ace_1):
            dealer_sum -= 10
            dealer_ace_1 += 1
    
    if(debug):
        print(f'player sum {player_sum} dealer sum {dealer_sum}')
    if (player_sum < dealer_sum):
        return his, -1
    if (player_sum == dealer_sum):
        return his, 0
    if (player_sum > dealer_sum):
        return his, 1
        

def MonteCarloPrediction(Num_iter, debug = False):

    # ========================== init =======================

    policy_dealer = np.zeros((22))
    policy_dealer[:17] = CFG.HIT
    policy_dealer[17:] = CFG.STOP

    policy_player = np.zeros((2, 22, 11), dtype = int)
    for i in range(2):
        for j in range(22):
            for k in range(11):
                policy_player[i,j,k] = random.choice(CFG.actions)

    

    value_action = np.zeros((2, 10, 10, 2))
    cnt = np.ones((2, 10, 10, 2))

    for iter in trange(Num_iter):
        if (debug):
            print(f'---------------- {iter} -------------------------')
        check = set()
        init_usable = random.choice(range(2))
        init_show = random_card()
        init_player_sum = random.choice(range(12,22))
        init_action = random.choice(CFG.actions)

        his, reward = random_play(policy_player, policy_dealer,
        (init_usable, init_show, init_player_sum, init_action), debug)
        if (debug):
            print(his, reward)
        for (usable, player_sum, dealer_show, action) in his:
            if ((usable, player_sum, dealer_show, action) in check):
                continue
            check.add((usable, player_sum, dealer_show, action))

            value_action[int(usable), player_sum - 12, dealer_show - 1, action] += reward
            cnt[int(usable), player_sum - 12, dealer_show - 1, action] += 1
            Q = np.zeros((2))
            Q[0] = value_action[int(usable), player_sum - 12, dealer_show - 1, 0]/cnt[int(usable), player_sum - 12, dealer_show - 1, 0]
            Q[1] = value_action[int(usable), player_sum - 12, dealer_show - 1, 1]/cnt[int(usable), player_sum - 12, dealer_show - 1, 1]
            policy_player[int(usable), player_sum, dealer_show] = np.argmax(Q)
    arr = value_action/cnt
    return policy_player[0, 12:,1:], policy_player[1, 12:,1:], arr

# ======================== main ==========================

NoUsable500k, Usable500k, arr = MonteCarloPrediction(10000000)

value = np.zeros((2,10,10))

for i in range(2):
    for j in range(10):
        for k in range(10):
            value[i,j,k] = np.max(arr[i,j,k,:])


ax = sns.heatmap(value[0,...], cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_5_value_NoUsable.png')
plt.close()

ax = sns.heatmap(value[1,...], cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_5_value_Usable.png')
plt.close()

ax = sns.heatmap(NoUsable500k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_5_policy_NoUsable.png')
plt.close()


ax = sns.heatmap(Usable500k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_5_policy_Usable.png')
plt.close()