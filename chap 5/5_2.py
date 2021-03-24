import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns

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

def random_play(policy_player, policy_dealer):
    player_ace = 0
    player_ace_1 = 0
    dealer_ace = 0
    dealer_ace_1 = 0
    player_sum = 0
    dealer_sum = 0
    dealer_show = 0
    his = []

    while(player_sum <12):
        card = random_card()
        # print(f'player card: {card}')
        if (card == 1):
            player_ace += 1
        player_sum += value_card(card)
        if (player_sum > 21):
            player_sum -= 10
            player_ace_1 += 1
    
    if (True):
        card = random_card()
        dealer_show = card
        # print(f'dealer show {dealer_show}')
        if (card == 1):
            dealer_ace += 1
        dealer_sum += value_card(card)

        card = random_card()
        # print(f'dealer {card}')
        if (card == 1):
            dealer_ace += 1
        dealer_sum += value_card(card)
        if (dealer_sum > 21):
            dealer_sum -= 10
            dealer_ace_1 += 1

    while(True):
        if (player_sum > 21):
            # print(f'quát {player_sum}')
            return his, -1
        action = policy_player[player_sum]
        his.append((player_ace > player_ace_1, player_sum, dealer_show))
        if (action == CFG.STOP):
            break
        card = random_card()
        # print(f'player card: {card}')
        if (card == 1):
            player_ace += 1
        player_sum += value_card(card)
        if (player_sum > 21 and player_ace > player_ace_1):
            player_sum -= 10
            player_ace_1 += 1
    
    while(True):
        if (dealer_sum == 21):
            # print(f'player {player_sum} dealer {dealer_sum}')
            if (player_sum == 21):
                return his, 0
            else:
                return his, -1
        if (dealer_sum > 21):
            # print(f'player {player_sum} dealer {dealer_sum}')
            return his, 1
        action = policy_dealer[dealer_sum]
        if (action == CFG.STOP):
            break
        card = random_card()
        # print(f'dealer {card}')
        if (card == 1):
            dealer_ace += 1
        dealer_sum += value_card(card)
        if(dealer_sum > 21 and dealer_ace > dealer_ace_1):
            dealer_sum -= 10
            dealer_ace_1 += 1
    
    # print(f'player {player_sum} dealer {dealer_sum}')
    if (player_sum < dealer_sum):
        return his, -1
    if (player_sum == dealer_sum):
        return his, 0
    if (player_sum > dealer_sum):
        return his, 1
        

def MonteCarloPrediction(Num_iter):

    # ========================== init =======================

    policy_dealer = np.zeros((22))
    policy_player = np.zeros((22))
    policy_player[:20] = CFG.HIT
    policy_player[20:] = CFG.STOP
    policy_dealer[:17] = CFG.HIT
    policy_dealer[17:] = CFG.STOP

    value = np.zeros((2, 10, 10))
    cnt = np.ones((2, 10, 10))

    for iter in trange(Num_iter):
        his, reward = random_play(policy_player, policy_dealer)
        for (usable, player_sum, dealer_show) in his:
            value[int(usable), player_sum - 12, dealer_show - 1] += reward
            cnt[int(usable), player_sum - 12, dealer_show - 1] += 1
    
    arr = value / cnt
    return arr[0,...], arr[1,...]


# ======================== main ==========================

NoUsable10k, Usable10k = MonteCarloPrediction(10000)
NoUsable500k, Usable500k = MonteCarloPrediction(500000)

ax = sns.heatmap(NoUsable10k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_2_10k_NoUsable.png')
plt.close()

ax = sns.heatmap(Usable10k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_2_10k_Usable.png')
plt.close()

ax = sns.heatmap(NoUsable500k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_2_500k_NoUsable.png')
plt.close()

ax = sns.heatmap(Usable500k, cmap="YlGnBu", xticklabels=range(1, 11)
    ,yticklabels=list(range(12, 22)))
plt.savefig('figure_5_2_500k_Usable.png')
plt.close()
