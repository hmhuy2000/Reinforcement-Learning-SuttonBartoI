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
    n_run = 300
    n_episode = 10000
    target_value = -0.27726



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

def random_move():
    if np.random.randint(10) %2:
        return CFG.HIT
    return CFG.STOP

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
        action = random_move()
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
        

def off_MonteCarlo(Num_iter, debug = False):

    # ========================== init =======================

    policy_dealer = np.zeros((22))
    policy_dealer[:17] = CFG.HIT
    policy_dealer[17:] = CFG.STOP

    target_policy = np.zeros((22))
    target_policy[:20] = CFG.HIT
    target_policy[20:] = CFG.STOP    
    isr = []
    rewards = []
    for iter in range(Num_iter):
        if (debug):
            print(f'---------------- {iter} -------------------------')
        init_usable = True
        init_show = 2
        init_player_sum = 13
        init_action = random.choice(CFG.actions)

        his, reward = random_play(target_policy, policy_dealer,
        (init_usable, init_show, init_player_sum, init_action), debug)
        if (debug):
            print(his, reward)
        numerator = 1.0
        demoninator = 1.0
        for (usable, player_sum, dealer_show, action) in his:
            if (action == target_policy[player_sum]):
                demoninator *= 0.5
            else:
                numerator = 0.0
                break
        isr.append(numerator/demoninator)
        rewards.append(reward)

    rewards = np.asarray(rewards, dtype = float)
    isr = np.asarray(isr, dtype = float)
    rewards_isr = rewards * isr
    rewards_isr = np.add.accumulate(rewards_isr)
    isr = np.add.accumulate(isr)
    oridinary = rewards_isr/np.arange(1, Num_iter + 1)
    # print(f'ori: {oridinary[:30]}')

    weighted = np.zeros(rewards_isr.shape)
    for i in range(weighted.shape[0]):
        if (isr[i] == 0):
            continue
        weighted[i] = rewards_isr[i]/isr[i]
    return (oridinary, weighted)

            

# ======================== main ==========================

oridinary = np.zeros((CFG.n_episode))
weighted = np.zeros((CFG.n_episode))
for r in trange(CFG.n_run):
    tmp_oridinary, tmp_weighted = off_MonteCarlo(CFG.n_episode)
    oridinary += (tmp_oridinary - CFG.target_value) ** 2
    weighted += (tmp_weighted - CFG.target_value) ** 2
oridinary /= CFG.n_run
weighted /= CFG.n_run

plt.plot(oridinary, label = 'oridinary')
plt.plot(weighted, label = 'weight')
plt.xscale('log')
plt.xlabel('episode')
plt.ylabel('MSE')
plt.legend()
plt.savefig('figure_5_7.png')
plt.show()
