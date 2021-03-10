import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd

N, M = 4,4
n_try = 1
ans = np.zeros((N, M))

for _ in trange(n_try):
    res = np.zeros((N, M))

    special = {}
    act = {}
    dirX = [-1, 0, 1, 0]
    dirY = [0, 1, 0, -1]
    grama = 1.0
    # ========================= def ============================================
    def check(x, y):
        return x>=0 and x<N and y>= 0 and y<M

    def softmax(x):
        e_x = x
        return e_x / e_x.sum(axis=0) 
    # ========================= init ============================================


    for i in range(N):
        for j in range(M):
            tmp = []
            reward = -1
            if ((i,j) == (0,0) or (i,j) == (N-1, M-1)):
                reward = 0
                act[(i,j)] = [((i,j), reward)]
                continue

            for stt in range(len(dirX)):

                newX = i + dirX[stt]
                newY = j + dirY[stt]
                if(check(newX, newY) == 0):
                    tmp.append(((i,j),reward))
                else:
                    tmp.append(((newX, newY), reward))

            act[(i,j)] = tmp

    # ======================= main ==============================================

    while(1):
        new_res = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                for ((newX, newY), reward) in act[(i,j)]:
                    new_res[i,j] += 0.25 * (reward + grama * res[newX, newY])
        if(np.sum(np.abs(res-new_res))<1e-4):
            break
        res = new_res
    
    ans += res

ans/= n_try

policy = []
for i in range(N):
    tmp = []
    for j in range(M):
        tmp.append(0)
    policy.append(tmp)

ans = np.round(ans, decimals = 2)
for i in range(N):
    for j in range(M):
        best_act=[]
        for ((newX, newY), reward) in act[(i,j)]:
            if (len(best_act) == 0):
                best_act.append([newX, newY])
                continue

            (bestX, bestY) = best_act[0]
            if (ans[bestX, bestY] < ans[newX, newY]):
                best_act = [[newX, newY]]
        policy[i][j] = np.array(best_act)

fig, ax = plt.subplots()
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
ax.table(cellText=ans,loc='center')
plt.plot()
plt.savefig('figure_4_2_a.png')
plt.show()

fig, ax = plt.subplots()
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
ax.table(cellText=policy,loc='center')
plt.plot()
plt.savefig('figure_4_2_b.png')
plt.show()
