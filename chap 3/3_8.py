import numpy as np 
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd

N, M = 10, 10
n_try = 1000
ans = np.zeros((N, M))

for _ in trange(n_try):
    res = np.zeros((N, M))

    special = {}
    special[(0,1)] = ((4,1),10)
    special[(0,3)] = ((2,3),5)
    act = {}
    dirX = [-1, 0, 1, 0]
    dirY = [0, 1, 0, -1]
    grama = 0.9
    # ========================= def ============================================
    def check(x, y):
        return x>=0 and x<N and y>= 0 and y<M

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) 
    # ========================= init ============================================


    for i in range(N):
        for j in range(M):
            tmp = []
            if (i, j) in special.keys():
                tmp.append(special[(i,j)])
                act[(i,j)] = tmp
                continue
            acc = np.random.randn(4)
            acc = softmax(acc)
            for stt in range(len(dirX)):
                newX = i + dirX[stt]
                newY = j + dirY[stt]
                if(check(newX, newY) == 0):
                    if(((i,j),-1) in tmp):
                        continue
                    tmp.append(((i,j),-1))
                else:
                    tmp.append(((newX, newY), 0))

            act[(i,j)] = tmp

    # ======================= main ==============================================

    while(1):
        new_res = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                for ((newX, newY), reward) in act[(i,j)]:
                    new_res[i,j] = max(reward + grama * res[newX, newY], new_res[i,j])
    
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
plt.savefig('figure_3_8_a.png')
plt.show()

fig, ax = plt.subplots()
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
ax.table(cellText=policy,loc='center')
plt.plot()
plt.savefig('figure_3_8_b.png')
plt.show()
