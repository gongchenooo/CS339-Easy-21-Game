import numpy as np
from environment import *


def calculate_by_sampling(numOfgames):
    stick_value_table = np.zeros(shape=[3, 11, 22],dtype=float) # dealercard: 1 - 10   playersum: 1 - 21    0:draw   1:win  2:lose
    for i in range(1, 11): # dealercard: 1 - 10
        print('dealercard: %d' % i)
        for j in range(1, 22): # playersum: 1 - 21
            for game in range(numOfgames):
                state = State()
                state.dealercard = i
                state.playersum = j
                _, reward = step(state=state, action=0)
                stick_value_table[int(reward), i, j] += 1
    stick_value_table = stick_value_table / numOfgames
    np.save('PolicyIterationData/stick_value_table_sampling.npy', stick_value_table)

def calculate_by_Markov(max_iteration=2500):
    stick_value_table = np.zeros(shape=[22, 22]) # 前者代表庄家sum，后者代表玩家playersum
    for i in range(16, 22):
        for j in range(1, 22):
            if (i > j):
                stick_value_table[i][j] = -1
            elif (i < j):
                stick_value_table[i][j] = 1
            else:
                stick_value_table[i][j] = 0
    numOfiteration = 0
    while True:
        numOfiteration += 1
        for i in range(1, 16): # 庄家sum从1到15需要迭代计算
            for j in range(1, 22): # 玩家sum从1到21需要迭代计算
                new_value = 0.0
                for black_card in range(1, 11):
                    if (i + black_card > 21):
                        new_value += 1 / 15
                    else:
                        new_value += stick_value_table[i+black_card][j] / 15
                for red_card in range(1, 11):
                    if (i - red_card < 1):
                        new_value += 1 / 30
                    else:
                        new_value += stick_value_table[i-red_card][j] / 15
                stick_value_table[i][j] = new_value
        if (numOfiteration%500==0):
            print("numOfiteration:%d" % numOfiteration)
        if numOfiteration > max_iteration:
            print("stick_value_table has been calculated successfully!")
            break
    np.save("PolicyIterationData/stick_value_table_Markov.npy", stick_value_table)

def show():
    stick_value_table_sampling = np.load('PolicyIterationData/stick_value_table_sampling.npy')
    stick_value_table_Markov = np.load('PolicyIterationData/stick_value_table_Markov.npy')
    print(stick_value_table_sampling.shape)
    print(stick_value_table_sampling[1]-stick_value_table_sampling[-1])
    print('-'*80)
    print(stick_value_table_Markov.shape)
    print(stick_value_table_Markov[:11, :])

calculate_by_sampling(100000)
calculate_by_Markov(10000)
show()
