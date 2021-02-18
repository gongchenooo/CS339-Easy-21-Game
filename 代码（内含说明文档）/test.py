import numpy as np
import random
from environment import *
import csv
def test_q_learning(Q_table_path, numOfgames):
    Q_table = np.load(Q_table_path)
    print(Q_table_path)
    numOfwin = 0
    numOflose = 0
    numOfdraw = 0
    for dealercard_start in range(1, 11):
        for player_start in range(1, 11):
            for i in range(0,(int)(numOfgames/100)):
                state = State()
                state.dealercard = dealercard_start
                state.playersum = player_start
                while state != "terminal":
                    action = np.argmax(Q_table[:, state.dealercard, state.playersum], axis=0)
                    state, reward = step(state, action)
                if reward > 0:
                    numOfwin += 1
                elif reward < 0:
                    numOflose += 1
                else:
                    numOfdraw += 1
    print('numOfgames: %d\tnumOfwin:%d\tnumOflose:%d\tnumOfdraw:%d' % (numOfgames, numOfwin, numOflose, numOfdraw))
    print('win rate: %.4f\twin or draw rate: %.4f' % (numOfwin / numOfgames, (numOfwin+numOfdraw) / numOfgames))
    return numOfgames, numOfwin, numOflose, numOfdraw

def test_random(numOfgames):
    numOfwin = 0
    numOflose = 0
    numOfdraw = 0
    for i in range(0, numOfgames):
        state = State()
        state.dealercard = random.randint(1, 10)
        state.playersum = random.randint(1, 10)
        while state != "terminal":
            action = random.randint(0, 1)
            state, reward = step(state, action)
        if reward > 0:
            numOfwin += 1
        elif reward < 0:
            numOflose += 1
        else:
            numOfdraw += 1
    print(path)
    print('numOfgames: %d\tnumOfwin:%d\tnumOflose:%d\tnumOfdraw:%d' % (numOfgames, numOfwin, numOflose, numOfdraw))
    print('win rate: %.4f\twin or draw rate: %.4f' % (numOfwin / numOfgames, (numOfwin+numOfdraw) / numOfgames))
    return numOfgames, numOfwin, numOflose, numOfdraw

def test_bound(maxSum, numOfgames):
    numOfwin = 0
    numOflose = 0
    numOfdraw = 0
    for i in range(0, numOfgames):
        state = State()
        state.dealercard = random.randint(1, 10)
        state.playersum = random.randint(1, 10)
        while state != "terminal":
            if (state.playersum >= maxSum):
                action = 0
            else:
                action = random.randint(0, 1)
            state, reward = step(state, action)
        if reward > 0:
            numOfwin += 1
        elif reward < 0:
            numOflose += 1
        else:
            numOfdraw += 1
    print('numOfgames: %d\tnumOfwin:%d\tnumOflose:%d\tnumOfdraw:%d' % (numOfgames, numOfwin, numOflose, numOfdraw))
    print('win rate: %.4f\twin or draw rate: %.4f' % (numOfwin / numOfgames, (numOfwin + numOfdraw) / numOfgames))
    return numOfgames, numOfwin, numOflose, numOfdraw

def test_policy_iteration(numOfgames, path):
    print(path)
    numOfwin = 0
    numOflose = 0
    numOfdraw = 0
    policy_table = np.load(path)
    for dealercard_start in range(1, 11):
        for player_start in range(1, 11):
            for i in range(0, (int)(numOfgames / 100)):
                state = State()
                state.dealercard = dealercard_start
                state.playersum = player_start
                while state != "terminal":
                    action = policy_table[state.dealercard][state.playersum]
                    state, reward = step(state, action)
                if reward > 0:
                    numOfwin += 1
                elif reward < 0:
                    numOflose += 1
                else:
                    numOfdraw += 1
    print('numOfgames: %d\tnumOfwin:%d\tnumOflose:%d\tnumOfdraw:%d' % (numOfgames, numOfwin, numOflose, numOfdraw))
    print('win rate: %.4f\twin or draw rate: %.4f' % (numOfwin / numOfgames, (numOfwin + numOfdraw) / numOfgames))
    return numOfgames, numOfwin, numOflose, numOfdraw

def see_Q_table(datapath):
    Q_table = np.load(datapath)
    policy = np.argmax(Q_table[:, :, :], axis=0)
    print(policy)

def main():
    '''
    name_lst = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00']
    for epsilon in name_lst:
        Q_value = np.load('QlearningData/epsilon=%s_alpha=0.010_epoches=1000000.npy' % epsilon)
        policy = np.argmax(Q_value, axis=0)
        value = np.max(Q_value, axis=0)
        print('----------------------epsilon:%s----------------------' % epsilon)
        print(policy)
    '''
    '''
    name_lst = [100, 500, 1000, 2000, 5000, 7000, 9000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
    for epoches in name_lst:
        path = 'QlearningData/epsilon=0.30_alpha=0.010_epoches=%d.npy' % epoches
        test_q_learning(path,100000)
    '''
    '''
    name_lst = ['0.010', '0.005', '0.001', '0.111']
    for epoches in name_lst:
        path = 'QlearningData/epsilon=0.30_alpha=%s_epoches=1000000.npy' % epoches
        test_q_learning(path, 100000)
    '''
    '''
    test_policy_iteration(1000000, 'PolicyIterationData/policy_table_1.npy')
    test_policy_iteration(1000000, 'PolicyIterationData/policy_table_2.npy')
    '''
    '''
    file = open('QlearningData/epsilon变化.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00'])
    epsilon_lst = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00']
    epoches_lst = ['100000', '200000', '300000', '400000', '500000', '600000', '700000', '800000', '900000', '1000000']
    for i in epoches_lst:
        tmp_rate = []
        for j in epsilon_lst:
            numOfgames, numOfwin, _, numOfdraw = test_q_learning(Q_table_path='QlearningData/epsilon=%s_alpha=0.010_epoches=%s.npy' % (j, i), numOfgames=1000000)
            tmp_rate.append((numOfwin+numOfdraw)/numOfgames)
        csv_writer.writerow(tmp_rate)
    file.close()
    '''
    file = open('PolicyIterationData/records.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow([0,1,2,3,4,5])
    iteration_lst = [0,1,2,3,4,5]
    win_or_draw_rate = []
    win_rate = []
    for i in iteration_lst:
        numOfgames, numOfwin, _, numOfdraw = test_policy_iteration(numOfgames=200000,
                                                                   path='PolicyIterationData/policy_table_%d.npy' % i)
        win_rate.append(numOfwin / numOfgames)
        win_or_draw_rate.append((numOfwin + numOfdraw) / numOfgames)
    csv_writer.writerow(win_rate)
    csv_writer.writerow(win_or_draw_rate)
    file.close()
main()
