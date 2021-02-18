import random
import numpy as np
import math
from environment import *
import csv


def q_learning(Q_table, epsilon, alpha, gamma, save_flag=1):
    state = State()
    state.dealercard = random.randint(1, 10)
    state.playersum = random.randint(1, 10)
    while state != "terminal":
        action = None
        if (random.random() < epsilon):  # 小于epsilon采取随机策略
            action = random.randint(0, 1)  # {0: stick, 1: hit}
        else:  # 大于epsilon采取贪心策略
            action = np.argmax(Q_table[:, state.dealercard, state.playersum], axis=0)
        old_dealercard = state.dealercard
        old_playersum = state.playersum
        state, reward = step(state, action)  # take action and observe s', reward
        # Q: Q_table  s: old_dealercard, old_playersum    a: action  s': state    r: reward
        if state != "terminal":
            # Q(s,a) = Q(s,a) + alpha(r+gamma*Q(s',a')-Q(s,a))
            Q_table[action, old_dealercard, old_playersum] = (1 - alpha) * Q_table[
                action, old_dealercard, old_playersum] \
                                                             + alpha * (reward + gamma * np.max(
                Q_table[:, state.dealercard, state.playersum]))
        else:
            Q_table[action, old_dealercard, old_playersum] = (1 - alpha) * Q_table[
                action, old_dealercard, old_playersum] \
                                                             + alpha * reward
    return Q_table


def test_q_learning(Q_table, numOfgames):
    # Q_table = np.load(Q_table_path)
    # print(Q_table_path)

    numOfwin = 0
    numOflose = 0
    numOfdraw = 0
    for dealercard_start in range(1, 11):
        for player_start in range(1, 11):
            for i in range(0, (int)(numOfgames / 100)):
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

    # print('numOfgames: %d\tnumOfwin:%d\tnumOflose:%d\tnumOfdraw:%d' % (numOfgames, numOfwin, numOflose, numOfdraw))
    # print('win rate: %.4f\twin or draw rate: %.4f' % (numOfwin / numOfgames, (numOfwin+numOfdraw) / numOfgames))
    return numOfgames, numOfwin, numOflose, numOfdraw


def main():
    # action: 0 or 1    dealercard: 1 - 10  playersum:  1 - 21
    Q_table = np.zeros(shape=[2, 11, 22])
    max_epoches = 150000
    epsilon = 0
    alpha = 0.01
    gamma = 1
    print("initial:" + "-" * 80)
    file = open('QlearningData2/epsilon=%.3f_alpha=0.010_gamma=%.4f.csv' % (epsilon, gamma), 'w', encoding='utf-8',
                newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['episode', 'wining or draw rate', 'wining rate'])

    for i in range(0, max_epoches + 1):
        if (i % 100 == 0):
            numOfgames, numOfwin, _, numOfdraw = test_q_learning(Q_table, 50000)
            csv_writer.writerow([i, (numOfdraw + numOfwin) / numOfgames, numOfwin / numOfgames])
            print("episode: %d\twin or draw rate: %.4f\t" % (i, (numOfwin + numOfdraw) / numOfgames))
        Q_table = q_learning(Q_table, epsilon=epsilon, alpha=alpha, gamma=gamma)

    file.close()

    '''
    while (alpha < 0.1):
        Q_table = np.zeros(shape=[2, 11, 22])
        alpha *= 10
        print("epsilon: %.4f\talpha:%.4f\tgamma:%.4f\tmax_epoches:%d" % (epsilon, alpha, gamma, max_epoches))
        file = open('QlearningData2/epsilon=%.3f_alpha=%.4f_gamma=%.4f.csv' % (epsilon, alpha, gamma), 'w',
                    encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow(['episode', 'wining or draw rate', 'wining rate'])

        for i in range(0, max_epoches + 1):
            # alpha = 0.01 - (0.01-0.001)/max_epoches * i
            # alpha = 0.01
            # if (i%100000==0):
            #        print('[%d/1000000]:%.4f' % (i, alpha))
            if (i % 100 == 0):
                numOfgames, numOfwin, _, numOfdraw = test_q_learning(Q_table, 50000)
                csv_writer.writerow([i, (numOfdraw + numOfwin) / numOfgames, numOfwin / numOfgames])
                print("episode: %d\twin or draw rate: %.4f\t" % (i, (numOfwin + numOfdraw) / numOfgames))
            Q_table = q_learning(Q_table, epsilon=epsilon, alpha=alpha, gamma=gamma)
        file.close()
    alpha = 0.001
    '''
    '''
    Q_table = np.zeros(shape=[2, 11, 22])
    file = open('QlearningData2/epsilon=%.3f_alpha=0.111_gamma=%.4f.csv' % (epsilon, gamma), 'w', encoding='utf-8',
                newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['episode', 'wining or draw rate', 'wining rate'])

    for i in range(0, max_epoches + 1):
        alpha = 0.01 - (0.01 - 0.001) / max_epoches * i
        if (i%100==0):
            numOfgames, numOfwin, _, numOfdraw = test_q_learning(Q_table, 50000)
            csv_writer.writerow([i, (numOfdraw + numOfwin) / numOfgames, numOfwin / numOfgames])
            print("episode: %d\twin or draw rate: %.4f\t" % (i, (numOfwin + numOfdraw) / numOfgames))
        Q_table = q_learning(Q_table, epsilon=epsilon, alpha=alpha, gamma=gamma)


    
    file.close()
    '''
# alpha = 0.111
# np.save('QlearningData/epsilon=%.2f_alpha=%.3f_epoches=%d.npy' % (epsilon, alpha, max_epoches), Q_table)
'''
for i in range(1, max_epoches+1):
    Q_table = q_learning(Q_table, epsilon=epsilon, alpha=alpha, gamma=1)
    if (i%100000==0 or i==100 or i==500 or i==1000 or i==2000 or i==5000 or i==7000 or i==9000):
        np.save('QlearningData/epsilon=%.2f_alpha=%.3f_epoches=%d.npy' % (epsilon, alpha, i), Q_table)
        print('num of epoches: [%d]' % i)
'''
'''
    win_or_draw_rate = test_q_learning(Q_table, 1000000)
    if (win_or_draw_rate > max_win_or_draw_rate):
        max_win_or_draw_rate = win_or_draw_rate
        print('win_or_draw_rate: %.4f' % max_win_or_draw_rate)
        np.save('Q_table_epsilon=0.1_alpha=0.001_maxwinordraw.npy', Q_table)
    # print(Q_table)
'''
main()
