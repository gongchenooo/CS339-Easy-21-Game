from environment import *
import numpy as np
import math
import random

def action_1_value(i, j, value_table, gamma):
    new_value = 0.0
    for red_card in range(1, 11):
        if (j - red_card) < 1:
            new_value -= 1 / 30
        else:
            new_value += gamma * value_table[i, j - red_card] / 30
    for black_card in range(1, 11):
        if (j + black_card) > 21:
            new_value -= 1 / 15
        else:
            new_value += gamma * value_table[i, j + black_card] / 15
    return new_value

def policy_evaluation_2(policy_table, stick_value_table, threshold, gamma):
    # dealercard: 1-10
    # playersum: 1-21
    value_table = np.zeros(shape=[11, 22])
    numOfiterations = 0

    while True:
        delta = 0
        numOfiterations += 1
        for i in range(1, 11):
            for j in range(1, 22):
                action = policy_table[i, j]
                if action == 0:
                    new_value = stick_value_table[i, j]
                else:
                    new_value = action_1_value(i, j, value_table, gamma)
                delta = max(delta, abs(new_value - value_table[i, j]))
                value_table[i, j] = new_value
        print('value iteration %d: %.20f' % (numOfiterations, delta))
        if delta < threshold:
            print('Success!')
            break
    return value_table

def policy_evaluation(policy_table, stick_value_table, threshold, gamma):
    # dealercard: 1-10
    # playersum: 1-21
    value_table = np.zeros(shape=[11, 22])
    numOfiterations = 0
    print(stick_value_table.shape)
    for i in range(1, 11):
        for j in range(1, 22):
            if policy_table[i, j] == 0:
                value_table[i, j] = stick_value_table[1, i, j] - stick_value_table[-1, i, j]  # value = win - lose
    while True:
        delta = 0
        numOfiterations += 1
        for i in range(1, 11):
            for j in range(1, 22):
                action = policy_table[i, j]
                if action == 0:
                    new_value = value_table[i, j]
                else:
                    new_value = action_1_value(i, j, value_table, gamma)
                delta = max(delta, abs(new_value-value_table[i, j]))
                value_table[i, j] = new_value
        print('value iteration %d: %.20f' % (numOfiterations, delta))
        if delta < threshold:
            print('Success!')
            break
    return value_table

def policy_improvement(policy_table, value_table, stick_value_table, gamma):
    flag = True
    for i in range(1, 11):
        for j in range(1, 22):
            value_0 = stick_value_table[1, i, j] - stick_value_table[-1, i, j]
            value_1 = action_1_value(i, j, value_table, gamma)
            if value_0 > value_1:
                optimal_action = 0
            else:
                optimal_action = 1
            if optimal_action != policy_table[i, j]:
                flag = False
            policy_table[i, j] = optimal_action
    return flag
def policy_improvement_2(policy_table, value_table, stick_value_table, gamma):
    flag = True
    for i in range(1, 11):
        for j in range(1, 22):
            value_0 = stick_value_table[i, j]
            value_1 = action_1_value(i, j, value_table, gamma)
            if value_0 > value_1:
                optimal_action = 0
            else:
                optimal_action = 1
            if optimal_action != policy_table[i, j]:
                flag = False
            policy_table[i, j] = optimal_action
    return flag

def policy_iteration():
    policy_table = np.zeros(shape=[11, 22])
    for i in range(1, 11):
        for j in range(1, 22):
            policy_table[i, j] = random.randint(0, 1)
    stick_value_table = np.load('PolicyIterationData/stick_value_table_sampling.npy')
    numOfiteration = 0
    np.save('PolicyIterationData/policy_table_0.npy', policy_table)
    while True:
        numOfiteration += 1
        print('------------------------------------Iteration : %d---------------------------------------' % numOfiteration)
        value_table = policy_evaluation(policy_table=policy_table, stick_value_table=stick_value_table, threshold=1e-20, gamma=1.0)
        flag = policy_improvement(policy_table=policy_table, value_table=value_table, stick_value_table=stick_value_table, gamma=1.0)
        np.save('PolicyIterationData/policy_table_%d.npy' % numOfiteration, policy_table)
        if flag:
            print('Policy Succeeds!')
            break
        print('Policy Fails!')
    # np.save('PolicyIterationData/value_table_2.npy', value_table)
    # np.save('PolicyIterationData/policy_table_2.npy', policy_table)
def show():
    policy_table_1 = np.load('PolicyIterationData/policy_table_1.npy')
    policy_table_2 = np.load('PolicyIterationData/policy_table_2.npy')
    print(policy_table_1)
    print('-'*80)
    print(policy_table_2)

# show()
policy_iteration()
