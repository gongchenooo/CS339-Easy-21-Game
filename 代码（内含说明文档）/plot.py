import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''
r1 = np.load("Qube/ABC/reward-hyper-example.npy")
r2 = np.load("Qube/Cannon/reward_cannon.npy")
r3 = np.load("Qube/Random/rewardslist_ran.npy")

a = []
b = []
c = []


for i in range(50):
    a.append(np.average(r1[2*i:2*(i+1)]))
    b.append(np.average(r2[4 * i:4 * (i + 1)]))
    c.append(np.average(r3[4 * i:4 * (i + 1)]))


fig = plt.figure(figsize=(8, 5))
plt.title("Traning Reward with Different Optimization Method")
plt.xlabel("Process")
plt.ylabel("Training Reward")

plt.plot(a, label="ABC")
plt.plot(c, color='green', label="Random")
plt.plot(b, color='red', label="Cannon")
legend = plt.legend(loc=1)
plt.savefig("figure/reward_qube.png")
plt.show()
'''
a = pd.read_csv('QlearningData2/epsilon=0.300_alpha=0.0010_gamma=1.0000.csv')
a = a['wining or draw rate'].values

b = pd.read_csv('QlearningData2/epsilon=0.300_alpha=0.0100_gamma=1.0000.csv')
b = b['wining or draw rate'].values

c = pd.read_csv('QlearningData2/epsilon=0.300_alpha=0.1000_gamma=1.0000.csv')
c = c['wining or draw rate'].values

d = pd.read_csv('QlearningData2/epsilon=0.300_alpha=0.111_gamma=1.0000.csv')
d = d['wining or draw rate'].values



r1 = []
r2 = []
r3 = []
r4 = []

for i in range(375):
    r1.append(np.average(a[4 * i:4 * (i + 1)]))
    r2.append(np.average(b[4 * i:4 * (i + 1)]))
    r3.append(np.average(c[4 * i:4 * (i + 1)]))
    r4.append(np.average(d[4 * i:4 * (i + 1)]))

'''
for i in range(300):
    r1.append(np.average(a[5 * i:5 * (i + 1)]))
    r2.append(np.average(b[5 * i:5 * (i + 1)]))
    r3.append(np.average(c[5 * i:5 * (i + 1)]))
    r4.append(np.average(d[5 * i:5 * (i + 1)]))
'''
fig = plt.figure(figsize=(8, 5))
plt.title("Winning Rate with Different Learning Rate")
plt.xlabel("episode number(×400)")
plt.ylabel("winning rate")
plt.plot(r4, color='yellow', label="alpha=0.001")
plt.plot(r1, color='blue', label="alpha=0.010")

plt.plot(r3, color='red', label="alpha=0.100")
plt.plot(r2, color='green', label="alpha=0.010 - 0.001")
legend = plt.legend(loc=4)
plt.savefig("figure/Winning Rate with Different Learning Rate2.png")
plt.show()
