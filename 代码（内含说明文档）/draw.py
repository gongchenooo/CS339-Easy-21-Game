import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


sns.set()
def plot_action(table):

    your = [i for i in range(1,22)]
    dealer = [j for j in range(1,11)]


    fig, ax = plt.subplots()
    im = ax.imshow(table)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(your)))
    ax.set_yticks(np.arange(len(dealer)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(your)
    ax.set_yticklabels(dealer)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.xlabel('Player Sum')
    plt.ylabel('Dealer Card')
    # Loop over data dimensions and create text annotations.
    #for i in range(len(dealer)):
        #for j in range(len(your)):
            #text = ax.text(j, i, table[i, j],ha="center", va="center", color="w")

    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_title("Policy of Each State through Sampling")
    fig.tight_layout()
    plt.show()
    plt.savefig("Qlearning_Policy_of_each_state.png")

def plot(Q):
    from mpl_toolkits.mplot3d import Axes3D

    pRange = list(range(1, 22))
    dRange = list(range(1, 11))
    vStar = list()
    for p in pRange:
        for d in dRange:
            vStar.append([d, p, Q[d, p]])

    df = pd.DataFrame(vStar, columns=['dealer', 'player', 'value'])

    # And transform the old column name in something numeric
    # df['player']=pd.Categorical(df['player'])
    # df['player']=df['player'].cat.codes

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    plt.title('Value of Each State through Sampling')
    plt.xlabel("Dealer Card")
    plt.ylabel("Player Current Sum")
    plt.show()

    # to Add a color bar which maps values to colors.
    surf = ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.viridis, linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()

    # Rotate it
    ax.view_init(30, 60)
    plt.show()

    # Other palette
    #ax.plot_trisurf(df['dealer'], df['player'], df['value'], cmap=plt.cm.jet, linewidth=0.01)
    #plt.show()
'''
Q_Markov = np.load('PolicyIterationData/stick_value_table_Markov.npy')
Q_Markov = Q_Markov[0:11, 0:22]
Q_sampling = np.load('PolicyIterationData/stick_value_table_sampling.npy')
Q_sampling = Q_sampling[1, :, :] - Q_sampling[-1, :, :]
Q_sampling = Q_sampling[0:11, 0:22]
#Q = Q_Markov - Q_sampling
'''
Q = np.load('PolicyIterationData/value_table_Sampling.npy')
# Q_0 = Q[0, :, :]
# Q_1 = Q[1, :, :]
# Q_2 = np.zeros(shape=[11, 22])
# for i in range(11):
    # for j in range(22):
        # Q_2[i,j] = np.argmax(Q[:, i, j])
# Q_2 = Q_2[1:11, 1:22]
plot(Q)
#print(Q)
# plot(Q_2)