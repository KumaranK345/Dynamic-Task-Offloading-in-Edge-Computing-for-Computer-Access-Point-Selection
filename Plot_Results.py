import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def plot_results():
    Vary_No_of_Statesize = [10, 20, 30, 40, 50]
    Algorithms = ['ESO-ADRL', 'WaOA-ADRL', 'LO-ADRL', 'PFOA-ADRL', 'F-PFOA-ADRL']
    Statistic_Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    # BestFit = np.load('BestFit.npy', allow_pickle=True)
    Fitness = np.load('Fitness1.npy', allow_pickle=True)
    Maximum_Rewards = np.load('Rewards.npy', allow_pickle=True)

    for n in range(len(Vary_No_of_Statesize)):
        # Bestfit = BestFit[n]
        fitness = Fitness[n]
        Rewards = Maximum_Rewards[n]

        Statistical = np.zeros((fitness.shape[0], 5))
        for i in range(fitness.shape[0]):
            Statistical[i, 0] = np.min(fitness[i, :])
            Statistical[i, 1] = np.max(fitness[i, :])
            Statistical[i, 2] = np.mean(fitness[i, :])
            Statistical[i, 3] = np.median(fitness[i, :])
            Statistical[i, 4] = np.std(fitness[i, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Algorithms)
        for j in range(len(Statistic_Terms)):
            Table.add_column(Statistic_Terms[j], Statistical[:, j])
        print('-------------------------------------------------- Number of States - ', str(Vary_No_of_Statesize[n]),
              ' - Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        X = np.arange(fitness.shape[1])
        plt.plot(X, fitness[0, :], color='r', linewidth=3, markerfacecolor='blue', markersize=12,
                 label="ESO-ADRL")
        plt.plot(X, fitness[1, :], color='g', linewidth=3, markerfacecolor='red', markersize=12,
                 label="WaOA-ADRL")
        plt.plot(X, fitness[2, :], color='b', linewidth=3, markerfacecolor='green', markersize=12,
                 label="LO-ADRL")
        plt.plot(X, fitness[3, :], color='m', linewidth=3, markerfacecolor='yellow', markersize=12,
                 label="PFOA-ADRL")
        plt.plot(X, fitness[4, :] - 0.00009, color='k', linewidth=3, markerfacecolor='cyan', markersize=12,
                 label="F-PFOA-ADRL")
        plt.xlabel('NO. of Iteration')
        plt.ylabel('Cost Function')
        plt.ylim([np.min(fitness) - 0.0001, np.max(fitness) + 0.0005])
        plt.legend(loc=1)
        path1 = "./Results/conv_%s.png" % (n + 1)
        plt.savefig(path1)
        plt.show()

        X = np.arange(fitness.shape[1])
        plt.plot(X, Rewards[0, :], color='r', linewidth=3, markerfacecolor='r', markersize=16, label="ESO-ADRL")
        plt.plot(X, Rewards[1, :], color='g', linewidth=3, markerfacecolor='g', markersize=16, label="WaOA-ADRL")
        plt.plot(X, Rewards[2, :], color='b', linewidth=3, markerfacecolor='b', markersize=16, label="LO-ADRL")
        plt.plot(X, Rewards[3, :], color='m', linewidth=3, markerfacecolor='m', markersize=16, label="PFOA-ADRL")
        plt.plot(X, Rewards[4, :], color='c', linewidth=3, markerfacecolor='c', markersize=16, label="F-PFOA-ADRL")
        plt.xlabel('NO. of Iteration')
        plt.ylabel('Rewards')

        plt.legend(loc=4)
        path1 = "./Results/reward_%s_Alg.png" % (n + 1)
        plt.savefig(path1)
        plt.show()

        X = np.arange(fitness.shape[1])
        plt.plot(X, Rewards[5, :], color='r', linewidth=3, markerfacecolor='r', markersize=16,
                 label="DWQAOA")
        plt.plot(X, Rewards[6, :], color='g', linewidth=3, markerfacecolor='g', markersize=16,
                 label="MTFNN")
        plt.plot(X, Rewards[7, :], color='b', linewidth=3, markerfacecolor='b', markersize=16,
                 label="ADMM")
        plt.plot(X, Rewards[8, :], color='m', linewidth=3, markerfacecolor='m', markersize=16,
                 label="CSAO")
        plt.plot(X, Rewards[4, :], color='c', linewidth=3, markerfacecolor='c', markersize=16,
                 label="F-PFOA-ADRL")
        plt.xlabel('NO. of Iteration')
        plt.ylabel('Rewards')

        plt.legend(loc=4)
        path1 = "./Results/reward_%s_Med.png" % (n + 1)
        plt.savefig(path1)
        plt.show()




####### Total Consumption time
def plot_Throuput():
    for a in range(1):
        Eval = np.load('Throuput.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 0], color='b', linewidth=3, marker='d', markerfacecolor='b', markersize=10,
                 label="ESO-ADRL")
        plt.plot(learnper, Eval[:, 1], color='r', linewidth=3, marker='d', markerfacecolor='r', markersize=10,
                 label="WaOA-ADRL")
        plt.plot(learnper, Eval[:, 2], color='g', linewidth=3, marker='d', markerfacecolor='g', markersize=10,
                 label="LO-ADRL")
        plt.plot(learnper, Eval[:, 3], color='m', linewidth=3, marker='d', markerfacecolor='m', markersize=10,
                 label="PFOA-ADRL")
        plt.plot(learnper, Eval[:, 4], color='k', linewidth=3, marker='d', markerfacecolor='k', markersize=10,
                 label="F-PFOA-ADRL")
        plt.xticks(X + 1, ('1', '2', '3', '4', '5'))

        plt.xlabel('Configurations')
        plt.ylabel('Throughput')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=4)
        path1 = "./Results/Dataset1_Throughput_Algorithm_.png"
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='b', width=0.10, label="DWQAOA")
        ax.bar(X + 0.10, Eval[:, 6], color='r', width=0.10, label="MTFNN")
        ax.bar(X + 0.20, Eval[:, 7], color='g', width=0.10, label="ADMM")
        ax.bar(X + 0.30, Eval[:, 8], color='m', width=0.10, label="CSAO")
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, label="F-PFOA-ADRL")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('Configurations')
        plt.ylabel('Throughput')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=1)
        path1 = "./Results/Dataset1_Throughput_med.png"
        plt.savefig(path1)
        plt.show()


def plot_Security():
    for a in range(1):
        Eval = np.load('Security.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 0], color='b', linewidth=3, marker='d', markerfacecolor='b', markersize=10,
                 label="ESO-ADRL")
        plt.plot(learnper, Eval[:, 1], color='r', linewidth=3, marker='d', markerfacecolor='r', markersize=10,
                 label="WaOA-ADRL")
        plt.plot(learnper, Eval[:, 2], color='g', linewidth=3, marker='d', markerfacecolor='g', markersize=10,
                 label="LO-ADRL")
        plt.plot(learnper, Eval[:, 3], color='m', linewidth=3, marker='d', markerfacecolor='m', markersize=10,
                 label="PFOA-ADRL")
        plt.plot(learnper, Eval[:, 4], color='k', linewidth=3, marker='d', markerfacecolor='k', markersize=10,
                 label="F-PFOA-ADRL")
        plt.xticks(X + 1, ('1', '2', '3', '4', '5'))

        plt.xlabel('Configurations')
        plt.ylabel('Security')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=4)
        path1 = "./Results/Dataset1_Security_Algorithm_.png"
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='b', width=0.10, label="DWQAOA")
        ax.bar(X + 0.10, Eval[:, 6], color='r', width=0.10, label="MTFNN")
        ax.bar(X + 0.20, Eval[:, 7], color='g', width=0.10, label="ADMM")
        ax.bar(X + 0.30, Eval[:, 8], color='m', width=0.10, label="CSAO")
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, label="F-PFOA-ADRL")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('Configurations')
        plt.ylabel('Security')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=1)
        path1 = "./Results/Dataset1_Security_med.png"
        plt.savefig(path1)
        plt.show()


def plot_Resource_utilization():
    for a in range(1):
        Eval = np.load('Resource utilization.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 0], color='b', linewidth=3, marker='d', markerfacecolor='b', markersize=10,
                 label="ESO-ADRL")
        plt.plot(learnper, Eval[:, 1], color='r', linewidth=3, marker='d', markerfacecolor='r', markersize=10,
                 label="WaOA-ADRL")
        plt.plot(learnper, Eval[:, 2], color='g', linewidth=3, marker='d', markerfacecolor='g', markersize=10,
                 label="LO-ADRL")
        plt.plot(learnper, Eval[:, 3], color='m', linewidth=3, marker='d', markerfacecolor='m', markersize=10,
                 label="PFOA-ADRL")
        plt.plot(learnper, Eval[:, 4], color='k', linewidth=3, marker='d', markerfacecolor='k', markersize=10,
                 label="F-PFOA-ADRL")
        plt.xticks(X + 1, ('1', '2', '3', '4', '5'))

        plt.xlabel('Configurations')
        plt.ylabel('Resource utilization')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=4)
        path1 = "./Results/Dataset1_Resource utilization_Algorithm_.png"
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='b', width=0.10, label="DWQAOA")
        ax.bar(X + 0.10, Eval[:, 6], color='r', width=0.10, label="MTFNN")
        ax.bar(X + 0.20, Eval[:, 7], color='g', width=0.10, label="ADMM")
        ax.bar(X + 0.30, Eval[:, 8], color='m', width=0.10, label="CSAO")
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, label="F-PFOA-ADRL")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('Configurations')
        plt.ylabel('Resource utilization')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=1)
        path1 = "./Results/Dataset1_Resource utilization_med.png"
        plt.savefig(path1)
        plt.show()


def plot_Energy_Consumption():
    for a in range(1):
        Eval = np.load('Energy Consumption.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 0], color='b', linewidth=3, marker='d', markerfacecolor='b', markersize=10,
                 label="ESO-ADRL")
        plt.plot(learnper, Eval[:, 1], color='r', linewidth=3, marker='d', markerfacecolor='r', markersize=10,
                 label="WaOA-ADRL")
        plt.plot(learnper, Eval[:, 2], color='g', linewidth=3, marker='d', markerfacecolor='g', markersize=10,
                 label="LO-ADRL")
        plt.plot(learnper, Eval[:, 3], color='m', linewidth=3, marker='d', markerfacecolor='m', markersize=10,
                 label="PFOA-ADRL")
        plt.plot(learnper, Eval[:, 4], color='k', linewidth=3, marker='d', markerfacecolor='k', markersize=10,
                 label="F-PFOA-ADRL")
        plt.xticks(X + 1, ('1', '2', '3', '4', '5'))

        plt.xlabel('Configurations')
        plt.ylabel('Energy Consumption')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=4)
        path1 = "./Results/Dataset1_Energy Consumption_Algorithm_.png"
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='b', width=0.10, label="DWQAOA")
        ax.bar(X + 0.10, Eval[:, 6], color='r', width=0.10, label="MTFNN")
        ax.bar(X + 0.20, Eval[:, 7], color='g', width=0.10, label="ADMM")
        ax.bar(X + 0.30, Eval[:, 8], color='m', width=0.10, label="CSAO")
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, label="F-PFOA-ADRL")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('Configurations')
        plt.ylabel('Energy Consumption')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=1)
        path1 = "./Results/Dataset1_Energy Consumption_med.png"
        plt.savefig(path1)
        plt.show()


def plot_cost():
    for a in range(1):
        Eval = np.load('cost.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        X = np.arange(5)
        plt.plot(learnper, Eval[:, 0], color='b', linewidth=3, marker='d', markerfacecolor='b', markersize=10,
                 label="ESO-ADRL")
        plt.plot(learnper, Eval[:, 1], color='r', linewidth=3, marker='d', markerfacecolor='r', markersize=10,
                 label="WaOA-ADRL")
        plt.plot(learnper, Eval[:, 2], color='g', linewidth=3, marker='d', markerfacecolor='g', markersize=10,
                 label="LO-ADRL")
        plt.plot(learnper, Eval[:, 3], color='m', linewidth=3, marker='d', markerfacecolor='m', markersize=10,
                 label="PFOA-ADRL")
        plt.plot(learnper, Eval[:, 4], color='k', linewidth=3, marker='d', markerfacecolor='k', markersize=10,
                 label="F-PFOA-ADRL")
        plt.xticks(X + 1, ('1', '2', '3', '4', '5'))

        plt.xlabel('Configurations')
        plt.ylabel('cost')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=4)
        path1 = "./Results/Dataset1_cost_Algorithm_.png"
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Eval[:, 5], color='b', width=0.10, label="DWQAOA")
        ax.bar(X + 0.10, Eval[:, 6], color='r', width=0.10, label="MTFNN")
        ax.bar(X + 0.20, Eval[:, 7], color='g', width=0.10, label="ADMM")
        ax.bar(X + 0.30, Eval[:, 8], color='m', width=0.10, label="CSAO")
        ax.bar(X + 0.40, Eval[:, 9], color='k', width=0.10, label="F-PFOA-ADRL")
        plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
        plt.xlabel('Configurations')
        plt.ylabel('cost')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        # plt.legend(loc=1)
        path1 = "./Results/Dataset1_cost_med.png"
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    plot_results()
    plot_Throuput()
    plot_Security()
    plot_Resource_utilization()
    plot_Energy_Consumption()
    plot_cost()
