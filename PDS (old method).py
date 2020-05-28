from pprint import pprint
import numpy as np
import pandas as pd
import torch
import os
import io
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import enum
import network
import fairness_metrics
import german
import compas
import sys

from torch.utils.data import Dataset, DataLoader
from matplotlib import cm


class Metrics(enum.Enum):
    dTPR = 0
    dFPR = 1
    SP = 2
    LIPS = 3
    INAC = 4


thin = True
loading_model = True
SEX_COL = 1
RACE_COL = 1
PROTECTED_COL = SEX_COL
dataset = 'compas'


def get_base_rate(X, Y):
    pos = 0
    for i in range(len(Y)):
        if (Y[i] == 1.0):
            pos += 1
    return pos/len(Y)


def fairness_2D_graphs(metric, thresholds, protected_attribute_index, zs, thin=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs, ys = [], []
    for ts in thresholds:
        xs.append(ts[0])
        ys.append(ts[1])
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    p = ax.scatter(np.asarray(xs), np.asarray(ys), c=np.log(zs), cmap=cm.jet, s=5, marker='s')
    ax.set_title(str(metric.name))
    ax.set_xlabel('G_0 threshold')
    ax.set_ylabel('G_1 threshold')
    cbar = fig.colorbar(p)
    cbar.set_label('log('+metric.name+')')

    plt.show()
    if (thin):
        s = 'thin'
    else:
        s = 'fat'
    # plt.savefig('figs/'+dataset+'_'+metric.name+'_'+s+'.png')


def print_confusion_matrix(M):
    print("          | PRED: NO | PRED: YES |")
    print("-------------------------------")
    print("ACTL:   NO| "+str(round(M[0, 0], 3)) +
          "	| "+str(round(M[0, 1], 3))+"	 |")
    print("-------------------------------")
    print("ACTL:  YES| "+str(round(M[1, 0], 3)) +
          "	| "+str(round(M[1, 1], 3))+"	 |")
    print("-------------------------------")


def get_best_pairs(X, Y_hat, target, protected_attribute_index, precision=100):
    li = []
    paired_ys = []
    recids = 0
    for big_i in range(precision):
        i = big_i/precision
        best = 1000000
        best_j = -1

        for big_j in range(precision):
            j = big_j/precision
            thresholds = [i, j]
            recids = 0

            for x in range(len(Y_hat)):
                # prediction 1 = recid, 0 = no recid
                protected_attribute = int(X[x, protected_attribute_index])
                if (Y_hat[x] > thresholds[protected_attribute]):
                    recids += 1
            # if number predicted to recid is approx equal to target
            if(abs(recids - target) < best):
                # print(str(i)+","+str(j)+" gives "+str(recids)+" positives")
                best = abs(recids-target)
                best_j = j
        li.append([i, best_j])
        # print(str(i)+" , "+str(best_j))
        paired_ys.append(best_j)
    # print(len(li))

    # go through again filling the ys with no corresponding xs
    for big_y in range(precision):
        y = big_y/100
        if (y not in paired_ys):
            count = 0
            best = 1000000
            best_j = -1
            for big_j in range(precision):
                j = big_j/precision
                thresholds = [j, y]
                recids = 0

                for x in range(len(Y_hat)):
                    # prediction 1 = recid, 0 = no recid
                    protected_attribute = int(X[x, protected_attribute_index])
                    if (Y_hat[x] > thresholds[protected_attribute]):
                        recids += 1
                # if number predicted to recid is approx equal to target
                if(abs(recids - target) < best):
                    # print(str(i)+","+str(x)+" gives "+str(recids)+" positives")
                    best = abs(recids-target)
                    best_j = j
            # print(str(best_j)+" , "+str(y))
            li.append([best_j, y])

    return li


def get_threshold_pairs(X, Y_hat, target, protected_attribute_index, max_relaxation_of_target=0.1, precision=100):
    li = []
    recids = 0
    # epsilon gives percentage distance we are willing to deviate from our target number of positives in order to consider a threshold pair
    epsilon = max_relaxation_of_target
    for big_i in range(precision):
        i = big_i/precision
        for big_j in range(precision):
            j = big_j/precision
            thresholds = [i, j]
            recids = 0
            for x in range(len(Y_hat)):
                # prediction 1 = recid, 0 = no recid
                protected_attribute = int(X[x, protected_attribute_index])
                if (Y_hat[x] > thresholds[protected_attribute]):
                    recids += 1
            # if number predicted to recid is approx equal to target
            if(abs(recids - target) < epsilon*target):
                # print(str(i)+","+str(j)+" gives "+str(recids)+" positives")
                li.append(thresholds)
    # print(len(li))
    return li


if(1 > 2):
    print('aa')


if (len(sys.argv) > 1):
    dataset = sys.argv[1]
    if (len(sys.argv) > 2):
        if (sys.argv[2] == 'thin'):
            thin = True
        elif (sys.argv[2] == 'fat'):
            thin = False

if (dataset == 'german'):
    Z_train, Z_test = german.get_train_and_test_data()
    # Z_train, Z_test = german.get_train_and_test_data()

    X_train = Z_train[:, :-1]
    Y_train = Z_train[:, -1]

    X_test = Z_test[:, :-1]
    Y_test = Z_test[:, -1]
    predictions = german.get_predictions(X_train, Y_train, X_test, True)
    BASE = get_base_rate(Z_train, Y_train)
elif (dataset == 'compas'):
    Z_train, Z_test = compas.get_train_and_test_data()

    X_train = Z_train[:, :-1]
    Y_train = Z_train[:, -1]

    X_test = Z_test[:, :-1]
    Y_test = Z_test[:, -1]
    predictions = compas.get_predictions(X_train, Y_train, X_test, loading_model)
    BASE = get_base_rate(Z_train, Y_train)
else:
    sys.exit("No dataset found for "+sys.argv[1])

if thin:
    thresholds = get_best_pairs(Z_test, np.asarray(predictions), BASE * len(predictions), PROTECTED_COL)
else:
    thresholds = get_threshold_pairs(Z_test, np.asarray(predictions), BASE * len(predictions), PROTECTED_COL)

min = []
min_ts = []
zs = []
currents = []
for met in Metrics:
    min.append(10000)
    min_ts.append([0, 0])
    zs.append([])
    currents

count = 0
for ts in thresholds:
    currents = [0]*len(Metrics)
    currents[Metrics.dTPR.value], currents[Metrics.dFPR.value] = fairness_metrics.get_equalised_odds(
        Z_test, np.asarray(predictions), Y_test, PROTECTED_COL, ts)

    currents[Metrics.SP.value] = fairness_metrics.get_statistical_parity(
        Z_test, np.asarray(predictions), Y_test, PROTECTED_COL, ts)
    currents[Metrics.INAC.value] = fairness_metrics.get_inaccuracy(
        Z_test, np.asarray(predictions), Y_test, PROTECTED_COL, ts)

    currents[Metrics.LIPS.value] = fairness_metrics.get_individual_fairness(
        X_test, np.asarray(predictions), PROTECTED_COL, ts)

    for met in Metrics:
        zs[met.value].append(currents[met.value])

        if (min[met.value] > currents[met.value]):
            min[met.value] = currents[met.value]
            min_ts[met.value] = ts

    count += 1

print("count= "+str(count))
zs_all = []
for met in Metrics:
    zs_all.append(zs[met.value])


print("")
switch_costs = np.zeros((len(Metrics), len(Metrics)))
for met_i in Metrics:

    fairness_2D_graphs(met_i, thresholds, PROTECTED_COL, np.asarray(zs_all[met_i.value]))

    print("min "+met_i.name+" = "+str(min[met_i.value])+" at "+str(min_ts[met_i.value]))
    print("accuracy here is "+str((1-fairness_metrics.get_inaccuracy(Z_test,
                                                                     np.asarray(predictions), Y_test, PROTECTED_COL, min_ts[met_i.value])) * 100)+"%")
    cms = fairness_metrics.get_confusion_matrices(
        Z_test, np.asarray(predictions),
        Y_test, PROTECTED_COL, min_ts[met_i.value])
    print("non_white")
    print_confusion_matrix(cms[0])
    print("white")
    print_confusion_matrix(cms[1])
    print("")
    print("")
    met_val = met_i.value
    for met_j in Metrics:
        # if (met_j.value > met_i.value):
        outcomes_i = fairness_metrics.probability_to_outcome(
            Z_test, np.asarray(predictions),
            PROTECTED_COL, min_ts[met_i.value])
        outcomes_j = fairness_metrics.probability_to_outcome(
            Z_test, np.asarray(predictions),
            PROTECTED_COL, min_ts[met_j.value])
        switch_costs[met_i.value, met_j.value] = fairness_metrics.get_cost_of_switch(outcomes_i, outcomes_j)
pprint(switch_costs)
costs = np.sum(switch_costs, 1)
print(costs)
optimal = np.argmin(costs)
print("best metric for cost minimisation is " + Metrics(optimal).name)
