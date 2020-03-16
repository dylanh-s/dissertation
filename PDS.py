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
from torch.utils.data import Dataset, DataLoader
from matplotlib import cm


class Metrics(enum.Enum):
    dTPR = 0
    dFPR = 1
    SP = 2
    INAC = 3


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


def get_best_pairs(X, Y_hat, target, precision=100):
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


if thin:
    thresholds = get_best_pairs(Z_test, np.asarray(predictions), BASE * len(predictions))
else:
    thresholds = get_threshold_pairs(Z_test, np.asarray(predictions), BASE * len(predictions), 1.0)

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
    # currents[Metrics.dTPR.value], currents[Metrics.dFPR.value] = get_equalised_odds(
    # Z_test, np.asarray(predictions), Y_test_np, ts)
    # currents[Metrics.SP.value] = get_statistical_parity(Z_test, np.asarray(predictions), Y_test_np, ts)
    currents[Metrics.INAC.value] = fairness-metrics.get_inaccuracy(Z_test, np.asarray(predictions), Y_test_np, ts)

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
    fairness_2D_graphs(met_i, thresholds, np.asarray(zs_all[met_i.value]))
    print("min "+met_i.name+" = "+str(min[met_i.value])+" at "+str(min_ts[met_i.value]))
