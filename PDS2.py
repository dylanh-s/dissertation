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
    # LIPS = 3
    INAC = 3


class Sum(enum.Enum):
    SUM = -1


thin = True
loading_model = True
SEX_COL = 1
RACE_COL = 1
PROTECTED_COL = SEX_COL
dataset = 'compas'
use_train_data_in_post = True


def get_base_rate(X, Y):
    pos = 0
    for i in range(len(Y)):
        if (Y[i] == 1.0):
            pos += 1
    return pos/len(Y)


def prediction_hist(predictions):
    return


def plot_colourline(x, y, c):
    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], c=c[i])
    return


def cost_curves(metric, min_ts, thresholds, cost, recids=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axes = plt.gca()
    # axes.set_xlim([0, 1])
    ax.set_title('cost of switching from optimal ' + str(metric.name) + ' at ' + str(min_ts))
    ax.set_xlabel('Threshold pair index')
    ax.set_ylabel('Cost')
    x = np.arange(len(thresholds))
    # print(len(x))
    # print(x)
    y = np.asarray(cost)
    if (not recids == -1):
        # plot_colourline(x, y, recids)
        p = plt.scatter(x, y, c=recids)
        cbar = plt.colorbar(p)
    else:
        plt.plot(x, y)
    plt.savefig('figs/'+dataset+'/'+metric.name+'_cost_curve.png')
    plt.show()


def thin_graphs(thresholds):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs, ys = [], []
    for ts in thresholds:
        xs.append(ts[0])
        ys.append(ts[1])
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    ax.set_xlabel('G_0 threshold')
    ax.set_ylabel('G_1 threshold')
    ax.set_title('threshold pairs which best maintain system utility')
    p = ax.scatter(np.asarray(xs), np.asarray(ys),  s=6, marker='s')
    # fig.colorbar(p)
    plt.savefig('figs/'+dataset+'/first_sweep_best_threshold_pairs.png')
    plt.show()


def switch_sum_2D_graphs(thresholds, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs, ys = [], []
    for ts in thresholds:
        xs.append(ts[0])
        ys.append(ts[1])
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    p = ax.scatter(np.asarray(xs), np.asarray(ys), c=zs, cmap=cm.jet, s=5, marker='s')
    ax.set_title('summed cost of switching from optimal configurations')
    ax.set_xlabel('G_0 threshold')
    ax.set_ylabel('G_1 threshold')
    cbar = fig.colorbar(p)
    cbar.set_label('cost')
    # plt.show()
    if (thin):
        s = 'THIN'
    else:
        s = ''
    plt.savefig('figs/'+dataset+'/summed_cost_of_switch'+s+'.png')


def cost_of_switch_2D_graphs(metric, min_ts, thresholds, zs, thin=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs, ys = [], []
    for ts in thresholds:
        xs.append(ts[0])
        ys.append(ts[1])
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    p = ax.scatter(np.asarray(xs), np.asarray(ys), c=zs, cmap=cm.jet, s=5, marker='s')
    ax.set_title('cost of switching from optimal ' + str(metric.name) + ' at ' + str(min_ts))
    ax.set_xlabel('G_0 threshold')
    ax.set_ylabel('G_1 threshold')
    cbar = fig.colorbar(p)
    cbar.set_label('cost')

    # plt.show()
    plt.savefig('figs/'+dataset+'/'+metric.name+'_'+'cost_of_switch'+'.png')


def print_confusion_matrix(M):
    print("          | PRED: NO | PRED: YES |")
    print("-------------------------------")
    print("ACTL:   NO| "+str(round(M[0, 0], 3)) +
          "	| "+str(round(M[0, 1], 3))+"	 |")
    print("-------------------------------")
    print("ACTL:  YES| "+str(round(M[1, 0], 3)) +
          "	| "+str(round(M[1, 1], 3))+"	 |")
    print("-------------------------------")


def get_all_pairs(precision=100):
    li = []
    for big_i in range(precision):
        i = big_i/precision
        for big_j in range(precision):
            j = big_j/precision
            li.append([i, j])
    return li


def argsort(l):
    return sorted(range(len(l)), key=lambda x: (x[0], -x[1]))


def NEW_get_best_pairs(X, Y_hat, target, protected_attribute_index, precision=100):
    all_vals = [(x*(1/precision)) for x in range(0, precision)]
    all_vals_rev = list(reversed(all_vals))
    unpaired_xs = [(x*(1/precision)) for x in range(0, precision)]
    unpaired_ys = [(x*(1/precision)) for x in range(0, precision)]
    li = []

    for x in all_vals:
        best = 1000000000
        best_y = -1
        best_recids = 0
        for y in all_vals:
            thresholds = [round(x, 2), round(y, 2)]
            recids = 0

            for i in range(len(Y_hat)):
                # prediction 1 = recid, 0 = no recid
                protected_attribute = int(X[i, protected_attribute_index])
                if (Y_hat[i] > thresholds[protected_attribute]):
                    recids += 1
            if(abs(recids - target) <= best):
                best = abs(recids-target)
                best_y = y
                best_recids = recids
        # print("removing " + str(best_y))
        if (best_y in unpaired_ys):
            unpaired_ys.remove(best_y)
        li.append([x, best_y, best_recids])
        # print(best_recids)
        # recids_li.append(best_recids)

    for y in unpaired_ys:
        best = 1000000
        best_x = -1
        best_recids = 0
        for x in all_vals:
            thresholds = [round(x, 2), round(y, 2)]
            recids = 0
            for i in range(len(Y_hat)):
                # prediction 1 = recid, 0 = no recid
                protected_attribute = int(X[i, protected_attribute_index])
                if (Y_hat[i] > thresholds[protected_attribute]):
                    recids += 1
            if(abs(recids - target) <= best):
                best = abs(recids-target)
                best_x = x
                best_recids = recids
        li.append([best_x, y, best_recids])

    sorted_li = sorted(li, key=lambda x: (x[0], -x[1]))
    # def recid_li(l): return [sl.pop(-1) for sl in l]
    # recids = recid_li(sorted_li)

    recids_li = [sl.pop(-1) for sl in sorted_li]

    return sorted_li, recids_li


def get_threshold_pairs(X, Y_hat, target, protected_attribute_index, max_relaxation_of_target=0.1, precision=100):
    li = []
    recids = 0
    recs = []
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
                recs.append(recids)
    return li, recs


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
    if (use_train_data_in_post):
        Z = np.concatenate((Z_train, Z_test))
        X = np.concatenate((X_train, X_test))
        Y = np.concatenate((Y_train, Y_test))
    else:
        Z = Z_test
        X = X_test
        Y = Y_test
    predictions = german.get_predictions(X_train, Y_train, X, True)
    BASE = get_base_rate(Z_train, Y_train)
elif (dataset == 'compas'):
    Z_train, Z_test = compas.get_train_and_test_data()

    X_train = Z_train[:, :-1]
    Y_train = Z_train[:, -1]

    X_test = Z_test[:, :-1]
    Y_test = Z_test[:, -1]
    if (use_train_data_in_post):
        Z = np.concatenate((Z_train, Z_test))
        X = np.concatenate((X_train, X_test))
        Y = np.concatenate((Y_train, Y_test))
    else:
        Z = Z_test
        X = X_test
        Y = Y_test
    predictions = compas.get_predictions(X_train, Y_train, X, loading_model)
    BASE = get_base_rate(Z_train, Y_train)
else:
    sys.exit("No dataset found for "+sys.argv[1])

pprint(Z.shape)
if (thin):
    thresholds, recids_li = NEW_get_best_pairs(
        Z, np.asarray(predictions),
        BASE * len(predictions),
        PROTECTED_COL, precision=25)
else:
    thresholds = get_threshold_pairs(Z, np.asarray(predictions), BASE * len(predictions), PROTECTED_COL)

# pprint(thresholds)
# thin_graphs(thresholds)
all_thresholds = get_all_pairs(precision=25)

min = []
min_ts = []
zs = []
currents = []
for met in Metrics:
    min.append(10000)
    min_ts.append([0, 0])
    zs.append([])

count = 0
# find optimal threshold pair for each metric
for ts in thresholds:
    currents = [0]*len(Metrics)
    currents[Metrics.dTPR.value], currents[Metrics.dFPR.value] = fairness_metrics.get_equalised_odds(
        Z, np.asarray(predictions), Y, PROTECTED_COL, ts)

    currents[Metrics.SP.value] = fairness_metrics.get_statistical_parity(
        Z, np.asarray(predictions), Y, PROTECTED_COL, ts)

    # currents[Metrics.LIPS.value] = fairness_metrics.get_individual_fairness(
    #     Z, np.asarray(predictions), PROTECTED_COL, ts)

    currents[Metrics.INAC.value] = fairness_metrics.get_inaccuracy(Z, np.asarray(predictions), Y, PROTECTED_COL, ts)

    # currents[Metrics.LIPS.value] = fairness_metrics.get_individual_fairness(
    #     X, np.asarray(predictions), PROTECTED_COL, ts)

    for met in Metrics:
        zs[met.value].append(currents[met.value])

        if (min[met.value] > currents[met.value]):
            min[met.value] = currents[met.value]
            min_ts[met.value] = ts

    count += 1

# find costs of switching from optimal
zs_all = [0]*len(Metrics)
for met_i in Metrics:
    zs = []
    outcomes_i = fairness_metrics.probability_to_outcome(
        Z, np.asarray(predictions),
        PROTECTED_COL, min_ts[met_i.value])
    for ts in thresholds:
        outcomes_j = fairness_metrics.probability_to_outcome(
            Z, np.asarray(predictions),
            PROTECTED_COL, ts)
        zs.append(fairness_metrics.get_cost_of_switch(
            outcomes_i, outcomes_j, neg_to_pos_cost=0.0, pos_to_neg_cost=1.0))
    # cost_of_switch_2D_graphs(met_i, min_ts[met_i.value], thresholds, zs)
    zs_all[met_i.value] = np.asarray(zs)
    cost_curves(met_i, min_ts[met_i.value], thresholds, np.asarray(zs)/np.asarray(recids_li), recids_li)
    # zs_all[met_i.value] = np.reshape(np.asarray(zs), (100, 100))

# pprint(zs_all)
cost_sum = np.zeros_like(zs_all[0])
for met_i in Metrics:
    cost_sum = np.add(cost_sum, zs_all[met_i.value])
min_index = cost_sum.argmin()
# print(min_index)
print("min at " + str(thresholds[min_index]))
cost_curves(Sum.SUM, thresholds[min_index], thresholds, cost_sum)
