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
import graphing

from torch.utils.data import Dataset, DataLoader
from matplotlib import cm


class Metrics(enum.Enum):
    SP = 0
    dTPR = 1
    dFPR = 2
    pSUFF = 3
    nSUFF = 4
    # INAC = 5


thin = True
loading_model = True
SEX_COL = 1
RACE_COL = 1
PROTECTED_COL = SEX_COL
dataset = 'compas'

use_train_data_in_post = True

P2N = 1.0
N2P = 0.0
precision = 100


s = ''
if (P2N < N2P):
    s = 'P2N_LT_N2P'
elif (P2N == N2P):
    s = 'P2N_EQ_N2P'
else:
    s = 'P2N_GT_N2P'


def get_base_rate(X, Y):
    pos = 0
    for i in range(len(Y)):
        if (Y[i] == 1.0):
            pos += 1
    return pos/len(Y)


def get_group_base_rate(X, Y, protected_attribute_index):
    pos = [0, 0]
    count = [0, 0]
    for i in range(len(Y)):
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y[i] == 1.0):
            pos[protected_attribute] += 1
        count[protected_attribute] += 1
    print("g0 base rate:", pos[0]/count[0])
    print("g1 base rate:", pos[1]/count[1])


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


def get_best_pairs(X, Y_hat, target, protected_attribute_index, precision=100):
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
    if (sys.argv[1] == "interactive"):
        dataset = input("Dataset:")
        P2N = float(input("P2N:"))
        N2P = float(input("N2P:"))
        # precision = int(input("Precision:"))
        target = int(input("target:"))
    else:
        dataset = sys.argv[1]


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
    # predictions = np.ones(Y.shape)*BASE
    # predictions = Y
else:
    sys.exit("No dataset found for "+sys.argv[1])
graphing.set_dataset(dataset)
pprint(Z.shape)
BASE = get_base_rate(Z, Y)

# generate best thresholds for target
thresholds, recids_li = get_best_pairs(
    Z, np.asarray(predictions),
    BASE * len(predictions),
    PROTECTED_COL, precision=precision)

graphing.prediction_hist(Z, PROTECTED_COL, predictions, s)
graphing.thin_graphs(thresholds, recids_li, s)

# generate list of all possible thresholds
all_thresholds = get_all_pairs(precision=precision)

min = []
min_ts = []
zs = []
currents = []
for met in Metrics:
    min.append(10000000)
    min_ts.append([0, 0])
    zs.append([])

count = 0
# find optimal threshold pair for each metric
for ts in thresholds:
    currents = [0]*len(Metrics)

    currents[Metrics.SP.value], currents[Metrics.dTPR.value], currents[Metrics.dFPR.value], currents[Metrics.pSUFF.
                                                                                                     value], currents[Metrics.nSUFF.value] = fairness_metrics.speedy_metrics(Z, np.asarray(predictions),
                                                                                                                                                                             Y, PROTECTED_COL, ts)
    # currents[Metrics.INAC.value] = fairness_metrics.get_inaccuracy(Z, np.asarray(predictions), Y, PROTECTED_COL, ts)

    for met in Metrics:
        zs[met.value].append(currents[met.value])

        if (min[met.value] > currents[met.value]):
            min[met.value] = currents[met.value]
            min_ts[met.value] = ts

    count += 1


# find costs of switching from optimal configuration for each metric
switch_costs_all = [0]*len(Metrics)
for met_i in Metrics:
    switch_costs = []
    outcomes_i = fairness_metrics.probability_to_outcome(
        Z, np.asarray(predictions),
        PROTECTED_COL, min_ts[met_i.value])
    for ts in thresholds:
        outcomes_j = fairness_metrics.probability_to_outcome(
            Z, np.asarray(predictions),
            PROTECTED_COL, ts)
        switch_costs.append(fairness_metrics.get_cost_of_switch(
            outcomes_i, outcomes_j, neg_to_pos_cost=N2P, pos_to_neg_cost=P2N))

    switch_costs_all[met_i.value] = np.asarray(switch_costs)
    graphing.cost_of_switch_2D_graphs(met_i, min_ts[met_i.value], thresholds, switch_costs, s)
    graphing.cost_curves(met_i.name, min_ts[met_i.value], thresholds, np.asarray(switch_costs), len(predictions), s)
    graphing.metric_fulfillment_curves(met_i, min_ts[met_i.value], thresholds, zs[met_i.value], s)


cost_sum = np.zeros_like(switch_costs_all[0])
for met_i in Metrics:
    cost_sum = np.add(cost_sum, switch_costs_all[met_i.value])
min_index = cost_sum.argmin()

print()
print("cost-optimal pair is at " + str(thresholds[min_index]))

graphing.cost_curves("Sum", thresholds[min_index], thresholds, cost_sum, len(predictions), s, -1)
graphing.value_breakdown_curve(Metrics, zs, thresholds, min_index, s)
graphing.cost_breakdown_curve(Metrics, switch_costs_all, thresholds, min_index, s)

for met_i in Metrics:
    print(str(met_i.name)+" here is " + str(zs[met_i.value][min_index]))
    print("min possible "+met_i.name+" is "+str(min[met_i.value]))
    print()

fairest_outcomes = fairness_metrics.probability_to_outcome(
    Z, np.asarray(predictions), PROTECTED_COL, thresholds[min_index])

probabilistic_outcomes = fairness_metrics.probability_to_outcome(
    Z, np.asarray(predictions), PROTECTED_COL, [0.5, 0.5])

print("positive instances here = ", np.sum(fairest_outcomes))
graphing.outcome_hists(Z, Y, fairest_outcomes, probabilistic_outcomes, PROTECTED_COL, s)
graphing.prediction_hist(Z, PROTECTED_COL, predictions, thresholds[min_index])
graphing.ungrouped_prediction_hist(Z, PROTECTED_COL, predictions, thresholds[min_index], s)
