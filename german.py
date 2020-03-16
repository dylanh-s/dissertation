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
from torch.utils.data import Dataset, DataLoader
from matplotlib import cm


MODEL_PATH = "german_model.pt"
AGE_COL = 0
SEX_COL = 1
JOB_COL = 2
HOUSING_COL = 3
gender_blind = False


class Metrics(enum.Enum):
    INAC = 0
    # dTPR = 0
    # dFPR = 1
    # SP = 2


thin = False


def fairness_2D_graphs(metric, thresholds, zs, thin=True):
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
    # plt.savefig('figs/'+metric.name+'_'+s+'.png')


def get_weighted_average_base(X, Y):
    pos = 0
    for i in range(len(Y)):
        if (Y[i] == 1.0):
            pos += 1
    return pos/len(Y)


def get_train_and_test_data():
    Z = np.loadtxt('german_dataset.csv', delimiter=',')
    np.random.shuffle(Z)
    (rows, cols) = Z.shape
    train_rows = round(0.7*rows)
    # decrement value of credit rating
    Z[:, -1] = Z[:, -1]-1
    Z_train = Z[:train_rows, :]
    Z_test = Z[train_rows:, :]
    return Z_train, Z_test


def get_predictions(X_train, Y_train, X_test, loading):
    if (loading):
        predictions = network.loadAndGetPredictions('german_model.pt', X_test)
    else:
        predictions = network.trainAndGetPredictions(
            X_train, Y_train, X_test, save=True, model_path='german_model.pt')
    return predictions


def get_predictions(X_train, Y_train, X_test, saving):

    predictions = network.trainAndGetPredictions(
        X_train, Y_train, X_test, save=saving, model_path='german_model.pt')
    return predictions


Z_train, Z_test = get_train_and_test_data()


X_train_np = Z_train[:, :-1]
Y_train_np = Z_train[:, -1]

X = torch.tensor(X_train_np, dtype=torch.float)
Y = torch.tensor(Y_train_np, dtype=torch.float)
(rows, cols) = X.shape


def get_inaccuracy(X, Y_hat, Y, thresholds):
    wrong = 0
    count = len(Y_hat)
    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        y_hat = round(Y_hat[i])
        sex = int(X[i, SEX_COL])
        if (Y_hat[i] > thresholds[sex]):
            y_hat = 1
        else:
            y_hat = 0
        # if correct
        if (y_hat != Y[i]):
            wrong += 1
    return wrong/count


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
                sex = int(X[x, SEX_COL])
                if (Y_hat[x] > thresholds[sex]):
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
                    sex = int(X[x, SEX_COL])
                    if (Y_hat[x] > thresholds[sex]):
                        recids += 1
                # if number predicted to recid is approx equal to target
                if(abs(recids - target) < best):
                    # print(str(i)+","+str(x)+" gives "+str(recids)+" positives")
                    best = abs(recids-target)
                    best_j = j
            # print(str(best_j)+" , "+str(y))
            li.append([best_j, y])

    return li


def get_threshold_pairs(X, Y_hat, target, max_relaxation_of_target=0.1, precision=100):
    li = []
    recids = 0
    # epsilon gives percentage distance we are willing to deviate from our target number of arrests in order to consider a threshold pair
    epsilon = max_relaxation_of_target
    for big_i in range(precision):
        i = big_i/precision
        for big_j in range(precision):
            j = big_j/precision
            thresholds = [i, j]
            recids = 0
            for x in range(len(Y_hat)):
                # prediction 1 = recid, 0 = no recid
                sex = int(X[x, SEX_COL])
                if (Y_hat[x] > thresholds[sex]):
                    recids += 1
            # if number predicted to recid is approx equal to target
            if(abs(recids - target) < epsilon*target):
                li.append(thresholds)

    return li


class MyClassifier(nn.Module):
    def __init__(self, cols):
        super(MyClassifier, self).__init__()

        self.fc1 = nn.Linear(cols, 20)

        # This applies linear transformation to produce output data
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 1)

    def forward(self, x):

        # Output of the first layer
        x = self.fc1(x)

        # Activation function
        x = torch.sigmoid(x)
        # layer 2
        x = self.fc2(x)
        # layer 3
        x = self.fc3(x)
        # layer 4
        x = self.fc4(x)
        # output
        x = self.fc5(x)
        return x

    # predicts the class (0 == low recid or 1 == high recid)
    def predict(self, x):
        # Apply softmax to output.
        pred = torch.sigmoid(self.forward(x))
        # print(pred)
        return pred


X_test_np = Z_test[:, :-1]
predictions = network.trainAndGetPredictions(X_train_np, Y_train_np, X_test_np, save=True, model_path='german_model.pt')
# predictions = network.loadAndGetPredictions('german_model.pt', X_test_np)
Y_test_np = Z_test[:, -1]

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
    currents[Metrics.INAC.value] = get_inaccuracy(Z_test, np.asarray(predictions), Y_test_np, ts)

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
