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

thin = False


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


# Z_train, Z_test = get_train_and_test_data()


# X_train_np = Z_train[:, :-1]
# Y_train_np = Z_train[:, -1]

# X = torch.tensor(X_train_np, dtype=torch.float)
# Y = torch.tensor(Y_train_np, dtype=torch.float)
# (rows, cols) = X.shape


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


# X_test_np = Z_test[:, :-1]
# predictions = network.trainAndGetPredictions(X_train_np, Y_train_np, X_test_np, save=True, model_path='german_model.pt')
# # predictions = network.loadAndGetPredictions('german_model.pt', X_test_np)
# Y_test_np = Z_test[:, -1]

# if thin:
#     thresholds = get_best_pairs(Z_test, np.asarray(predictions), BASE * len(predictions))
# else:
#     thresholds = get_threshold_pairs(Z_test, np.asarray(predictions), BASE * len(predictions), 1.0)

# min = []
# min_ts = []
# zs = []
# currents = []
# for met in Metrics:
#     min.append(10000)
#     min_ts.append([0, 0])
#     zs.append([])
#     currents

# count = 0
# for ts in thresholds:
#     currents = [0]*len(Metrics)
#     # currents[Metrics.dTPR.value], currents[Metrics.dFPR.value] = get_equalised_odds(
#     # Z_test, np.asarray(predictions), Y_test_np, ts)
#     # currents[Metrics.SP.value] = get_statistical_parity(Z_test, np.asarray(predictions), Y_test_np, ts)
#     currents[Metrics.INAC.value] = get_inaccuracy(Z_test, np.asarray(predictions), Y_test_np, ts)

#     for met in Metrics:
#         zs[met.value].append(currents[met.value])

#         if (min[met.value] > currents[met.value]):
#             min[met.value] = currents[met.value]
#             min_ts[met.value] = ts

#     count += 1


# print("count= "+str(count))
# zs_all = []
# for met in Metrics:
#     zs_all.append(zs[met.value])

# print("")
# switch_costs = np.zeros((len(Metrics), len(Metrics)))
# for met_i in Metrics:
#     fairness_2D_graphs(met_i, thresholds, np.asarray(zs_all[met_i.value]))
#     print("min "+met_i.name+" = "+str(min[met_i.value])+" at "+str(min_ts[met_i.value]))
