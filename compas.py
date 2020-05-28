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

OLD_AGE_COL = 3
YOUNG_AGE_COL = 4
GENDER_COL = 0
RACE_COL = 1
FEMALE = 1
MALE = 0
WHITE = 1
NON_WHITE = 0


MODEL_RACE_BLIND_PATH = 'model_race_blind.pt'
MODEL_PATH = 'model.pt'
# thresholds as a function of protected characteristics
# cost is number of people classified differently based on change to the classifier
# calibration
# equity
# causal fairness paper (maybe ucl), arguing its in mportant to include protected characteristics in classifier
# can compensate for other problems

# backprop cost functions,


def plot_hist(X, Y_hat):
    white_ys = []
    non_white_ys = []
    # bins = np.linspace(0, 1, 15)
    for i in range(len(Y_hat)):
        if (int(X[i, RACE_COL]) == WHITE):
            white_ys.append(Y_hat[i])
        else:
            non_white_ys.append(Y_hat[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, rectangles = ax.hist(
        np.asarray(white_ys),
        10, alpha=0.8, density=True, label='white')
    n, bins, rectangles = ax.hist(
        np.asarray(non_white_ys),
        10, alpha=0.5, density=True, label='non_white')
    # plt.hist(np.asarray(white_ys),bins,alpha=0.5,label='white')
    # plt.hist(np.asarray(non_white_ys),bins,alpha=0.5,label='non_white')
    plt.legend(loc='upper right')
    fig.canvas.draw()
    plt.show()


def fairness_3D_graphs(metric, thresholds, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys = [], []
    for ts in thresholds:
        xs.append(ts[0])
        ys.append(ts[1])
    print("xs"+str(len(xs)))
    print("ys"+str(len(ys)))
    print("zs"+str(len(zs)))
    # xs, ys = np.meshgrid(xs, ys)

    p = ax.scatter(np.asarray(xs), np.asarray(ys), np.log(zs), c=zs, zdir='z')
    # p = ax.bar3d(
    #	 np.asarray(xs),
    #	 np.asarray(ys),
    #	 np.ones_like(zs),
    #	 0.01*np.ones_like(xs),
    #	 0.01*np.ones_like(ys),np.log(zs),shade=True)
    # fig.canvas.draw()
    plt.show()


def get_train_and_test_data():
    Z = np.loadtxt('compas_data.csv', delimiter=',')
    Z = Z[:, 1:]
    np.random.shuffle(Z)
    (rows, cols) = Z.shape
    # print(Z.shape)
    train_rows = round(0.7*rows)
    # decrement value of credit rating
    Z_train = Z[:train_rows, :]
    # print(Z_train.shape)
    Z_test = Z[train_rows:, :]
    # print(Z_test.shape)
    return Z_train, Z_test


def get_predictions(X_train, Y_train, X_test, loading):
    if (loading):
        predictions = network.loadAndGetPredictions('compas_model.pt', X_test)
    else:
        predictions = network.trainAndGetPredictions(
            X_train, Y_train, X_test, save=True, model_path='compas_model.pt')
    return predictions


def print_confusion_matrix(M):
    print("          | PRED: NO | PRED: YES |")
    print("-------------------------------")
    print("ACTL:   NO| "+str(round(M[0, 0], 3)) +
          "	| "+str(round(M[0, 1], 3))+"	 |")
    print("-------------------------------")
    print("ACTL:  YES| "+str(round(M[1, 0], 3)) +
          "	| "+str(round(M[1, 1], 3))+"	 |")
    print("-------------------------------")


def print_matrix(M):
    print("		 | MALE | FEMALE	|")
    print("-------------------------------")
    print("NON-WHITE| "+str(round(M[0, 0], 3)) +
          " | "+str(round(M[0, 1], 3))+"	| ")
    print("-------------------------------")
    print("	WHITE| "+str(round(M[1, 0], 3)) +
          " | "+str(round(M[1, 1], 3))+"	| ")
    print("-------------------------------")


def get_cost_of_switch(Y_hat_0, Y_hat_1, neg_to_pos_cost=1, pos_to_neg_cost=1):
    cost = 0

    for i in range(len(Y_hat_0)):
        if (Y_hat_0[i] == 1 and Y_hat_1[i] == 0):
            cost += pos_to_neg_cost
            # print("cost_pos")
        elif (Y_hat_0[i] == 0 and Y_hat_1[i] == 1):
            cost += neg_to_pos_cost
            # print("cost_neg")
    return cost


def getAccuracies(X, Y_hat, Y, thresholds):
    count_matrix = np.zeros((2, 2))
    true_matrix = np.zeros((2, 2))
    false_matrix = np.zeros((2, 2))
    # positive -> will recid
    # true -> correct prediction
    # Different actors want different minimisations
    # The individual wants to minimise False Positives i.e. Chance of wrongly being predicted to reoffend
    # The system wants to minimise False Negatives i.e. Chance of letting someone go who would reoffend
    true_positive_matrix = np.zeros((2, 2))
    false_positive_matrix = np.zeros((2, 2))

    true_negative_matrix = np.zeros((2, 2))
    false_negative_matrix = np.zeros((2, 2))

    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        y_hat = round(Y_hat[i])
        group = [int(X[i, GENDER_COL]), int(X[i, RACE_COL])]
        race = int(X[i, RACE_COL])
        count_matrix[group[0], group[1]] += 1
        # if correct
        if (y_hat == Y[i]):
            true_matrix[group[0], group[1]] += 1
            # and will recid
            if (y_hat > thresholds[race]):
                true_positive_matrix[group[0], group[1]] += 1
            # and will not recid
            else:
                true_negative_matrix[group[0], group[1]] += 1
        # if incorrect
        else:
            # and will recid
            if (y_hat > thresholds[race]):
                false_positive_matrix[group[0], group[1]] += 1
            # and will not recid
            else:
                false_negative_matrix[group[0], group[1]] += 1
    false_matrix = count_matrix-true_matrix
    print("------------------------------")
    print("accuracy matrix: ")
    print_matrix(true_matrix/count_matrix)
    print("------------------------------")
    # print("TPR matrix - : ")
    # pprint(true_positive_matrix/(true_positive_matrix + false_negative_matrix))
    # print("------------------------------")
    print("FPR matrix - predicted to reoffend, but didn't: ")
    print_matrix(
        false_positive_matrix /
        (true_positive_matrix + false_negative_matrix))
    print("------------------------------")
    print("FNR matrix - predicted NOT to reoffend, but did: ")
    print_matrix(
        false_positive_matrix /
        (false_positive_matrix + true_negative_matrix))
    print("------------------------------")


def get_dataset_base_rates_intersectional(X, Y):
    (rows, cols) = X.shape
    # M  F
    # 0  0 NON-WHITE
    # 0  0 WHITE
    count_matrix = np.zeros((2, 2))
    reoffend_matrix = np.zeros((2, 2))
    recid_rate_matrix = np.zeros((2, 2))

    for i in range(rows):
        if (X[i, GENDER_COL] == MALE and X[i, RACE_COL] == WHITE):
            count_matrix[MALE, WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[MALE, WHITE] += 1
        if (X[i, GENDER_COL] == MALE and X[i, RACE_COL] == NON_WHITE):
            count_matrix[MALE, NON_WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[MALE, NON_WHITE] += 1

        if (X[i, GENDER_COL] == FEMALE and X[i, RACE_COL] == WHITE):
            count_matrix[FEMALE, WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[FEMALE, WHITE] += 1
        if (X[i, GENDER_COL] == FEMALE and X[i, RACE_COL] == NON_WHITE):
            count_matrix[FEMALE, NON_WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[FEMALE, NON_WHITE] += 1
    # pprint(count_matrix)
    # pprint(reoffend_matrix)

    recid_rate_matrix = reoffend_matrix/count_matrix
    # pprint(recid_rate_matrix)
    print("white male recid rate: " +
          str(recid_rate_matrix[MALE, WHITE]))
    print("non-white male recid rate: " +
          str(recid_rate_matrix[MALE, NON_WHITE]))
    print("")
    print("white female data recid rate: " +
          str(recid_rate_matrix[FEMALE, WHITE]))
    print("non-white female data recid rate: " +
          str(recid_rate_matrix[FEMALE, NON_WHITE]))
    print("------------------------------")


def get_weighted_average_base(X, Y):

    (rows, cols) = X.shape
    # M  F
    # 0  0 NON-WHITE
    # 0  0 WHITE
    count_matrix = np.zeros((2, 2))
    reoffend_matrix = np.zeros((2, 2))
    recid_rate_matrix = np.zeros((2, 2))

    for i in range(rows):
        if (X[i, GENDER_COL] == MALE and X[i, RACE_COL] == WHITE):
            count_matrix[MALE, WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[MALE, WHITE] += 1
        if (X[i, GENDER_COL] == MALE and X[i, RACE_COL] == NON_WHITE):
            count_matrix[MALE, NON_WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[MALE, NON_WHITE] += 1

        if (X[i, GENDER_COL] == FEMALE and X[i, RACE_COL] == WHITE):
            count_matrix[FEMALE, WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[FEMALE, WHITE] += 1
        if (X[i, GENDER_COL] == FEMALE and X[i, RACE_COL] == NON_WHITE):
            count_matrix[FEMALE, NON_WHITE] += 1
            if (Y[i] >= 0.5):
                reoffend_matrix[FEMALE, NON_WHITE] += 1

    recid_rate_matrix = reoffend_matrix/count_matrix
    total_count = np.sum(count_matrix)
    weighted_base_rate = (count_matrix[MALE, WHITE]*recid_rate_matrix[MALE, WHITE]
                          + count_matrix[FEMALE, WHITE]*recid_rate_matrix[FEMALE, WHITE]
                          + count_matrix[MALE, NON_WHITE]*recid_rate_matrix[MALE, NON_WHITE]
                          + count_matrix[FEMALE, NON_WHITE]*recid_rate_matrix[FEMALE, NON_WHITE]) / total_count
    print("Weighted Average Base Rate: "+str(weighted_base_rate))
    return weighted_base_rate
