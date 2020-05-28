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
from torch.utils.data import Dataset, DataLoader
from matplotlib import cm


class Classifier(nn.Module):
    def __init__(self, cols):
        super(Classifier, self).__init__()

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
        # Apply sigmoid to output.
        pred = torch.sigmoid(self.forward(x))
        # print(pred)
        return pred


def getClassifier(input_size):
    model = Classifier(input_size)
    return model


def trainClassifier(X, Y, model):
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01)
    # Number of epochs
    epochs = 4000
    losses = []
    Y = Y.unsqueeze(1)
    for i in range(epochs):
        # Precit the output for Given input
        y_pred = model.forward(X)
        # Compute Cross entropy loss
        loss = criterion(y_pred, Y)
        # Add loss to the list
        losses.append(loss.item())
        # Clear the previous gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Adjust weights
        optimizer.step()
        if (i % 1000 == 0):
            print(loss)
            print(str(i/epochs * 100) + "%")
        # torch.save(model.state_dict(), MODEL_PATH)
    return model

    # model.load_state_dict(torch.load(MODEL_PATH))
    # model.eval()


def getPredictions(X, model):
    (inputs, _) = X.shape
    predictions = []
    for i in range(0, inputs):
        X_pred = torch.tensor(X[i, :], dtype=torch.float)
        prediction = model.predict(X_pred)
        y_hat = prediction.item()
        predictions.append(y_hat)
    return predictions


def trainAndGetPredictions(X_train, Y_train, X_test, save=False, model_path='last_model.pt'):
    (rows, cols) = X_train.shape
    model = getClassifier(cols)
    model = trainClassifier(X_train, Y_train, model)
    if(save):
        torch.save(model.state_dict(), model_path)
    return getPredictions(X_test, model)


def loadAndGetPredictions(model_path, X_test):
    (rows, cols) = X_test.shape
    model = getClassifier(cols)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return getPredictions(X_test, model)
