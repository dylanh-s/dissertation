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

dataset = 'compas'


def set_dataset(d):
    global dataset
    dataset = d


def prediction_hist(Z, protected_index, predictions, ts=False):
    predictions_0 = []
    predictions_1 = []
    for i in range(len(predictions)):
        protected_attribute = int(Z[i, protected_index])
        if (protected_attribute == 1):
            predictions_1.append(predictions[i])
        else:
            predictions_0.append(predictions[i])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('prediction')
    ax.set_ylabel('quantity')
    cols = ['blue', 'orange']
    plt.hist(np.asarray(predictions_0), bins=20, range=(0, 1), alpha=0.7, label='non_white_predictions', color=cols[0])
    plt.hist(np.asarray(predictions_1), bins=20, range=(0, 1), alpha=0.7, label='white_predictions', color=cols[1])
    if (not ts == False):
        plt.axvline(x=ts[0], linestyle='--', c=cols[0], label='G_0 threshold')
        plt.axvline(x=ts[1], linestyle='--', c=cols[1], label='G_1 threshold')
    plt.legend()
    plt.savefig('figs/'+dataset+'/'+dataset+'_NN_predictions_hist.pdf')
    plt.close()


def ungrouped_prediction_hist(Z, protected_index, predictions, s, ts=False):
    predictions

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('prediction')
    ax.set_ylabel('quantity')
    cols = ['blue', 'orange']
    plt.hist(np.asarray(predictions), bins=20, range=(0, 1), alpha=1.0, label='predictions', color=cols[0])
    if (not ts == False):
        plt.axvline(x=ts[0], linestyle='--', c=cols[0], label='G_0 threshold')
        plt.axvline(x=ts[1], linestyle='--', c=cols[1], label='G_1 threshold')
    plt.legend()
    plt.savefig('figs/'+dataset+'/'+dataset+'_ungrouped_NN_predictions_hist.pdf')
    plt.close()


def outcome_hists(Z, Y, Y_hat_fair, Y_hat_probabilistic, protected_index, s):
    ground_truth_positives = [0, 0]
    fair_predicted_positives = [0, 0]
    probabilistic_predicted_positives = [0, 0]
    width = 0.35
    x = np.arange(3)

    favoured_count = np.sum(Z[:, protected_index])
    unfavoured_count = len(Y)-favoured_count
    for i in range(len(Y)):
        protected_attribute = int(Z[i, protected_index])
        ground_truth_positives[protected_attribute] += Y[i]
        fair_predicted_positives[protected_attribute] += Y_hat_fair[i]
        probabilistic_predicted_positives[protected_attribute] += Y_hat_probabilistic[i]

    fig, ax = plt.subplots()
    ax.set_ylim([0, 2000])
    cols = ['blue', 'orange']

    p1 = ax.bar(
        x, [ground_truth_positives[0],
            fair_predicted_positives[0],
            probabilistic_predicted_positives[0]],
        width, color=cols[0],
        label='African-American')
    p2 = ax.bar(
        x + width, [ground_truth_positives[1],
                    fair_predicted_positives[1],
                    probabilistic_predicted_positives[1]],
        width, color=cols[1],
        label='White')
    ax.set_title(dataset+' ground truths vs predictions')
    ax.set_ylabel('Number of positive instances')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(('ground truth', 'LGFO', 'uncorrected'))
    plt.legend()
    plt.savefig('figs/'+dataset+'/ground_truth_vs_pred.pdf')

    # plt.show()


def plot_colourline(x, y, c):
    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))

    ax = plt.gca()

    for i in np.arange(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], c=c[i])
    return


def plot_cost_comparison_curves(met_i, cost_i, met_j, cost_j, thresholds, data_size, s):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axes = plt.gca()
    ax.set_title(dataset+' cost of '+met_i.name+' versus cost of ' + met_j.name)
    axes.set_ylim([0, data_size])
    ax.set_xlabel('Threshold pair index')
    ax.set_ylabel('Cost')
    x = np.arange(len(thresholds))
    y_i = np.asarray(cost_i)
    y_j = np.asarray(cost_j)
    plt.plot(x, y_i, c='green', label=met_i.name)
    plt.plot(x, y_j, c='red', label=met_j.name)
    plt.legend()
    plt.savefig('figs/'+dataset+'/'+met_i.name+'_versus_'+met_j.name+'_cost.pdf')
    plt.close()


def plot_value_comparison_curves(met_i, val_i, met_j, val_j, thresholds, data_size, s):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axes = plt.gca()
    ax.set_title(dataset+' value of '+met_i.name+' versus value of ' + met_j.name)
    axes.set_ylim([0, 1.01])
    ax.set_xlabel('Threshold pair index')
    ax.set_ylabel('Value')
    x = np.arange(len(thresholds))
    y_i = np.asarray(val_i)
    y_j = np.asarray(val_j)
    plt.plot(x, y_i, c='green', label=met_i.name)
    plt.plot(x, y_j, c='red', label=met_j.name)
    plt.legend()
    plt.savefig('figs/'+dataset+'/'+met_i.name+'_versus_'+met_j.name+'_value.pdf')
    plt.close()


def value_breakdown_curve(mets, zs, thresholds, best_pair_index, s, cost_sums=False):
    line_styles = ['-', '-', '-', '-.']*(round(len(mets) / 4)+1)

    fig, ax1 = plt.subplots()
    x = np.arange(len(thresholds))
    ax1.set_title('Metric values at each threshold pair')

    if (not cost_sums):

        ax1.set_xlabel('Threshold pair index')
        ax1.set_ylabel('Values')
        ax1.set_ylim([0, 1.01])
        plt.axvline(x=best_pair_index, linestyle='--', c='black', label='optimal threshold pair')
        for met_i in mets:
            ax1.plot(x, zs[met_i.value], label=met_i.name, linestyle=line_styles[met_i.value])

    else:

        ax1.set_xlabel('Threshold pair index')
        ax1.set_ylabel('Cost sum')
        c = 'red'
        ax1.plot(x, cost_sums, label='SUM', c=c)
        ax1.tick_params(axis='y', labelcolor=c)
        plt.axvline(x=best_pair_index, linestyle='--', c='black', label='optimal threshold pair')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Values')
        ax2.set_ylim([0, 1.01])
        for met_i in mets:
            ax2.plot(x, zs[met_i.value], label=met_i.name, linestyle=line_styles[met_i.value])
            # ax2.tick_params(axis='y')

    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('figs/'+dataset+'/Metric_values_at_each_threshold_pair_' + s+'.pdf')
    plt.close()


def cost_breakdown_curve(mets, switch_costs, thresholds, best_pair_index, s, cost_sums=False):

    line_styles = ['-', '-', '-', '--']*(round(len(mets) / 4)+1)

    fig, ax1 = plt.subplots()
    ax1.set_title('Metric costs at each threshold pair')
    if (not cost_sums):

        x = np.arange(len(thresholds))
        ax1.set_xlabel('Threshold pair index')
        ax1.set_ylabel('Cost of switch from optimal')
        plt.axvline(x=best_pair_index, linestyle='--', c='black', label='optimal threshold pair')
        for met_i in mets:
            ax1.plot(x, switch_costs[met_i.value], label=met_i.name, linestyle=line_styles[met_i.value])

    else:
        x = np.arange(len(thresholds))
        ax1.set_xlabel('Threshold pair index')
        ax1.set_ylabel('Cost sum')
        c = 'red'
        ax1.plot(x, cost_sums, label='SUM', c=c)
        ax1.tick_params(axis='y', labelcolor=c)
        plt.axvline(x=best_pair_index, linestyle='--', c='black', label='optimal threshold pair')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.set_ylabel('Cost of switch from optimal')
        for met_i in mets:
            ax2.plot(x, switch_costs[met_i.value], label=met_i.name, linestyle=line_styles[met_i.value])

    plt.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('figs/'+dataset+'/Metric_costs_at_each_threshold_pair_' + s+'.pdf')
    plt.close()


def compare_metrics(met_i, met_j, Z, zs, thresholds, predictions, min_ts, PROTECTED_COL, N2P, P2N, s):

    switch_costs_i = []
    switch_costs_j = []

    outcomes_i = fairness_metrics.probability_to_outcome(Z, np.asarray(predictions), PROTECTED_COL, min_ts[met_i.value])
    for ts in thresholds:
        outcomes_j = fairness_metrics.probability_to_outcome(Z, np.asarray(predictions), PROTECTED_COL, ts)
        switch_costs_i.append(fairness_metrics.get_cost_of_switch(
            outcomes_i, outcomes_j, neg_to_pos_cost=N2P, pos_to_neg_cost=P2N))

    outcomes_i = fairness_metrics.probability_to_outcome(Z, np.asarray(predictions), PROTECTED_COL, min_ts[met_j.value])
    for ts in thresholds:
        outcomes_j = fairness_metrics.probability_to_outcome(Z, np.asarray(predictions), PROTECTED_COL, ts)
        switch_costs_j.append(fairness_metrics.get_cost_of_switch(
            outcomes_i, outcomes_j, neg_to_pos_cost=N2P, pos_to_neg_cost=P2N))

    plot_cost_comparison_curves(met_i, switch_costs_i, met_j, switch_costs_j, thresholds, len(predictions), s)
    plot_value_comparison_curves(met_i, zs[met_i.value], met_j, zs[met_j.value], thresholds, len(predictions), s)
    sum_costs = np.add(np.asarray(switch_costs_i), np.asarray(switch_costs_j))
    cost_curves(met_i.name + " summed with "+met_j.name,
                thresholds[sum_costs.argmin()], thresholds, sum_costs, len(predictions), s)
    print(met_i.name+" at summed minimum is "+str(zs[met_i.value][sum_costs.argmin()]))
    print(met_j.name+" at summed minimum is "+str(zs[met_j.value][sum_costs.argmin()]))


def metric_fulfillment_curves(metric, min_ts, thresholds, metric_values, s):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axes = plt.gca()
    ax.set_title(dataset+' value of '+metric.name+' for each configuration')
    ax.set_xlabel('Threshold pair index')
    ax.set_ylabel(metric.name)
    axes.set_ylim([0, 1.01])
    x = np.arange(len(thresholds))
    y = np.asarray(metric_values)
    plt.plot(x, y, label=metric.name)
    plt.legend()
    plt.savefig('figs/'+dataset+'/'+metric.name+'_values.pdf')
    plt.close()


def cost_curves(metric_name, min_ts, thresholds, cost, data_size, s, recids=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axes = plt.gca()
    if (metric_name == "Sum"):
        ax.set_title('Summed costs')
    else:
        # ax.set_title(dataset+' cost of switching from optimal ' + str(metric_name) + ' at ' + str(min_ts))
        ax.set_title(metric_name)
        axes.set_ylim([0, data_size])
    ax.set_xlabel('Threshold pair index')
    ax.set_ylabel('Cost')
    x = np.arange(len(thresholds))
    # print(len(x))
    # print(x)
    y = np.asarray(cost)
    if (not recids == -1):
        # plot_colourline(x, y, recids)
        p = plt.scatter(x, y, c=recids)
        # cbar = plt.colorbar(p)
    else:
        plt.plot(x, y)
    plt.savefig('figs/'+dataset+'/'+metric_name+'_cost_curve_'+s+'.pdf')
    plt.close()


def thin_graphs(thresholds, zs, s):
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
    ax.set_title(dataset+' threshold pairs which best maintain system utility')
    p = ax.scatter(np.asarray(xs), np.asarray(ys), c=zs, s=6, marker='s')
    cbar = fig.colorbar(p)
    cbar.set_label('positive classifications')
    plt.savefig('figs/'+dataset+'/best_threshold_pairs.pdf')
    plt.close()


def metric_fulfillment_graphs(thresholds, zs, metric, s):
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
    ax.set_title(dataset+str(metric.name)+' threshold pairs')
    p = ax.scatter(np.asarray(xs), np.asarray(ys), c=zs, s=6, marker='s')
    cbar = fig.colorbar(p)
    cbar.set_label(str(metric.name))
    plt.savefig('figs/'+dataset+'/'+str(metric.name)+'_metric_fulfillment.pdf')
    plt.close()


def switch_sum_2D_graphs(thresholds, zs, s):
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
    ax.set_title(dataset+' summed cost of switching from optimal configurations')
    ax.set_xlabel('G_0 threshold')
    ax.set_ylabel('G_1 threshold')
    cbar = fig.colorbar(p)
    cbar.set_label('cost')
    # plt.show()
    if (thin):
        s = 'THIN'
    else:
        s = ''
    plt.savefig('figs/'+dataset+'/summed_cost_of_switch'+s+'.pdf')
    plt.close()


def cost_of_switch_2D_graphs(metric, min_ts, thresholds, zs, s, thin=True):
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
    ax.set_title(dataset+' cost of switching from optimal ' + str(metric.name) + ' at ' + str(min_ts))
    ax.set_xlabel('G_0 threshold')
    ax.set_ylabel('G_1 threshold')
    cbar = fig.colorbar(p)
    cbar.set_label('cost')

    plt.savefig('figs/'+dataset+'/'+metric.name+'_'+'cost_of_switch'+'.pdf')
    plt.close()
