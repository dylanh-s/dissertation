import numpy as np

FAVOURED = 1
UNFAVOURED = 0


def probability_to_outcome(X, Y_hat, protected_attribute_index, thresholds):
    outcomes = np.zeros_like(Y_hat)
    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat = 1
        else:
            y_hat = 0
        outcomes[i] = y_hat
    return outcomes


def get_inaccuracy(X, Y_hat, Y, protected_attribute_index, thresholds):
    wrong = 0
    count = len(Y_hat)
    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        # prediction 1 = bad_credit, 0 = good_credit
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat = 1
        else:
            y_hat = 0
        # if correct
        if (y_hat != Y[i]):
            wrong += 1
    return wrong/count


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


def get_confusion_matrices(X, Y_hat, Y, protected_attribute_index, thresholds):
    confusion_matrices = [
        np.zeros((2, 2)),
        np.zeros((2, 2))]

    counts = np.zeros(2)
    trues = np.zeros(2)
    falses = np.zeros(2)
    true_positives = np.zeros(2)
    true_negatives = np.zeros(2)
    false_positives = np.zeros(2)
    false_negatives = np.zeros(2)
    # positive -> will recid
    # true -> correct prediction
    # Different actors want different minimisations
    # The individual wants to minimise False Positives i.e. Chance of wrongly being predicted to reoffend
    # The system wants to minimise False Negatives i.e. Chance of letting someone go who would reoffend
    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        prot_attr = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[prot_attr]):
            y_hat = 1
        else:
            y_hat = 0
        counts[prot_attr] += 1
        # if correct
        if (y_hat == Y[i]):
            trues[prot_attr] += 1
            # and will recid
            if (y_hat == 1):
                true_positives[prot_attr] += 1
            # and will not recid
            else:
                true_negatives[prot_attr] += 1
        # if incorrect
        else:
            # and will recid
            if (y_hat == 1):
                false_positives[prot_attr] += 1
            # and will not recid
            else:
                false_negatives[prot_attr] += 1
    falses = counts-trues
    for i in range(2):
        prot_attr = i
        # print("------------------------------")
        # print("RACE = " + str(i))
        # print("COUNT = " + str(counts[prot_attr]))
        CM = confusion_matrices[prot_attr]

        # true negatives
        CM[0, 0] = true_negatives[prot_attr] / (
            true_negatives[prot_attr]+false_positives[prot_attr])

        # false positives
        CM[0, 1] = false_positives[prot_attr] / (
            true_negatives[prot_attr]+false_positives[prot_attr])

        # false negatives
        CM[1, 0] = false_negatives[prot_attr] / (
            true_positives[prot_attr]+false_negatives[prot_attr])

        # true positives
        CM[1, 1] = true_positives[prot_attr] / (
            true_positives[prot_attr]+false_negatives[prot_attr])
        confusion_matrices[prot_attr] = CM
        # print_confusion_matrix(CM)
    return confusion_matrices


def get_equalised_odds(X, Y_hat, Y, protected_attribute_index, thresholds):
    confusion_matrices = get_confusion_matrices(X, Y_hat, Y, protected_attribute_index, thresholds)
    CM_DIFF = confusion_matrices[UNFAVOURED] - confusion_matrices[FAVOURED]
    # for dTPR, this is actually false negatives because we want the odds of a positive outcome for the users across groups to be equal.

    dTPR = CM_DIFF[1, 1]
    dFPR = CM_DIFF[0, 1]

    # dTPR = CM_DIFF[0, 0]
    # dFPR = CM_DIFF[1, 0]

    # print("dTPR = " + str(dTPR))
    # print("dFPR = " + str(dFPR))
    return abs(dTPR), abs(dFPR)


def get_statistical_parity(X, Y_hat, Y, protected_attribute_index, thresholds):
    confusion_matrices = [np.zeros((2, 2)), np.zeros((2, 2))]

    pos_outcomes = np.zeros(2)

    counts = np.zeros(2)
    # positive -> predicted to recid
    # true -> correct prediction
    # Different actors want different minimisations

    for i in range(len(Y_hat)):
        # prediction 1 = recid, 0 = no recid
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat = 1
        else:
            y_hat = 0
        pos_outcomes[protected_attribute] += y_hat

        counts[protected_attribute] += 1
    P_favoured_outcome_unfavoured = ((counts[UNFAVOURED]-pos_outcomes[UNFAVOURED])/counts[UNFAVOURED])
    P_favoured_outcome_favoured = ((counts[FAVOURED]-pos_outcomes[FAVOURED])/counts[FAVOURED])
    epsilon = (P_favoured_outcome_unfavoured/P_favoured_outcome_favoured)
    # print("p(Y_hat|unfavoured) "+str(P_favoured_outcome_unfavoured))
    # print("p(Y_hat|favoured) "+str(P_favoured_outcome_favoured))
    # print(str(abs(epsilon*100))+" percent fair")

    return abs(1-epsilon)
