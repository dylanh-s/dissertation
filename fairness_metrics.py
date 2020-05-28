import numpy as np

# 1 == male, white
FAVOURED = 0
# 0 == female, non_white
UNFAVOURED = 1


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


def distance(pointA, pointB, _norm=np.linalg.norm):
    return _norm(pointA - pointB)


def speedy_metrics(X, Y_hat, Y, protected_attribute_index, thresholds):
    # calculate SP, dTPR, dFPR, pSUFF, nSUFF in one loop through Y
    counts = np.zeros(2)

    positive_preds = np.zeros(2)

    negative_preds = np.zeros(2)

    true_positives = np.zeros(2)

    true_negatives = np.zeros(2)

    false_positives = np.zeros(2)

    false_negatives = np.zeros(2)

    for i in range(len(Y_hat)):
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat = 1
        else:
            y_hat = 0
        counts[protected_attribute] += 1

        if (Y[i] == 1 and y_hat == 1):
            true_positives[protected_attribute] += 1
        if (Y[i] == 0 and y_hat == 0):
            true_negatives[protected_attribute] += 1

        if (Y[i] == 0 and y_hat == 1):
            false_positives[protected_attribute] += 1
        if ((Y[i] == 1 and y_hat == 0)):
            false_negatives[protected_attribute] += 1

        if (y_hat == 1):
            positive_preds[protected_attribute] += 1

    negative_preds = counts-positive_preds

    # SP
    P = positive_preds/(counts)
    SP = abs(P[0]-P[1])
    # SP = 1-(min(P[0], P[1])/(max(P[0], P[1])))

    # dTPR
    TPR = true_positives / (true_positives+false_negatives)
    dTPR = abs(TPR[0]-TPR[1])

    # dFPR
    FPR = false_positives / (true_negatives+false_positives)
    dFPR = abs(FPR[0]-FPR[1])

    # pSUFF
    P = true_positives/(positive_preds+1)
    pSUFF = abs(P[0]-P[1])

    # nSUFF
    P = true_negatives/(negative_preds+1)
    nSUFF = abs(P[0]-P[1])

    return SP, dTPR, dFPR, pSUFF, nSUFF


def get_individual_fairness(X, Y_hat, protected_attribute_index, thresholds):
    min = 10000000
    for i in range(len(Y_hat)):
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat_i = 1
        else:
            y_hat_i = 0
        for j in range(len(Y_hat)):

            protected_attribute = int(X[j, protected_attribute_index])
            if (Y_hat[j] > thresholds[protected_attribute]):
                y_hat_j = 1
            else:
                y_hat_j = 0

            if (y_hat_i != y_hat_j):
                dist = distance(X[i, :], X[j, :])
                if (dist < min):
                    min = dist
    # print("fin")
    return min


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

    return abs(dTPR), abs(dFPR)
    # return abs(dTPR), abs(dFPR)


def get_statistical_parity(X, Y_hat, Y, protected_attribute_index, thresholds):

    # difference between probability Y=1 given Yhat = 1 for white and black
    count_favoured = 0
    favoured_positive_preds = 0

    count_unfavoured = 0
    unfavoured_positive_preds = 0
    for i in range(len(Y_hat)):
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat = 1
        else:
            y_hat = 0

        if (protected_attribute == FAVOURED):
            count_favoured += 1
            if (y_hat == 1):
                favoured_positive_preds += 1

        if (protected_attribute == UNFAVOURED):
            count_unfavoured += 1
            if (y_hat == 1):
                unfavoured_positive_preds += 1

    P0 = favoured_positive_preds/(count_favoured+1)
    P1 = unfavoured_positive_preds/(count_unfavoured+1)
    SP = 1-(min(P0, P1)/(max(P0, P1)))

    return SP


def positive_sufficiency(X, Y_hat, Y, protected_attribute_index, thresholds):
    # difference between probability Y=1 given Yhat = 1 for white and black
    favoured_true_positives = 0
    favoured_positive_preds = 0

    unfavoured_true_positives = 0
    unfavoured_positive_preds = 0
    for i in range(len(Y_hat)):
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat = 1
        else:
            y_hat = 0

        if (protected_attribute == FAVOURED):
            if (Y[i] == 1 and y_hat == 1):
                favoured_true_positives += 1
            if (y_hat == 1):
                favoured_positive_preds += 1

        if (protected_attribute == UNFAVOURED):
            if (Y[i] == 1 and y_hat == 1):
                unfavoured_true_positives += 1
            if (y_hat == 1):
                unfavoured_positive_preds += 1

    P0 = favoured_true_positives/(favoured_positive_preds+1)
    P1 = unfavoured_true_positives/(unfavoured_positive_preds+1)
    return abs(P0-P1)


def negative_sufficiency(X, Y_hat, Y, protected_attribute_index, thresholds):
    # difference between probability Y=0 given Y_hat = 0 for white and black
    favoured_true_negatives = 0
    favoured_negative_preds = 0
    unfavoured_true_negatives = 0
    unfavoured_negative_preds = 0
    for i in range(len(Y_hat)):
        protected_attribute = int(X[i, protected_attribute_index])
        if (Y_hat[i] > thresholds[protected_attribute]):
            y_hat = 1
        else:
            y_hat = 0

        if (protected_attribute == FAVOURED):
            if (Y[i] == 0 and y_hat == 0):
                favoured_true_negatives += 1
            if (y_hat == 0):
                favoured_negative_preds += 1

        if (protected_attribute == UNFAVOURED):
            if (Y[i] == 0 and y_hat == 0):
                unfavoured_true_negatives += 1
            if (y_hat == 0):
                unfavoured_negative_preds += 1

    P0 = favoured_true_negatives/(favoured_negative_preds+1)
    P1 = unfavoured_true_negatives/(unfavoured_negative_preds+1)
    return abs(P0-P1)


def sufficiency(X, Y_hat, Y, protected_attribute_index, thresholds):
    pos_suff = positive_sufficiency(X, Y_hat, Y, protected_attribute_index, thresholds)
    neg_suff = negative_sufficiency(X, Y_hat, Y, protected_attribute_index, thresholds)

    return pos_suff, neg_suff
