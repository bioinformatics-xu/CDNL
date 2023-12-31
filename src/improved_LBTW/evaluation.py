import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, accuracy_score
from function import reshape_data_into_2_dim


def roc_auc_multi(y_true, y_pred, sample_weight, eval_indices, eval_mean_or_median):
    """
    This if for multi-task evaluation
    """
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    auc = np.zeros(nb_classes)
    for i in eval_indices:
        # print('Label number is: {}'.format(len(np.unique(y_true[:, i]))))
        if len(np.unique(y_true[:, i])) == 1:
            auc[i] = 0
            print('Class: {} has only one label'.format(i))
        else:
            auc[i] = roc_auc_single(y_true[:, i], y_pred[:, i], sample_weight[:, i])
    return eval_mean_or_median(auc)

def roc_auc_single(actual, predicted, sample_weight=None):
    actual = reshape_data_into_2_dim(actual)
    predicted = reshape_data_into_2_dim(predicted)
    if sample_weight is not None:
        non_missing_indices = np.argwhere(sample_weight == 1)[:, 0]
        actual = actual[non_missing_indices]
        predicted = predicted[non_missing_indices]
    return roc_auc_score(actual, predicted)

def acc_single(actual, predicted, sample_weight=None):
    actual = reshape_data_into_2_dim(actual)
    predicted = reshape_data_into_2_dim(predicted)
    if sample_weight is not None:
        non_missing_indices = np.argwhere(sample_weight == 1)[:, 0]
        actual = actual[non_missing_indices]
        predicted = predicted[non_missing_indices]
    return roc_auc_score(actual, predicted)

def precision_auc_multi(y_true, y_pred, sample_weight, eval_indices, eval_mean_or_median):
    """
    This if for multi-task evaluation
    """
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    auc = np.zeros(nb_classes)
    for i in eval_indices:
        # print('Label number is: {}'.format(len(np.unique(y_true[:, i]))))
        if len(np.unique(y_true[:, i])) == 1:
            auc[i] = 0
            print('Class: {} has only one label'.format(i))
        else:
            auc[i] = precision_auc_single(y_true[:, i], y_pred[:, i], sample_weight[:, i])
    return eval_mean_or_median(auc)

def precision_auc_single(actual, predicted, sample_weight=None):
    actual = reshape_data_into_2_dim(actual)
    predicted = reshape_data_into_2_dim(predicted)
    if sample_weight is not None:
        non_missing_indices = np.argwhere(sample_weight == 1)[:, 0]
        actual = actual[non_missing_indices]
        predicted = predicted[non_missing_indices]
    prec_auc = average_precision_score(actual, predicted)
    return prec_auc

def acc_single(actual, predicted, sample_weight=None):
    actual = reshape_data_into_2_dim(actual)
    predicted = reshape_data_into_2_dim(predicted)
    if sample_weight is not None:
        non_missing_indices = np.argwhere(sample_weight == 1)[:, 0]
        actual = actual[non_missing_indices]
        predicted = predicted[non_missing_indices]
    acc = accuracy_score(actual, predicted)
    return acc

def acc_multi(y_true, y_pred, sample_weight, eval_indices, eval_mean_or_median):
    """
    This if for multi-task evaluation
    """
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    acc = np.zeros(nb_classes)
    for i in eval_indices:
        acc[i] = acc_single(y_true[:, i], y_pred[:, i], sample_weight[:, i])
    return eval_mean_or_median(acc)


def enrichment_factor_single(labels_arr, scores_arr, percentile, sample_weight):
    '''
    calculate the enrichment factor
    '''
    if sample_weight is not None:
        non_missing_indices = np.argwhere(sample_weight == 1)[:, 0]
        labels_arr = labels_arr[non_missing_indices]
        scores_arr = scores_arr[non_missing_indices]

    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    pred = np.sort(scores_arr, axis=0)[::-1][:sample_size]  # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr, axis=0)[::-1][:sample_size]  # get the index positions for these in library
    n_actives = np.nansum(labels_arr)  # count number of positive labels in library
    total_actives = np.nansum(labels_arr)
    total_count = len(labels_arr)
    n_experimental = np.nansum(labels_arr[indices])  # count number of positive labels in subset
    temp = scores_arr[indices]

    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile  # calc EF at percentile
        ef_max = min(n_actives, sample_size) / (n_actives * percentile)
    else:
        ef = 'ND'
        ef_max = 'ND'
    return n_actives, ef, ef_max