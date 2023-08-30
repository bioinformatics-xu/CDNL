#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from lassonet import LassoNetClassifier, plot_path
from lassonet.interfaces import LassoNetClassifierCV
from lassonet import plot_cv
import pandas as pd
import data_process
import numpy as np
import random
import heapq

seed_value = 666
np.random.seed(seed_value)
random.seed(seed_value)

def eval_on_path(model, path, X_test, y_test, *, score_function=None):
    if score_function is None:
        score_fun = model.score
    else:
        assert callable(score_function)

        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict_proba(X_test))

    score = []
    for save in path:
        model.load(save.state_dict)
        score.append(score_fun(X_test, y_test))
    return score

# 将列名写入文件
def write_selected_features(file_path, selected_features):
    with open(file_path, 'w') as file:
        for column in selected_features:
            file.write(column + '\n')

# def outputFI(model,path,score_index, file_path_fi):
#     model.load(path[score_index])
#
#     importances = model.feature_importances_.numpy()
#     order = np.argsort(importances)[::-1]
#     importances = importances[order]
#     feature_names = test_dataset.feature_name
#     ordered_feature_names = [feature_names[i] for i in order]
#
#     importance_data = np.column_stack((ordered_feature_names, importances))
#     np.savetxt(file_path_fi, importance_data, delimiter=' ',
#                fmt='%s')

conf = {
    "task_list": ['HighBloodPressure', 'Diabetes', 'Hyperlipidemia', 'HeartFailure'],
    "data_dir": "../../toyData/",
    "feature_num": 43
}

train_dataset, val_dataset, _ = data_process.load_data(conf)


for i in range(0,4):
    label_name = conf['task_list'][i]
    # file_ = open('./{}.out'.format(label_name), 'w')
    X_train = train_dataset.data
    y_train = train_dataset.label[:, i]

    X_val = val_dataset.data
    y_val = val_dataset.label[:, i]

    # X_test = test_dataset.data
    # y_test = test_dataset.label[:, i]

    model = LassoNetClassifier(hidden_dims=(200, 200), dropout=0.25, batch_size=64, n_iters=(1000, 100))
    path = model.path(X_train, y_train)

    score = eval_on_path(model, path, X_val, y_val, score_function=None)

    top_scores = heapq.nlargest(3, enumerate(score), key=lambda x: x[1])
    top_indices = [index for index, _ in top_scores]

    selected_features_index0 = path[top_indices[0]].selected
    selected_features0 = val_dataset.feature_name[selected_features_index0]
    # file_path0 = './coronary/FocalLoss2/selected_features_' + label_name + '_Top1.txt'
    # write_selected_features(file_path0, selected_features0)
    # outputFI(model,path,top_indices[0],'./coronary/FocalLoss2/features_importance_' + label_name + '_Top1.txt')

    selected_features_index1 = path[top_indices[1]].selected
    selected_features1 = val_dataset.feature_name[selected_features_index1]
    # file_path1 = './coronary/FocalLoss2/selected_features_' + label_name + '_Top2.txt'
    # write_selected_features(file_path1, selected_features1)
    # outputFI(model,path,top_indices[1], './coronary/FocalLoss2/features_importance_' + label_name + '_Top2.txt')

    selected_features_index2 = path[top_indices[2]].selected
    selected_features2 = val_dataset.feature_name[selected_features_index2]
    # file_path2 = './coronary/FocalLoss2/selected_features_' + label_name + '_Top3.txt'
    # write_selected_features(file_path2, selected_features2)
    # outputFI(model,path,top_indices[2], './coronary/FocalLoss2/features_importance_' + label_name + '_Top3.txt')

    selected_features = list((set(selected_features0) & set(selected_features1)) | (set(selected_features0) & set(selected_features2)) | (set(selected_features1) & set(selected_features2)))

    file_path = 'selected_features_' + label_name + '.txt'
    write_selected_features(file_path, selected_features)




