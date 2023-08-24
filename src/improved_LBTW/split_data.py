import pandas as pd
import numpy as np
import random
import os
import json
import csv
from sklearn.model_selection import StratifiedKFold, KFold
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler



'''
Apply greedy method to split data when merging multi-task
'''


# def greedy_multi_splitting(data, k, directory, file_list):
#     class Node:
#         def __init__(self, retest, fp, rmi):
#             self.retest = retest
#             self.fp = fp
#             self.rmi = rmi
#             if not np.isnan(self.rmi):
#                 self.rmi = int(self.rmi)
#             else:
#                 self.rmi = np.NaN
#
#         def __str__(self):
#             ret = 'retest: {}, fp: {}, rmi: {}'.format(self.retest, self.fp, self.rmi)
#             return ret
#
#         def __eq__(self, other):
#             return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
#
#         def __hash__(self):
#             return hash(self.retest) ^ hash(self.fp) ^ hash(self.rmi)
#
#         def __cmp__(self, other):
#             return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
#
#     dict_ = {}
#     for ix, row in data.iterrows():
#         node = Node(row['Keck_Pria_AS_Retest'], row['Keck_Pria_FP_data'], row['Keck_RMI_cdd'])
#         if node not in dict_.keys():
#             dict_[node] = []
#         dict_[node].append(ix)
#
#     list_ = []
#     for key in dict_.keys():
#         one_group_list = np.array(dict_[key])
#         current = []
#
#         if len(one_group_list) <= k:
#             n = len(one_group_list)
#             for i in range(n):
#                 current.append(np.array(one_group_list[i]))
#             for i in range(n, k):
#                 current.append(np.array([]))
#         else:
#             # kf = KFold(len(one_group_list), k, shuffle=True) by xulu
#             kf = KFold(k, shuffle=True)
#             for _, test_index in kf.split(one_group_list):
#                 current.append(one_group_list[test_index])
#         random.shuffle(current)
#         list_.append(current)
#
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     print
#     len(list_)
#
#     for split in range(k):
#         index_block = np.hstack((list_[0][split],
#                                  list_[1][split],
#                                  list_[2][split],
#                                  list_[3][split],
#                                  list_[4][split],
#                                  list_[5][split],
#                                  list_[6][split],
#                                  list_[7][split],
#                                  list_[8][split]))
#         index_block = index_block.astype(int)
#         df_block = data.iloc[index_block]
#         print
#         df_block.shape
#
#         file_path = directory + file_list[split]
#         df_block.to_csv(file_path, index=None)
#
#     return

# 设置随机数种子
seed_value = 666
np.random.seed(seed_value)
random.seed(seed_value)

def greedy_multi_splitting(data, k, directory, file_list):
    class Node:
        def __init__(self, class1, class2, class3, class4):
            self.class1 = class1
            self.class2 = class2
            self.class3 = class3
            self.class4 = class4
            # if not np.isnan(self.rmi):
            #     self.rmi = int(self.rmi)
            # else:
            #     self.rmi = np.NaN

        def __str__(self):
            ret = 'class1: {}, class2: {}, class3: {}, class4: {}'.format(self.class1, self.class2, self.class3, self.class4)
            return ret

        def __eq__(self, other):
            return (self.class1, self.class2, self.class3, self.class4) == (other.class1, other.class2, other.class3, other.class4)

        def __hash__(self):
            return hash(self.class1) ^ hash(self.class2) ^ hash(self.class3) ^ hash(self.class4)

        def __cmp__(self, other):
            return (self.class1, self.class2, self.class3, self.class4) == (other.class1, other.class2, other.class3, other.class4)

    dict_ = {}
    for ix, row in data.iterrows():
        node = Node(row['HighBloodPressure'], row['Diabetes'], row['Hyperlipidemia'], row['HeartFailure'])
        if node not in dict_.keys():
            dict_[node] = []
        dict_[node].append(ix)

    list_ = []
    for key in dict_.keys():
        one_group_list = np.array(dict_[key])
        current = []

        if len(one_group_list) <= k:
            n = len(one_group_list)
            for i in range(n):
                current.append(np.array(one_group_list[i]))
            for i in range(n, k):
                current.append(np.array([]))
        else:
            # kf = KFold(len(one_group_list), k, shuffle=True) by xulu
            kf = KFold(k, shuffle=True)
            for _, test_index in kf.split(one_group_list):
                current.append(one_group_list[test_index])
        random.shuffle(current)
        list_.append(current)

    if not os.path.exists(directory):
        os.makedirs(directory)

    print
    len(list_)

    for split in range(k):
        index_block = np.hstack((list_[0][split],
                                 list_[1][split],
                                 list_[2][split],
                                 list_[3][split],
                                 list_[4][split],
                                 list_[5][split],
                                 list_[6][split],
                                 list_[7][split],
                                 list_[8][split],
                                 list_[9][split],
                                 list_[10][split],
                                 list_[11][split],
                                 list_[12][split],
                                 list_[13][split],
                                 list_[14][split],
                                 list_[15][split]))
        index_block = index_block.astype(int)
        df_block = data.iloc[index_block]
        print
        df_block.shape

        file_path = directory + file_list[split]
        df_block.to_csv(file_path, index=None)

    return


if __name__ == '__main__':
    data_test = pd.read_csv('../dataset/coronary/coronary_filter.csv')
    data_test['Gender'] = data_test['Gender'].replace(2, 0)
    data_test = data_test.drop("CoronaryDisease", axis=1)

    # 提取连续型特征及标签列
    continuous_features = data_test.columns[~data_test.columns.isin(['Gender', 'HighBloodPressure', 'Diabetes', 'Hyperlipidemia', 'HeartFailure'])]
    label_names = ['HighBloodPressure', 'Diabetes', 'Hyperlipidemia', 'HeartFailure']
    labels = data_test[label_names]

    # 使用 Z-Score 标准化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    data_test[continuous_features] = scaler.fit_transform(data_test[continuous_features])

    # 合并连续型特征和标签列
    one_hot_encoded = pd.get_dummies(data_test['Gender'], prefix='Gender')
    data_normalized = pd.concat([one_hot_encoded, data_test[continuous_features], labels], axis=1)

    # # 创建一个空列表用于存储特征列表
    # feature_list = []
    #
    # import itertools
    # # 定义文件名列表
    # file_names = ['../dataset/coronary/selected_features_class0_focal_loss_ROC.txt',
    #               '../dataset/coronary/selected_features_class1_focal_loss_ROC.txt',
    #               '../dataset/coronary/selected_features_class2_focal_loss_ROC.txt',
    #               '../dataset/coronary/selected_features_class3_focal_loss_ROC.txt']
    # # 循环读取文件，并将特征添加到列表中
    # for file_name in file_names:
    #     with open(file_name, 'r', encoding='utf-8') as file:
    #         features = set(line.rstrip('\n') for line in file)
    #         feature_list.append(features)
    #
    # # 查找在任意三个或三个以上文件中共同出现的特征
    # common_features = set()
    # for r in range(1, len(file_names) + 1):  # 遍历从2到文件总数的范围
    #     for combination in itertools.combinations(feature_list, r):  # 组合r个文件的特征集合
    #         common_features.update(set.intersection(*combination))
    #
    # # common_features = set.intersection(*feature_list)
    # col_names = list(common_features) + label_names
    # data_normalized = data_normalized[col_names]
    greedy_multi_splitting(data_normalized, 5, '../dataset/coronary/', ['file_0.csv', 'file_1.csv', 'file_2.csv', 'file_3.csv', 'file_4.csv'])