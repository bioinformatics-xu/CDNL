from __future__ import print_function
from __future__ import absolute_import

import argparse
import pandas as pd
import numpy as np
import json
import math
import sys
import os
import time
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as auto
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

from pcba_model import *
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())


def variable_to_tensor(x):
    return x.data


def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

def load_data(conf):
    task_list = conf['task_list']
    data_directory = given_args.data_dir
    k = 5
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    data_file_list = [data_directory + 'file_{}.csv'.format(i) for i in range(5)]

    feature_name_list = []
    file_name = '../dataset/coronary/selected_features_' + task_list[0] + '.txt'
    with open(file_name, 'r', encoding='utf-8') as file:
        features = [line.rstrip('\n') for line in file]
        feature_name_list.extend(features)

    if len(task_list) == 4:
        data_pd = pd.read_csv(data_file_list[1])
        feature_name_list = list(data_pd.columns[~data_pd.columns.isin(task_list)])

    train_dataset = PCBADataset(data_files=data_file_list[0:3],
                                feature_name=feature_name_list,
                                task_list=task_list,
                                transform=ToTensor())
    val_dataset = PCBADataset(data_files=data_file_list[3:4],
                              feature_name=feature_name_list,
                              task_list=task_list,
                              transform=ToTensor())
    test_dataset = PCBADataset(data_files=data_file_list[4:5],
                               feature_name=feature_name_list,
                               task_list=task_list,
                               transform=ToTensor())
    print('Done Data Preparation!')
    return train_dataset, val_dataset, test_dataset

def train_and_test_LBTW(dp_classifier, max_epoch, training_dataset, **kwargs):
    train_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                   batch_size=dp_classifier.fit_batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    enable_gpu = kwargs['enable_gpu']
    task_list = kwargs['task_list']
    alpha = kwargs['alpha']
    dp_classifier.fc_nn.eval()

    N = len(training_dataset)
    dp_classifier.on_train_begin()
    for epoch in range(max_epoch):
        print('Epoch: {}'.format(epoch))
        dp_classifier.on_epoch_begin()
        training_loss_value = 0
        for i, (X_batch, y_batch, sample_weight_batch) in enumerate(train_dataloader):
            if enable_gpu:
                X_batch = Variable(X_batch.float().cuda())
                y_batch = Variable(y_batch.float().cuda())
                sample_weight_batch = Variable(sample_weight_batch.float().cuda())
            else:
                X_batch = Variable(X_batch.float())
                y_batch = Variable(y_batch.float())
                sample_weight_batch = Variable(sample_weight_batch.float())

            dp_classifier.optimizer.zero_grad()
            y_pred = dp_classifier.fc_nn(X_batch)
            loss_list = dp_classifier.multi_task_cost(y_pred, y_batch, sample_weight_batch, reduce=False)

            if epoch < 5:
                loss_ratio = torch.ones(len(task_list))
            elif (epoch == 5) and (i == 0):
                initial_task_loss_list = loss_list.data
                logging.info('initial loss\t{}'.format(initial_task_loss_list))

            if (epoch == 5 and i > 0) or (epoch > 5):
                loss_ratio = loss_list.data / initial_task_loss_list

            inverse_traing_rate = loss_ratio
            class_weights = inverse_traing_rate.pow(alpha)
            class_weights = class_weights / sum(class_weights) * len(task_list)
            class_weights = Variable(class_weights, requires_grad=False)
            logging.info('class weight is: {}'.format(class_weights))
            logging.info('loss is: {}'.format(loss_list))
            loss_list = torch.mul(loss_list, class_weights)
            logging.debug('loss is: {}'.format(loss_list))

            loss = loss_list.sum()
            # training_loss_value += loss.data[0] by xulu
            training_loss_value += loss.item()
            # backward prop
            loss.backward(retain_graph=True)
            # update the model
            dp_classifier.optimizer.step()

            logging.debug('{}/{}: Loss is {:.6f}'.format(i, N / dp_classifier.fit_batch_size, loss.item()))

        avg_loss = training_loss_value / (1.0*N/dp_classifier.fit_batch_size)
        dp_classifier.on_epoch_end(avg_loss)
        if dp_classifier.stop_training:
            break
    dp_classifier.on_train_end()
    return

def main(args):
    config_json_file = args.config_json_file
    with open(config_json_file, 'r') as f:
        conf = json.load(f)

    task_list = args.task_list

    conf['task_list'] = task_list
    conf['common_feature_num'] = args.common_feature_num
    conf['mode'] = args.mode
    print('task_list ', task_list)

    train_dataset, val_dataset, test_dataset = load_data(conf)
    conf['feature_num'] = len(train_dataset.feature_name)

    kwargs = {'file_path': args.model_weight_file, 'score_path': args.score_path,
              'enable_gpu': args.enable_gpu, 'seed': args.seed,
              'training_dataset': train_dataset, 'validation_dataset':val_dataset, 'test_dataset':test_dataset}
    dp_classifier = MultiTaskModel(conf=conf, **kwargs)
    dp_classifier.build_model()

    if args.enable_gpu:
        dp_classifier.fc_nn.cuda()

    print('Running improved Loss-Balanced Task Weighting.')
    kwargs = {'score_path': args.score_path, 'task_list': task_list,
              'alpha': conf['alpha'], 'enable_gpu': args.enable_gpu}
    train_and_test_LBTW(dp_classifier, max_epoch=conf['fitting']['nb_epoch'],
                        training_dataset=train_dataset, **kwargs)

#['HighBloodPressure', 'Diabetes', 'Hyperlipidemia', 'HeartFailure']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_list', action='store', dest='task_list', nargs='+', required=False, default=['HighBloodPressure', 'Diabetes', 'Hyperlipidemia', 'HeartFailure'])
    parser.add_argument('--config_json_file', action='store', dest='config_json_file', required=False, default="LBTW.json")
    parser.add_argument('--model_weight_file', action='store', dest='model_weight_file', required=False, default="./model_weight_file_coronary")
    parser.add_argument('--score_path', action='store', dest='score_path', required=False, default= '../results_LBTW')
    parser.add_argument('--enable-gpu', action='store_true', dest='enable_gpu', required=False, default=False)
    parser.add_argument('--data_dir', action='store', dest='data_dir',  required=False, default='../../toyData/')
    parser.set_defaults(enable_gpu=False)
    parser.add_argument('--seed', action='store', dest='seed', default=123, required=False)
    parser.add_argument('--common_feature_num', dest='common_feature_num', default=2, type=int, required=False)
    parser.set_defaults(enable_gpu=False)
    given_args = parser.parse_args()
    main(given_args)
