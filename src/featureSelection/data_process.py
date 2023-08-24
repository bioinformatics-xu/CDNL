import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)

def filter_missing_values(dataframe):
    columns = dataframe.columns
    for c in columns:
        dataframe[c][dataframe[c].notnull()] = 1
    dataframe.fillna(0, inplace=True)
    return dataframe

def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data

def extract_feature_and_label(data_pd,
                              feature_name,
                              task_list):

    X_data = data_pd[feature_name]
    # X_data = map(lambda x: list(x), X_data)  by xulu
    # X_data = list(map(lambda x: list(x), X_data))
    X_data = np.array(X_data)

    y_data = data_pd[task_list].values.tolist()
    y_data = np.array(y_data)
    y_data = reshape_data_into_2_dim(y_data)

    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    return X_data, y_data

def transform_dataframe2array(dataframe):
    data = np.array(dataframe.values.tolist())
    return reshape_data_into_2_dim(data)

class PCBADataset(Dataset):
    def __init__(self, data_files, feature_name, task_list, transform):
        column_names = feature_name.tolist()
        column_names.extend(task_list)
        # column_names = [str(feature_name)]
        # column_names.extend([str(task) for task in task_list])
        pcba_frame = read_merged_data(data_files)
        pcba_frame = pcba_frame[column_names]
        # pcba_frame.dropna(inplace=True, subset=task_list, how='all')
        print('Update data shape\t', pcba_frame.shape)
        weight_frame = filter_missing_values(pcba_frame[task_list].copy())
        # by xulu
        # weight_frame = pcba_frame[task_list]

        self.feature_name = feature_name
        self.task_list = task_list
        self.transform = transform
        # pcba_frame.fillna(0, inplace=True)
        self.data, self.label = extract_feature_and_label(pcba_frame,
                                                          feature_name=self.feature_name,
                                                          task_list=self.task_list)
        self.weight = transform_dataframe2array(weight_frame)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        label_sample = self.label[idx]
        weight_sample = self.weight[idx]
        if self.transform:
            data_sample = self.transform(data_sample)
            label_sample = self.transform(label_sample)
            weight_sample = self.transform(weight_sample)
        return data_sample, label_sample, weight_sample


def load_data(conf):
    task_list = conf['task_list']
    data_directory = conf['data_dir']
    k = 5
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    data_file_list = [data_directory + 'file_{}.csv'.format(i) for i in range(5)]

    data_pd = pd.read_csv(data_file_list[1])
    # feature_name_list = data_pd.columns[~data_pd.columns.isin(task_list)]
    feature_name_list = data_pd.columns[:conf['feature_num']]

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

def read_merged_data(input_file_list):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        # # TODO: This is for debugging
        # data_pd = pd.read_csv(input_file, nrows=5000)
        data_pd = pd.read_csv(input_file)
        whole_pd = whole_pd.append(data_pd)
    print('Data shape\t{}'.format(whole_pd.shape))
    return whole_pd