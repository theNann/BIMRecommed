import pandas as pd
import numpy as np
import csv


def prepare_data():
    data_train = pd.read_csv('input/data_train.csv')
    data_train_p = np.array(data_train[['data_id', 'p1', 'p2', 'p3']])
    data_train_d = np.array(data_train[['data_id', 'd1', 'd2', 'd3']])

    data_test = pd.read_csv('input/data_test.csv')
    data_test_p = np.array(data_test[['data_id', 'p1', 'p2', 'p3']])
    data_test_d = np.array(data_test[['data_id', 'd1', 'd2', 'd3']])

    target_test = []
    csv_file = open('input/target_test.csv', 'r')
    lines = csv.reader(csv_file)
    first = True
    for line in lines:
        if first == True:
            first = False
            continue
        tmp = []
        for num in line:
            tmp.append(int(num))
        target_test.append(tmp)
    target_test = np.array(target_test)

    target_train = []
    csv_file = open('input/target_train.csv', 'r')
    lines = csv.reader(csv_file)
    first = True
    for line in lines:
        if first == True:
            first = False
            continue
        tmp = []
        for num in line:
            tmp.append(int(num))
        target_train.append(tmp)
    target_train = np.array(target_train)
    # print(type(data_train_p), type(data_test_p), type(data_train_d), type(data_test_d),
    #  type(target_train), type(target_test))
    # print(data_train_p.shape, data_test_p.shape, data_train_d.shape, data_test_d.shape,
    # target_train.shape, target_test.shape)
    return data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test