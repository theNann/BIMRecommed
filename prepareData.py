import pandas as pd
import numpy as np
import csv


def prepare_data_test():
    test_input = pd.read_csv('input/test_input.csv')
    test_input_p = np.array(test_input[['data_id', 'p1', 'p2', 'p3']])
    test_input_d = np.array(test_input[['data_id', 'd1', 'd2', 'd3']])

    test_output = []
    csv_file = open('input/test_output.csv', 'r')
    lines = csv.reader(csv_file)
    first = True
    for line in lines:
        if first == True:
            first = False
            continue
        tmp = []
        for num in line:
            tmp.append(int(num))
        test_output.append(tmp)
    test_output = np.array(test_output)
    return test_input_p, test_input_d, np.array(test_input), test_output
    pass


def prepare_data():
    data_train = pd.read_csv('input/data_train.csv')
    data_train_p = np.array(data_train[['data_id', 'p1', 'p2', 'p3']])
    data_train_d = np.array(data_train[['data_id', 'd1', 'd2', 'd3']])

    data_test = pd.read_csv('input/data_test.csv')
    data_test_p = np.array(data_test[['data_id', 'p1', 'p2', 'p3']])
    data_test_d = np.array(data_test[['data_id', 'd1', 'd2', 'd3']])

    target_test = []
    file_obj = open('input/target_test.txt')
    lines = file_obj.readlines()
    for line in lines:
        tmp = line.strip().split(", ")
        datas = []
        for tmp_elem in tmp:
            datas.append(int(tmp_elem))
        target_test.append(datas)
    target_test = np.array(target_test)
    file_obj.close()

    target_train = []
    file_obj = open('input/target_train.txt')
    lines = file_obj.readlines()
    for line in lines:
        tmp = line.strip().split(", ")
        datas = []
        for tmp_elem in tmp:
            datas.append(int(tmp_elem))
        target_train.append(datas)
    target_train = np.array(target_train)
    file_obj.close()
    # print(type(data_train_p), type(data_test_p), type(data_train_d), type(data_test_d),
    #  type(target_train), type(target_test))
    # print(data_train_p.shape, data_test_p.shape, data_train_d.shape, data_test_d.shape,
    # target_train.shape, target_test.shape)
    return data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test


def get_data_test_p():
    data_test = pd.read_csv('input/data_test.csv')
    data_test_p = np.array(data_test[['data_id', 'p1', 'p2', 'p3']])
    return data_test_p


def get_data_train_p():
    data_train = pd.read_csv('input/data_train.csv')
    data_train_p = np.array(data_train[['data_id', 'p1', 'p2', 'p3']])
    return data_train_p


def get_data_train_d():
    data_train = pd.read_csv('input/data_train.csv')
    data_train_d = np.array(data_train[['data_id', 'd1', 'd2', 'd3']])
    return data_train_d


def get_data_test_d():
    data_test = pd.read_csv('input/data_test.csv')
    data_test_d = np.array(data_test[['data_id', 'd1', 'd2', 'd3']])
    return data_test_d


def get_target_train():
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
    return target_train


def get_target_test():
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
    return target_test


def get_data_train():
    data_train = pd.read_csv('input/data_train.csv')
    return np.array(data_train)


def get_data_test():
    data_test = pd.read_csv('input/data_test.csv')
    return np.array(data_test)


def get_objects_info():
    objects_info = pd.read_csv('input/objectspositon.csv')
    return np.array(objects_info)
# def get_bim_data():
#     bim_data_df = pd.read_csv('input/bim_data.csv')
#     bim_data = np.array(bim_data_df)
#     print(type(bim_data))

