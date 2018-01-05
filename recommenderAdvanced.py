import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import prepareData
import csv
import myparser
import knn
import recommender
import time
Fov = 1.0472


# find n neighbors according to 'position'
# return inds, whose shape is(1L, nL)
def cold_start(data_train_p, data_test_p, neighbors):
    dists, inds = knn.get_nearest_neighbors(data_train_p, data_test_p, neighbors, type='position')
    return inds


# find n neighbors according to 'position', and then delete neighbor which cos<d1,d2> > cos(fov)
# return inds, whose shape is(1L, nL)
def cold_start_advanced(data_train_p, data_test_p, data_train_d, data_test_d, neighbors, sub_neighbors):
    global Fov
    d_test = data_test_d[0]
    dits, inds = knn.get_nearest_neighbors(data_train_p, data_test_p, neighbors, type='position')
    coss = {}
    for i in range(len(inds[0])):
        index = inds[0][i]
        dist = dits[0][i]
        d_train = data_train_d[index]
        cos, sim = myparser.cal_vector_similarity(np.mat(d_test), np.mat(d_train))
        coss[index] = (sim, dist)
    sort_coss = sorted(coss.items(), key=lambda item: (item[1][0], -item[1][1]), reverse=True)
    # print(sort_coss)
    ans_inds = []
    for cos in sort_coss:
        if cos < math.cos(Fov)*0.5 + 0.5:
            break
        else:
            ans_inds.append(cos[0])
        if len(ans_inds) == sub_neighbors:
            break
    if len(ans_inds) == 0:
        ans_inds.append(inds[0][0])
    return np.array([ans_inds])


def user_based_recommend(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test, neighbors,
                         sub_neighbors, how_many):
    dist, inds = knn.extract_neighbors_by_sort_direction(data_train_p[:, 1:4], data_test_p[:, 1:4],
                                                         data_train_d[:, 1:4], data_test_d[:, 1:4],
                                                         neighbors, sub_neighbors)
    scores = []
    statics = []
    for i in range(len(data_test_p)):
        time1 = time.clock()
        set_total_predict_list = set()
        for j in range(len(inds[i])):
            index = inds[i][j]
            predict_list = target_train[index][1:]
            set_predict_list = recommender.top_match(target_train, predict_list, how_many=how_many)
            set_total_predict_list = set_total_predict_list | set_predict_list
        len_true_positive = len(set_total_predict_list & set(target_test[i][1:]))
        sim_acc = len_true_positive * 1.0 / len(set_total_predict_list)
        sim_recall = len_true_positive * 1.0 / len(set(target_test[i][1:]))
        scores.append([sim_acc, sim_recall])
        statics.append([sim_acc, sim_recall, len(set_total_predict_list), len(set(target_test[i][1:])),
                        len_true_positive, time.clock()-time1])
        print(i, time.clock()-time1)
    np_score = np.array(scores)
    mean = np.mean(np_score, axis=0)
    return mean, statics


def main():
    data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test = prepareData.prepare_data()
    test_input_p, test_input_d, test_input, test_output = prepareData.prepare_data_test()
    neighbors = 15
    sub_neighbors = 3
    how_many = 2
    print(test_input_p.shape, test_input_d.shape, test_input.shape, test_output.shape)
    mean, statics = user_based_recommend(data_train_p, test_input_p, data_train_d, test_input_d, target_train,
                                         test_output, neighbors, sub_neighbors, how_many)
    csv_file = open('output/test_statics.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['acc', 'recall', 'len_predict', 'len_output', 'len_true_positive', 'time'])
    csv_writer.writerows(statics)
    csv_file.close()

    add_data_train = []
    add_target_train = []
    for i in range(len(statics)):
        if statics[i][1] <= 0.5:
            add_data_train.append(test_input[i])
            add_target_train.append(test_output[i])

    csv_file = open('input/data_train.csv', 'ab')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(add_data_train)
    csv_file.close()

    csv_file = open('input/target_train.csv', 'ab')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(add_target_train)
    csv_file.close()



    # scores = []
    # neighbors_list = [4, 5]
    # for sub_neighbors in neighbors_list:
    #     mean = user_based_recommend(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test,
    #                                 neighbors, sub_neighbors, how_many)
    #     scores.append([sub_neighbors, mean[0], mean[1]])
    # csv_file = open('output/tmp.csv', 'ab')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['sub_neighbors', 'sim_acc', 'sim_recall'])
    # csv_writer.writerows(scores)
    # csv_file.close()
    pass


if __name__ == "__main__":
    main()
