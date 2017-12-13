# data analysis and wrangling
import pandas as pd
import numpy as np
import csv
import math
# machine learning
from sklearn.neighbors import NearestNeighbors
import myparser


def prepare_data():
    data_train = pd.read_csv('input/data_train.csv')
    data_train_p = np.array(data_train[['p1', 'p2', 'p3']])
    data_train_d = np.array(data_train[['d1', 'd2', 'd3']])

    data_test = pd.read_csv('input/data_test.csv')
    data_test_p = np.array(data_test[['p1', 'p2', 'p3']])
    data_test_d = np.array(data_test[['d1', 'd2', 'd3']])

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


def get_nearest_neighbors(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test,
                           neighbors, radiu, type):
    if type == 'position':
        neigh = NearestNeighbors(n_neighbors=neighbors)
        neigh.fit(data_train_p)
        dists, inds = neigh.kneighbors(data_test_p, return_distance=True)
    else:
        neigh = NearestNeighbors(n_neighbors=neighbors, metric='cosine', algorithm='brute')
        neigh.fit(data_train_d)
        dists, inds = neigh.kneighbors(data_test_d, return_distance=True)

        # neigh = NearestNeighbors(radius=1 - math.cos(radiu), metric='cosine', algorithm='brute')
        # neigh.fit(data_train)
        # rng = neigh.radius_neighbors(data_test)
        # dists = np.asarray(rng[0])
        # inds = np.asarray(rng[1])
    return dists, inds


def get_score_by_neighbors(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test,
                           neighbors_p):
    Fov = 1.0472
    neighbors_d = 5
    dist_p, inds_p = get_nearest_neighbors(data_train_p, data_test_p, data_train_d, data_test_d, target_train,
                                           target_test, neighbors_p, radiu=0, type='position')
    dist_d, inds_d = get_nearest_neighbors(data_train_p, data_test_p, data_train_d, data_test_d, target_train,
                                           target_test, neighbors_d, radiu=0, type='direction')
    inds = []
    dist = []
    # culling p which it's d <d, test_d> > fov
    inds_p = list(inds_p)
    dist_p = list(dist_p)
    for i in range(len(data_test_p)):
        for j in range(len(inds_p[i])):
            del_ind = []
            d_test = data_test_d[i]
            d_train = data_train_d[inds_p[i][j]]
            cos, sim = myparser.cal_vector_similarity(np.mat(d_test), np.mat(d_train))
            if cos < math.cos(Fov/2):
                del_ind.append(j)
        inds_p[i] = np.delete(inds_p[i], del_ind)
        dist_p[i] = np.delete(dist_p[i], del_ind)
        # inds.append(list(set(inds_p[i]) | set(inds_d[i])))
        inds.append(inds_p[i])
        dist.append(dist_p[i])

    inds = np.array(inds)
    dist = np.array(dist)

    predict_by_union = []
    predict_by_inter = []
    score = []
    for i in range(len(inds)):
        union = set([])
        for j in range(len(inds[i])):
            union = union | set(target_train[inds[i][j]])

        intersection = set(target_train[inds[i][0]])
        for j in range(1, len(inds[i])):
            intersection = intersection & set(target_train[inds[i][j]])

        sim_by_inter_acc = 0
        sim_by_inter_recall = 0
        sim_by_union_acc = 0
        sim_by_union_recall = 0
        if len(union) != 0:
            sim_by_union_acc = len(union & set(target_test[i]))/len(union)
        if len(target_test[i]) != 0:
            sim_by_union_recall = len(union & set(target_test[i])) / len(target_test[i])
        if len(intersection) != 0:
            sim_by_inter_acc = len(intersection & set(target_test[i])) / len(intersection)
        if len(target_test[i]) != 0:
            sim_by_inter_recall = len(intersection & set(target_test[i])) / len(target_test[i])
        score.append([sim_by_union_acc, sim_by_union_recall, sim_by_inter_acc, sim_by_inter_recall])
        predict_by_union.append(list(union))
        predict_by_inter.append(list(intersection))

    return predict_by_union, predict_by_inter, score, dist, inds


def knn(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test):
    score_by_neighbors = []
    for neighbors in range(3, 16, 2):
        print('neighbor: ', neighbors, 'start...')
        predict_by_union, predict_by_inter, score, dists, inds = get_score_by_neighbors(data_train_p, data_test_p,
                                            data_train_d, data_test_d, target_train, target_test, neighbors)
        # statistics = []
        # for i in range(len(target_test)):
        #     statistics.append([i, score[i][0], score[i][1], score[i][2], score[i][3], len(predict_by_union[i]),
        #                        len(predict_by_inter[i]), len(target_test[i]), dists[i][0], dists[i][len(dists[i])-1],
        #                        inds[i]])
        # # write predict and score of each neighbor
        # csv_file = open('output/statistics_'+str(neighbors)+'_pd_culP.csv', 'w', newline='')
        # csv_writer = csv.writer(csv_file)
        # csv_writer.writerow(['index', 'sim_by_union_acc', 'sim_by_union_recall', 'sim_by_inter_acc',
        #                      'sim_by_inter_recall', 'len_union', 'len_inter', 'len_target', 'min_dist',
        #                      'max_dist', 'inds'])
        # csv_writer.writerows(statistics)
        # csv_file.close()

        # cal mean of each neighbor
        np_score = np.array(score)
        mean = np.mean(np_score, axis=0)
        score_by_neighbors.append([neighbors, mean[0], mean[1], mean[2], mean[3]])
        print('neighbor: ', neighbors, 'finish...')

    csv_file = open('output/score_by_neighbors_p_culP_30d.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['neighbors', 'score_by_union_acc', 'score_by_union_recall',
                         'score_by_inter_acc', 'score_by_inter_recall'])
    csv_writer.writerows(score_by_neighbors)
    csv_file.close()


def main():
    data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test = prepare_data()
    print('Prepare data finish...')
    knn(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test)


if __name__ == '__main__':
    main()
