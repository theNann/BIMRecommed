# data analysis and wrangling
import numpy as np
import csv
import math
# machine learning
from sklearn.neighbors import NearestNeighbors
import myparser
import prepareData
Fov = 1.0472

# def get_nearest_neighbors(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test,
#                           neighbors, radiu, type):
#     if type == 'position':
#         neigh = NearestNeighbors(n_neighbors=neighbors)
#         neigh.fit(data_train_p[:, 1:])
#         dists, inds = neigh.kneighbors(data_test_p[:, 1:], return_distance=True)
#     else:
#         neigh = NearestNeighbors(n_neighbors=neighbors, metric='cosine', algorithm='brute')
#         neigh.fit(data_train_d[:, 1:])
#         dists, inds = neigh.kneighbors(data_test_d[:, 1:], return_distance=True)
#
#         # neigh = NearestNeighbors(radius=1 - math.cos(radiu), metric='cosine', algorithm='brute')
#         # neigh.fit(data_train)
#         # rng = neigh.radius_neighbors(data_test)
#         # dists = np.asarray(rng[0])
#         # inds = np.asarray(rng[1])
#     return dists, inds


# data_train and data_test is np.array, and shape is(XX,3L)
def get_nearest_neighbors(data_train, data_test, neighbors, type):
    if type == 'position':
        neigh = NearestNeighbors(n_neighbors=neighbors)
        neigh.fit(data_train)
        dists, inds = neigh.kneighbors(data_test, return_distance=True)
    else:
        neigh = NearestNeighbors(n_neighbors=neighbors, metric='cosine', algorithm='brute')
        neigh.fit(data_train)
        dists, inds = neigh.kneighbors(data_test, return_distance=True)
    return dists, inds


def extract_neighbors_by_fov(data_train_p, data_test_p, data_train_d, data_test_d, neighbors):
    global Fov
    dist_p, inds_p = get_nearest_neighbors(data_train_p, data_test_p, neighbors, type='position')

    inds = []
    dist = []
    # culling p which it's d <d, test_d> > fov
    for i in range(len(data_test_p)):
        # print(neighbors, i)
        # for i in range(6):
        sub_inds = []
        sub_dist = []
        for j in range(len(inds_p[i])):
            d_test = data_test_d[i]
            d_train = data_train_d[inds_p[i][j]]
            cos, sim = myparser.cal_vector_similarity(np.mat(d_test), np.mat(d_train))
            if cos >= math.cos(Fov):
                sub_inds.append(inds_p[i][j])
                sub_dist.append(dist_p[i][j])
        if len(sub_inds) == 0:
            sub_inds.append(inds_p[i][0])
            sub_dist.append(dist_p[i][0])
        inds.append(sub_inds)
        dist.append(sub_dist)

    inds = np.array(inds)
    dist = np.array(dist)
    return dist, inds
    pass


def extract_neighbors_by_sort_direction(data_train_p, data_test_p, data_train_d, data_test_d, neighbors, sub_neighbors):
    dist_p, inds_p = get_nearest_neighbors(data_train_p, data_test_p, neighbors, type='position')

    inds = []
    dist = []
    # culling p which it's d <d, test_d> > fov
    for i in range(len(data_test_p)):
        # print(neighbors, i)
        # for i in range(6):
        sub_inds = []
        sub_dist = []
        sub_cos = {}
        d_test = data_test_d[i]

        for j in range(len(inds_p[i])):
            index = inds_p[i][j]
            d_train = data_train_d[index]
            cos, sim = myparser.cal_vector_similarity(np.mat(d_test), np.mat(d_train))
            sub_cos[index] = (cos, dist_p[i][j])
        sort_sub_cos = sorted(sub_cos.items(), key=lambda item: (item[1][0], -item[1][1]), reverse=True)
        for cos in sort_sub_cos:
            if cos[1][0] < math.cos(Fov):
                break
            else:
                sub_inds.append(cos[0])
                sub_dist.append(cos[1][1])
            if len(sub_inds) == sub_neighbors:
                break
        if len(sub_inds) == 0:
            sub_inds.append(inds_p[i][0])
            sub_dist.append(dist_p[i][0])
        inds.append(sub_inds)
        dist.append(sub_dist)
    inds = np.array(inds)
    dist = np.array(dist)
    return dist, inds
    pass


def get_score_by_neighbors(inds, dist, target_train, target_test, neighbors):
    predict_by_union = []
    predict_by_inter = []
    score = []
    for i in range(len(inds)):
        # print(neighbors, i)
        union = set([])
        for j in range(len(inds[i])):
            union = union | set(target_train[inds[i][j]][1:])
        intersection = set(target_train[inds[i][0]][1:])
        for j in range(1, len(inds[i])):
            intersection = intersection & set(target_train[inds[i][j]][1:])

        sim_by_inter_acc = 0
        sim_by_inter_recall = 0
        sim_by_union_acc = 0
        sim_by_union_recall = 0
        if len(union) != 0:
            sim_by_union_acc = len(union & set(target_test[i][1:]))*1.0/len(union)
        if len(target_test[i][1:]) != 0:
            sim_by_union_recall = len(union & set(target_test[i][1:]))*1.0 / len(target_test[i][1:])
        if len(intersection) != 0:
            sim_by_inter_acc = len(intersection & set(target_test[i][1:]))*1.0 / len(intersection)
        if len(target_test[i][1:]) != 0:
            sim_by_inter_recall = len(intersection & set(target_test[i][1:]))*1.0 / len(target_test[i][1:])
        score.append([sim_by_union_acc, sim_by_union_recall, sim_by_inter_acc, sim_by_inter_recall])
        predict_by_union.append(list(union))
        predict_by_inter.append(list(intersection))

    return predict_by_union, predict_by_inter, score, dist, inds


# decide call which function according to sub_neighbors is None or not
def knn_small(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test,
              neighbors, sub_neighbors=None):
    if sub_neighbors is not None:
        dist, inds = extract_neighbors_by_sort_direction(data_train_p, data_test_p, data_train_d, data_test_d, neighbors
                                                         , sub_neighbors)
    else:
        dist, inds = extract_neighbors_by_fov(data_train_p, data_test_p, data_train_d, data_test_d, neighbors)
    predict_by_union, predict_by_inter, score, dist, inds = get_score_by_neighbors(inds, dist, target_train,
                                                                                   target_test, neighbors)
    return predict_by_union, predict_by_inter, score, dist, inds
    pass


def knn_8(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test):
    score_by_neighbors = []
    for neighbors in range(5, 6, 1):
        print('neighbor: ', neighbors, 'start...')
        predict_by_union, predict_by_inter, score, dists, inds = \
            knn_small(data_train_p[:, 1:4], data_test_p[:, 1:4], data_train_d[:, 1:4], data_test_d[:, 1:4],
                      target_train, target_test, neighbors=15, sub_neighbors=neighbors)

        statistics = []
        for i in range(len(target_test)):
            statistics.append([i, score[i][0], score[i][1], score[i][2], score[i][3], len(predict_by_union[i]),
                               len(predict_by_inter[i]), len(target_test[i][1:])])
        # write predict and score of each neighbor
        csv_file = open('output/knn_8.csv', 'wb')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['index', 'sim_by_union_acc', 'sim_by_union_recall', 'sim_by_inter_acc',
                             'sim_by_inter_recall', 'len_union', 'len_inter', 'len_target'])
        csv_writer.writerows(statistics)
        csv_file.close()

        # cal mean of each neighbor
        # np_score = np.array(score)
        # mean = np.mean(np_score, axis=0)
        # score_by_neighbors.append([neighbors, mean[0], mean[1], mean[2], mean[3]])
    # csv_file = open('output/tmp.csv', 'ab')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['neighbors', 'score_by_union_acc', 'score_by_union_recall',
    #                      'score_by_inter_acc', 'score_by_inter_recall'])
    # csv_writer.writerows(score_by_neighbors)
    # csv_file.close()


def knn_2(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test, neighbors=None):
    # for neighbors in range(6, 7, 1):
    # print('neighbor: ', neighbors, 'start...')
    if neighbors is None:
        neighbors = 5
    predict_by_union, predict_by_inter, score, dists, inds = \
        knn_small(data_train_p[:, 1:4], data_test_p[:, 1:4], data_train_d[:, 1:4], data_test_d[:, 1:4],
                  target_train, target_test, neighbors=neighbors)

    # statistics = []
    # for i in range(len(target_test)):
    #     statistics.append([i, score[i][0], score[i][1], score[i][2], score[i][3], len(predict_by_union[i]),
    #                        len(predict_by_inter[i]), len(target_test[i][1:])])
    # # write predict and score of each neighbor
    # csv_file = open('output/tmp_knn_2.csv', 'wb')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['index', 'sim_by_union_acc', 'sim_by_union_recall', 'sim_by_inter_acc',
    #                      'sim_by_inter_recall', 'len_union', 'len_inter', 'len_target'])
    # csv_writer.writerows(statistics)
    # csv_file.close()
    return score


def main():
    data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test = prepareData.prepare_data()
    test_input_p, test_input_d, test_input, test_output = prepareData.prepare_data_test()
    for target_test_i in target_test:
        print(target_test_i)
    print('Prepare data finish...')
    # knn_8(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test)
    # knn_2(data_train_p, test_input_p, data_train_d, test_input_d, target_train, test_output)


if __name__ == '__main__':
    main()
