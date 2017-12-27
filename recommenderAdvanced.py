import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import prepareData
import csv
import myparser
import knn
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
    for index in inds[0]:
        d_train = data_train_d[index]
        cos, sim = myparser.cal_vector_similarity(np.mat(d_test), np.mat(d_train))
        coss[index] = cos
    sort_coss = sorted(coss.items(), key=lambda item: item[1], reverse=True)
    print(sort_coss)
    ans_inds = []
    for cos in sort_coss:
        if cos < math.cos(Fov):
            break
        else:
            ans_inds.append(cos[0])
        if len(ans_inds) == sub_neighbors:
            break
    if len(ans_inds) == 0:
        ans_inds.append(inds[0][0])
    return np.array([ans_inds])


def user_based_recommend(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test, neighbors,
                         sub_neighbors, bound):
    scores = []
    for data_test_id in range(len(data_test_p)):
    # for data_test_id in range(5, 6, 1):
        print(neighbors, data_test_id)
        # cold start
        # inds = cold_start(data_train_p[:, 1:4], np.array([data_test_p[data_test_id, 1:4]]), neighbors=neighbors)
        inds = cold_start_advanced(data_train_p[:, 1:4], np.array([data_test_p[data_test_id, 1:4]]),
                                   data_train_d[:, 1:4], np.array([data_test_d[data_test_id, 1:4]]),
                                   neighbors=neighbors, sub_neighbors=sub_neighbors)
        # print("inds: ", inds)
        # print(inds.shape)
        set_total_predict_list = set()
        for i in range(len(inds[0])):
            predict_list = target_train[inds[0][i]][1:]
            set_predict_list = set(predict_list)
            similarity = {}
            for j in range(len(target_train)):
                set_target_train_j = set(target_train[j][1:])
                if len(set_predict_list | set_target_train_j) == 0:
                    sim = 0
                else:
                    sim = len(set_predict_list & set_target_train_j) * 1.0 / len(set_predict_list | set_target_train_j)
                similarity[j] = sim
            sort_similarity = sorted(similarity.items(), key=lambda item: item[1], reverse=True)

            for sim in sort_similarity:
                if sim[1] < bound:
                    break
                set_predict_list = set_predict_list | set(target_train[sim[0]][1:])
            set_total_predict_list = set_total_predict_list | set_predict_list

        len_true_positive = len(set_total_predict_list & set(target_test[data_test_id][1:]))
        sim_acc = len_true_positive * 1.0 / len(set_total_predict_list)
        sim_recall = len_true_positive * 1.0 / len(set(target_test[data_test_id][1:]))
        scores.append([sim_acc, sim_recall])
    # csv_file = open('output/statistics_user_based_recommend.csv', 'wb')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['id, sim_acc', 'sim_recall', 'nearest_id', 'predict_len', 'len'])
    # csv_writer.writerows(scores)
    # csv_file.close()
    np_score = np.array(scores)
    mean = np.mean(np_score, axis=0)
    return mean


def main():
    data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test = prepareData.prepare_data()
    neighbors_list = [1, 2, 3, 4, 5, 6]
    score_recommend_by_neighbors = []
    for neighbors in neighbors_list:
        mean = user_based_recommend(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test,
                                    neighbors=15, sub_neighbors=neighbors, bound=0.99)
        print(mean[0], mean[1])
        score_recommend_by_neighbors.append([neighbors, mean[0], mean[1]])
    csv_file = open('output/tmp.csv', 'ab')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['sub_neighbors', 'sim_acc', 'sim_recall'])
    csv_writer.writerows(score_recommend_by_neighbors)
    csv_file.close()
    pass


if __name__ == "__main__":
    main()
