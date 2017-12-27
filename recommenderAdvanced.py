import numpy as np
from sklearn.neighbors import NearestNeighbors
import prepareData
import csv
import myparser
import knn


def cold_start(data_train_p, data_test_p, data_train_d, data_test_d, type, neighbors):
    dists, inds = knn.get_nearest_neighbors(data_train_p, data_test_p, neighbors, type)
    return inds


def user_based_recommend(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test, neighbors,
                         bound):
    scores = []
    for data_test_id in range(len(data_test_p)):
        print(neighbors, data_test_id)
        # cold start
        inds = cold_start(data_train_p[:, 1:4], np.array([data_test_p[data_test_id, 1:4]]), data_train_d[:, 1:4],
                          np.array([data_test_d[data_test_id, 1:4]]), type='position', neighbors=neighbors)
        # print("inds: ", inds)
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
    neighbors_list = [3, 5]
    score_recommend_by_neighbors = []
    for neighbors in neighbors_list:
        mean = user_based_recommend(data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test,
                                    neighbors, 0.99)
        score_recommend_by_neighbors.append([neighbors, mean[0], mean[1]])
    csv_file = open('output/tmp.csv', 'ab')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['neighbors', 'sim_acc', 'sim_recall'])
    csv_writer.writerows(score_recommend_by_neighbors)
    csv_file.close()
    pass


if __name__ == "__main__":
    main()
