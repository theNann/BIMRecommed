import numpy as np
from sklearn.neighbors import NearestNeighbors
import prepareData
import csv
import myparser
import knn
import time
OBJECTS = 27025


# data_train_p/data_train_d is np.array(9472L,3L), data_test_p/data_test_d is np.array(1L, 3L)
def cold_start(data_train_p, data_test_p, data_train_d, data_test_d, cold_start_strategy):
    if cold_start_strategy == "position":
        dists, inds = knn.get_nearest_neighbors(data_train_p, data_test_p, 1, cold_start_strategy)
        return inds
    elif cold_start_strategy == 'direction':
        dists, inds = knn.get_nearest_neighbors(data_train_d, data_test_d, 1, cold_start_strategy)
        return inds
    else:
        dists, inds = knn.extract_neighbors_by_sort_direction(data_train_p, data_test_p, data_train_d, data_test_d,
                                                              neighbors=15, sub_neighbors=1)
        return dists, inds


def get_nearest_objects(objects_info, object_test, radius):
    # samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    # neigh = NearestNeighbors(radius=5.0)
    # neigh.fit(samples)
    # rng = neigh.radius_neighbors([[1., 1., 1.]])
    # print(rng)
    # print(np.asarray(rng[0][0]))
    # print(np.asarray(rng[1][0]))
    neigh = NearestNeighbors(radius=radius)
    neigh.fit(objects_info)
    rng = neigh.radius_neighbors(object_test)
    # dists/inds is np.array ,shape is (3L,) like [0 1 2]
    dists = np.asarray(rng[0][0])
    inds = np.asarray(rng[1][0])
    return dists, inds


def top_match(target_train, predict_list, how_many=None, bound=None):
    set_predict_list = set(predict_list)
    similarity = {}
    for i in range(len(target_train)):
        set_target_train_i = set(target_train[i][1:])
        if len(set_predict_list | set_target_train_i) == 0:
            sim = 0
        else:
            sim = len(set_predict_list & set_target_train_i) * 1.0 / len(set_predict_list | set_target_train_i)
        similarity[i] = sim
    sort_similarity = sorted(similarity.items(), key=lambda item: item[1], reverse=True)

    if how_many is not None:
        count_how_many = 0
        for sim in sort_similarity:
            set_predict_list = set_predict_list | set(target_train[sim[0]][1:])
            count_how_many += 1
            if count_how_many == how_many:
                break
    if bound is not None:
        for sim in sort_similarity:
            if sim[1] < bound:
                break
            set_predict_list = set_predict_list | set(target_train[sim[0]][1:])
    return set_predict_list
    pass


def user_based_recommend(target_train, target_test, data_train_p, data_test_p, data_train_d, data_test_d,
                         how_many=None, bound=None):
    # movies_data = {
    #     1: {1: 3.0, 2: 4.0, 3: 3.5, 4: 5.0, 5: 3.0},
    #     2: {1: 3.0, 2: 4.0, 3: 2.0, 4: 3.0, 5: 3.0, 6: 2.0},
    #     3: {2: 3.5, 3: 2.5, 4: 4.0, 5: 4.5, 6: 3.0},
    #     4: {1: 2.5, 2: 3.5, 3: 2.5, 4: 3.5, 5: 3.0, 6: 3.0},
    #     5: {2: 4.5, 3: 1.0, 4: 4.0, 8: 2.4, 10: 4.5},
    #     6: {1: 3.0, 2: 3.5, 3: 3.5, 4: 5.0, 5: 3.0, 6: 1.5},
    #     10: {1: 2.5, 2: 3.0, 4: 3.5, 5: 4.0}
    # }
    # model = MatrixPreferenceDataModel(movies_data)
    # print(model.index.shape)
    # print(model.index)
    # similarity = UserSimilarity(model, pearson_correlation)
    # sim = similarity.get_similarities(5)
    # print(type(sim))
    # print(sim)

    # print(type(data_train_p[:, 1:4]), data_train_p[:, 1:4].shape, type(np.array([data_test_p[0, 1:4]])), np.array([data_test_p[0, 1:4]]).shape)
    # print(np.array([data_test_p[0, 1:4]]))
    dists, inds = cold_start(data_train_p[:, 1:4], data_test_p[:, 1:4], data_train_d[:, 1:4], data_test_d[:, 1:4],
                             cold_start_strategy='position_direction')
    scores = []
    statistics = []
    for data_test_id in range(len(data_test_p)):
        print(how_many, data_test_id)
        time1 = time.clock()
        predict_list = target_train[inds[data_test_id][0]][1:]
        set_predict_list = top_match(target_train, predict_list, how_many, bound)
        len_true_positive = len(set_predict_list & set(target_test[data_test_id][1:]))
        sim_acc = len_true_positive*1.0 / len(set_predict_list)
        sim_recall = len_true_positive*1.0 / len(set(target_test[data_test_id][1:]))
        statistics.append([data_test_id, sim_acc, sim_recall, len(set_predict_list), len(target_test[data_test_id][1:]),
                           len_true_positive])
        scores.append([sim_acc, sim_recall])
        print(data_test_id, time.clock()-time1)
    # csv_file = open('output/statistics_user_based_recommend.csv', 'wb')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['id, sim_acc', 'sim_recall', 'nearest_id', 'predict_len', 'len'])
    # csv_writer.writerows(scores)
    # csv_file.close()
    np_score = np.array(scores)
    mean = np.mean(np_score, axis=0)
    return mean, statistics


def content_based_recommend(target_train, target_test, data_train_p, data_test_p, radius):
    scores = []
    for data_test_id in range(len(data_test_p)):
        print(radius, data_test_id)
        # get predist_list
        dists, inds = knn.get_nearest_neighbors(data_train_p[:, 1:4], np.array([data_test_p[data_test_id, 1:4]]), 1, type="position")
        predict_list = target_train[inds[0][0]][1:]
        set_predict_list = set(predict_list)
        print("len: ", len(set_predict_list))

        # content_based_recommend
        # np.array  shape is (27025,3L)
        objects_info = prepareData.get_objects_info()
        for predict_id in predict_list:
            object_test = np.array([objects_info[predict_id]])
            dists, inds = get_nearest_objects(objects_info, object_test, radius=radius)
            set_predict_list = set_predict_list | set(inds)
        print("len: ", len(set_predict_list))

        sim_acc = len(set_predict_list & set(target_test[data_test_id][1:])) * 1.0 / len(set_predict_list)
        sim_recall = len(set_predict_list & set(target_test[data_test_id][1:])) * 1.0 / len(set(target_test[data_test_id][1:]))
        scores.append([sim_acc, sim_recall])

    np_score = np.array(scores)
    mean = np.mean(np_score, axis=0)
    return mean


def main():
    # target_train = prepareData.get_target_train()
    # target_test = prepareData.get_target_test()
    # data_train_p = prepareData.get_data_train_p()
    # data_test_p = prepareData.get_data_test_p()
    # data_train_d = prepareData.get_data_train_d()
    # data_test_d = prepareData.get_data_test_d()
    data_train_p, data_test_p, data_train_d, data_test_d, target_train, target_test = prepareData.prepare_data()
    test_input_p, test_input_d, test_input, test_output = prepareData.prepare_data_test()
    # radiuss = [0.3]
    # scores_recommend_by_radius = []
    # for radius in radiuss:
    #     mean = content_based_recommend(target_train, target_test, data_train_p, data_test_p, radius=radius)
    #     scores_recommend_by_radius.append([radius, mean[0], mean[1]])
    # csv_file = open('output/scores_recommend.csv', 'ab')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['radius', 'sim_acc', 'sim_recall'])
    # csv_writer.writerows(scores_recommend_by_radius)
    # csv_file.close()

    how_many = 6
    mean, statistics = user_based_recommend(target_train, test_output, data_train_p, test_input_p, data_train_d,
                                            test_input_d, how_many=how_many)
    csv_file = open('output/tmp_recommender_2.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['id', 'sim_acc', 'sim_recall', 'len_predict', 'len_output', 'len_true_positive'])
    csv_writer.writerows(statistics)
    csv_file.close()


if __name__ == '__main__':
    main()
