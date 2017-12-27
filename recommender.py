import numpy as np
from sklearn.neighbors import NearestNeighbors
import prepareData
import csv
import myparser
import knn
OBJECTS = 27025


# data_train_p/data_train_d is np.array(9472L,3L), data_test_p/data_test_d is np.array(1L, 3L)
def cold_start(data_train_p, data_test_p, data_train_d, data_test_d, type, neighbors):
    if type == "position":
        dists, inds = knn.get_nearest_neighbors(data_train_p, data_test_p, 1, type)
        return inds
    elif type == 'direction':
        dists, inds = knn.get_nearest_neighbors(data_train_d, data_test_d, 1, type)
        return inds
    else:
        dists_p, inds_p = knn.get_nearest_neighbors(data_train_p, data_test_p, neighbors, 'position')
        d_test = data_test_d[0]
        max_sim = 0
        max_sim_id = 0
        for index in inds_p[0]:
            # print('neighbors:')
            d_train = data_train_d[index]
            cos, sim = myparser.cal_vector_similarity(np.mat(d_test), np.mat(d_train))
            # print("index, sim: ", index, sim)
            if sim > max_sim:
                max_sim = sim
                max_sim_id = index
        return np.array([np.array([max_sim_id])])
    pass


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


def user_based_recommend(target_train, target_test, data_train_p, data_test_p, data_train_d, data_test_d, type,
                         neighbors, bound):
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

    scores = []
    for data_test_id in range(len(data_test_p)):
    # for data_test_id in range(567, 568, 1):
        print(neighbors, data_test_id)
        # cold start
        inds = cold_start(data_train_p[:, 1:4], np.array([data_test_p[data_test_id, 1:4]]), data_train_d[:, 1:4],
                          np.array([data_test_d[data_test_id, 1:4]]), type=type, neighbors=neighbors)
        # print("inds: ", inds)
        predict_list = target_train[inds[0][0]][1:]
        set_predict_list = set(predict_list)
        similarity = {}
        for i in range(len(target_train)):
            set_target_train_i = set(target_train[i][1:])
            if len(set_predict_list | set_target_train_i) == 0:
                sim = 0
            else:
                sim = len(set_predict_list & set_target_train_i)*1.0 / len(set_predict_list | set_target_train_i)
            similarity[i] = sim
        sort_similarity = sorted(similarity.items(), key=lambda item: item[1], reverse=True)

        for sim in sort_similarity:
            if sim[1] < bound:
                break
            set_predict_list = set_predict_list | set(target_train[sim[0]][1:])
        len_true_positive = len(set_predict_list & set(target_test[data_test_id][1:]))
        sim_acc = len_true_positive*1.0 / len(set_predict_list)
        sim_recall = len_true_positive*1.0 / len(set(target_test[data_test_id][1:]))
        scores.append([data_test_id, sim_acc, sim_recall, inds[0][0], len(set_predict_list), len(set(target_test[data_test_id][1:]))])
        print(sim_acc, sim_recall)
    # csv_file = open('output/statistics_user_based_recommend.csv', 'wb')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['id, sim_acc', 'sim_recall', 'nearest_id', 'predict_len', 'len'])
    # csv_writer.writerows(scores)
    # csv_file.close()
    # np_score = np.array(scores)
    # mean = np.mean(np_score, axis=0)
    # return mean


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

    bounds_list = [0.99]
    scores_recommend_by_bound = []
    for bounds in bounds_list:
        user_based_recommend(target_train, target_test, data_train_p, data_test_p, data_train_d, data_test_d,
                             type="position_direction", neighbors=15, bound=bounds)
        # scores_recommend_by_bound.append([bounds, mean[0], mean[1]])
    # csv_file = open('output/tmp.csv', 'ab')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['bounds', 'sim_acc', 'sim_recall'])
    # csv_writer.writerows(scores_recommend_by_bound)
    # csv_file.close()


if __name__ == '__main__':
    main()
