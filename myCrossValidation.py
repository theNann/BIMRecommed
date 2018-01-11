import numpy as np
from sklearn.model_selection import KFold
import prepareData
import knn
import csv


def k_fold_cross_validation():
    data_train = prepareData.get_data_train()
    data_test = prepareData.get_data_test()
    target_train = prepareData.get_target_train()
    target_test = prepareData.get_target_test()
    # print(data_train.shape, data_test.shape, target_train.shape, target_test.shape)
    data = np.concatenate([data_train, data_test], axis=0)
    target = np.concatenate([target_train, target_test], axis=0)
    kf = KFold(n_splits=14, shuffle=True)

    score_3 = []
    score_5 = []
    score_7 = []
    for train_idx, test_idx in kf.split(data):
        # print("%s %s" % (train_idx, test_idx))
        # print(train_idx.shape, test_idx.shape)
        x_train, x_test, y_train, y_test = data[train_idx], data[test_idx], target[train_idx], target[test_idx]
        # shape are (10778L, 10L), (1198L, 10L), (10778L,), (1198L,)
        # print(x_train, x_test, y_train, y_test)
        x_train_p = x_train[:, [0, 1, 2, 3]]
        x_train_d = x_train[:, [0, 4, 5, 6]]
        x_test_p = x_test[:, [0, 1, 2, 3]]
        x_test_d = x_test[:, [0, 4, 5, 6]]
        k_list = [3, 5, 7]
        for k in k_list:
            print("k : ", k)
            score = knn.knn_2(x_train_p, x_test_p, x_train_d, x_test_d, y_train, y_test, neighbors=k)
            mean = np.mean(np.array(score), axis=0)
            if k == 3:
                score_3.append([k, mean[0], mean[1], mean[2], mean[3]])
            elif k == 5:
                score_5.append([k, mean[0], mean[1], mean[2], mean[3]])
            elif k == 7:
                score_7.append([k, mean[0], mean[1], mean[2], mean[3]])

    csv_file = open('output/tmp_cross_validation.csv', 'ab')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['k', 'score_by_union_acc', 'score_by_union_recall',
                         'score_by_inter_acc', 'score_by_inter_recall'])
    csv_writer.writerows(score_3)
    csv_writer.writerows(score_5)
    csv_writer.writerows(score_7)
    csv_file.close()
    pass


if __name__ == "__main__":
    k_fold_cross_validation()
