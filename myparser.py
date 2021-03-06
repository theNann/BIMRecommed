import string
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

DATA_ID = -1


def read_data(file_path):
    global DATA_ID
    file_object = open(file_path)
    file_lines = file_object.readlines()

    queries = []
    results = []
    for i in range(0, len(file_lines), 2):
        DATA_ID += 1
        line = file_lines[i]
        start_with = "query: "
        begin = line.find(start_with) + len(start_with)
        end = len(line) - 1
        datas = line[begin:end].strip().split(',')
        query = []
        query.append(DATA_ID)
        for d in datas:
            query.append(float(d))
        queries.append(query)

        line = file_lines[i+1]
        start_with = "["
        end_with = "]"
        begin = line.find(start_with) + len(start_with)
        end = line.find(end_with)
        datas = []
        data_str = line[begin:end].strip()
        if len(data_str) > 0:
            datas = data_str.split(',')
        result = []
        result.append(DATA_ID)
        for d in datas:
            result.append(int(d))
        results.append(result)

    return queries, results


def generate_data(file_paths):
    queries = []
    results = []
    for file_path in file_paths:
        q, r = read_data(file_path)
        queries.extend(q)
        results.extend(r)
    #write file
    # file_object = open("queries.txt", "w+")
    # for q in queries:
    #     file_object.write(str(q)+"\n")
    # file_object.close()
    #
    # file_object = open("results.txt", "w+")
    # for i in range(10):
    #     file_object.write(str(results[i])+"\n")
    # file_object.close()

    return queries, results


def cal_sim_by_set(s1, s2):
    intersection = s1 & s2
    union = s1 | s2
    if len(intersection) == 0:
        return 0
    else:
        return 1.0*len(intersection)/len(union)


def cal_result_similarity(r1, r2):
    set_r1 = set(r1)
    set_r2 = set(r2)
    intersection = set_r1 & set_r2
    union = set_r1 | set_r2
    if len(intersection) == 0:
        return 0
    else:
        return 1.0*len(intersection)/len(union)


# v1 and v2 are row vector
# v1,v2 type is matrix
def cal_vector_similarity(v1, v2):
    num = float(v1 * v2.T)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return cos, sim


# v1 and v2 are row vector
# v1,v2 type is matrix
def cal_euclidean_distance(v1, v2):
    dist = np.linalg.norm(v1-v2)
    sim = 1.0 / (1.0 + dist)
    return dist, sim


def get_pdu_vector(queries):
    queries_by_pdu = []
    for i in range(len(queries)):
        # T*B = view
        query = queries[i]
        view_matrix = np.mat(np.array(query[2:18]).reshape(4, 4))
        b_matrix = view_matrix.copy()
        b_matrix[3, :] = 0
        b_matrix[3, 3] = 1
        t_matrix = view_matrix * b_matrix.getI()
        p_vector = [-t_matrix[3, 0], -t_matrix[3, 1], -t_matrix[3, 2]]
        d_vector = [view_matrix[0, 2], view_matrix[1, 2], view_matrix[2, 2]]
        u_vector = [view_matrix[0, 1], view_matrix[1, 1], view_matrix[2, 1]]
        queries_by_pdu.append([query[0], p_vector[0], p_vector[1], p_vector[2], d_vector[0], d_vector[1], d_vector[2],
                               u_vector[0], u_vector[1], u_vector[2]])
    return queries_by_pdu


def read_data_txt(file_paths):
    queries, results = generate_data(file_paths)
    queries_by_pdu = get_pdu_vector(queries)
    data = np.array(queries_by_pdu)
    target = np.array(results)
    return data, target


def solve_data():
    # file_paths = ["input/DataSet/input1.txt", "input/DataSet/input2.txt", "input/DataSet/input3.txt",
    #               "input/DataSet/input4.txt", "input/DataSet/input5.txt"]
    file_paths = ["/home/pyn/Desktop/DataSet/test5.txt"]
    data, target = read_data_txt(file_paths)
    print(data.shape)
    print(target.shape)

    # csv_file = open('input/queries_by_pdu.csv', 'wb')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['data_id',  'p1', 'p2', 'p3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
    # csv_writer.writerows(queries_by_pdu)
    # csv_file.close()

    # data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=0)
    # print(data_train.shape, data_test.shape)
    # print(target_train.shape, target_test.shape)

    # write csv file
    csv_file = open('/home/pyn/Desktop/DataSet/data5.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['data_id', 'p1', 'p2', 'p3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
    csv_writer.writerows(data)
    csv_file.close()

    txt_file = open('/home/pyn/Desktop/DataSet/target5.txt', 'wb')
    for target_train_i in target:
        tmp = str(target_train_i)
        tmp = tmp[1:len(tmp) - 1]
        txt_file.writelines(tmp + '\n')
    txt_file.close()

    # index = 0
    # for data_test_i in data_test:
    #     data_test_i[0] = index
    #     index = index + 1
    # csv_file = open('input/data_test.csv', 'wb')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['data_id', 'p1', 'p2', 'p3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
    # csv_writer.writerows(data_test)
    # csv_file.close()
    #

    # txt_file = open('input/target_train.txt', 'wb')
    # index = 0
    # for target_train_i in target_train:
    #     target_train_i[0] = index
    #     tmp = str(target_train_i)
    #     tmp = tmp[1:len(tmp) - 1]
    #     txt_file.writelines(tmp + '\n')
    #     index += 1
    # txt_file.close()

    # txt_file = open('input/target_test.txt', 'wb')
    # index = 0
    # for target_test_i in target_test:
    #     target_test_i[0] = index
    #     tmp = str(target_test_i)
    #     tmp = tmp[1:len(tmp) - 1]
    #     txt_file.writelines(tmp + '\n')
    #     index += 1
    # txt_file.close()


def expandTestData():
    data_csv_files = ["/home/pyn/Desktop/DataSet/data1.csv", "/home/pyn/Desktop/DataSet/data2.csv",
                      "/home/pyn/Desktop/DataSet/data3.csv", "/home/pyn/Desktop/DataSet/data4.csv",
                      "/home/pyn/Desktop/DataSet/data5.csv"]
    write_data_file = "/home/pyn/Desktop/BIMRecommed/input/data_test.csv"

    for file in data_csv_files:
        df = pd.read_csv(file)
        df.to_csv(write_data_file, encoding="utf_8", index=False, header=False, mode='a+')
    print("write finish")

    # target_txt_files = ['/home/pyn/Desktop/DataSet/target1.txt', '/home/pyn/Desktop/DataSet/target2.txt',
    #                     '/home/pyn/Desktop/DataSet/target3.txt', '/home/pyn/Desktop/DataSet/target4.txt',
    #                     '/home/pyn/Desktop/DataSet/target5.txt']
    # target = read_target_test(target_txt_files)
    # txt_file = open('/home/pyn/Desktop/BIMRecommed/input/target_test.txt', 'a')
    # index = 2368
    # for target_i in target:
    #     target_i[0] = index
    #     tmp = str(target_i)
    #     tmp = tmp[1:len(tmp) - 1]
    #     txt_file.writelines(tmp + '\n')
    #     index += 1
    # txt_file.close()
    # print("write finish")


def read_target_test(target_txt_files):
    target = []
    for file_path in target_txt_files:
        file_object = open(file_path)
        file_lines = file_object.readlines()
        for line in file_lines:
            data_str = line[0:len(line)-1].strip()
            # print(data_str)
            if len(data_str) > 0:
                data_list = data_str.split(", ")
                data = []
                for d in data_list:
                    data.append(int(d))
                target.append(data)
    return target


def solve_test():
    file_paths = ["input/DataSet/test.txt"]
    queries_by_pdu, results = read_data_txt(file_paths)
    target = np.array(results)
    data = np.array(queries_by_pdu)
    # write csv file
    csv_file = open('input/test_input.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['data_id', 'p1', 'p2', 'p3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
    csv_writer.writerows(data)
    csv_file.close()

    csv_file = open('input/test_output.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['result'])
    csv_writer.writerows(target)
    csv_file.close()
    pass


if __name__ == "__main__":
    # solve_data()
    # solve_test()
    expandTestData()
