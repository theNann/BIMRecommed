import string
import csv
import numpy as np
from sklearn.model_selection import train_test_split

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
    file_paths = ["input/DataSet/input1.txt", "input/DataSet/input2.txt", "input/DataSet/input3.txt",
                  "input/DataSet/input4.txt", "input/DataSet/input5.txt"]
    data, target = read_data_txt(file_paths)

    # csv_file = open('input/queries_by_pdu.csv', 'wb')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['data_id',  'p1', 'p2', 'p3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
    # csv_writer.writerows(queries_by_pdu)
    # csv_file.close()

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=0)

    print(data_train.shape, data_test.shape)
    print(target_train.shape, target_test.shape)

    # return
    # write csv file
    csv_file = open('input/data_train.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['data_id', 'p1', 'p2', 'p3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
    csv_writer.writerows(data_train)
    csv_file.close()

    csv_file = open('input/data_test.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['data_id', 'p1', 'p2', 'p3', 'd1', 'd2', 'd3', 'u1', 'u2', 'u3'])
    csv_writer.writerows(data_test)
    csv_file.close()

    csv_file = open('input/target_train.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['result'])
    csv_writer.writerows(target_train)
    csv_file.close()

    csv_file = open('input/target_test.csv', 'wb')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['result'])
    csv_writer.writerows(target_test)
    csv_file.close()


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
    solve_test()
