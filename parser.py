import os
import sys
import string
import csv
from numpy import *


def read_data(file_path):
    file_object = open(file_path)
    file_lines = file_object.readlines()

    queries = []
    results = []
    for i in range(0, len(file_lines), 2):
        line = file_lines[i]
        start_with = "query:"
        begin = line.find(start_with) + len(start_with)
        end = len(line) - 1
        datas = line[begin:end].strip().split(',')
        query = []
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
        for d in datas:
            result.append(int(d))
        results.append(result)

    return queries, results


def generate_data():
    file_paths = ["input/TestData/2016-01-27.txt", "input/TestData/2016-02-25.txt", "input/TestData/full2half.txt",
                  "input/TestData/full2half2.0.txt", "input/TestData/inandturn.txt", "input/TestData/inandturn2.0.txt",
                  "input/TestData/out2in.txt"]
    queries = []
    results = []
    for file_path in file_paths:
        q, r = read_data(file_path)
        queries.extend(q)
        results.extend(r)

    # write file
    # file_object = open("queries.txt", "w+")
    # for q in queries:
    #     file_object.write(str(q)+"\n")
    # file_object.close()
    #
    # file_object = open("results.txt", "w+")
    # for r in results:
    #     file_object.write(str(r)+"\n")
    # file_object.close()

    return queries, results


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
    denom = linalg.norm(v1) * linalg.norm(v2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def cal_euclidean_distance(v1, v2):
    dist = linalg.norm(v1-v2)
    sim = 1.0 / (1.0 + dist)
    return sim


def get_pdu_vector(queries):
    queries_by_pdu = []
    for i in range(len(queries)):
        # T*B = view
        query = queries[i]
        view_matrix = mat(array(query[4:20]).reshape(4, 4))
        b_matrix = view_matrix.copy()
        b_matrix[3, :] = 0
        b_matrix[3, 3] = 1
        t_matrix = view_matrix * b_matrix.getI()
        p_vector = [-t_matrix[3, 0], -t_matrix[3, 1], -t_matrix[3, 2]]
        d_vector = [view_matrix[0, 2], view_matrix[1, 2], view_matrix[2, 2]]
        u_vector = [view_matrix[0, 1], view_matrix[1, 1], view_matrix[2, 1]]
        queries_by_pdu.append([p_vector, d_vector, u_vector])
    return queries_by_pdu


def solve_data():
    queries, results = generate_data()
    similarities = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            print("i : ", i, "j : ", j)
            sim = cal_result_similarity(results[i], results[j])
            similarities.append((i, j, sim))

    # write csv file
    # csv_file = open('input/similarities.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['query_i', 'query_j', 'similarity'])
    # csv_writer.writerows(similarities)
    # csv_file.close()
    #
    # csv_file = open('input/queries.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['mode', 'fov', 'near', 'far', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
    #                      '12', '13', '14', '15'])
    # csv_writer.writerows(queries)
    # csv_file.close()
    #
    # csv_file = open('input/results.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['result'])
    # csv_writer.writerows(results)
    # csv_file.close()

    queries_by_pdu = get_pdu_vector(queries)
    simlarities_by_pdu = []
    for i in range(len(similarities)):
        query_i = similarities[i][0]
        query_j = similarities[i][1]
        pdu_i = queries_by_pdu[query_i]
        pdu_j = queries_by_pdu[query_j]
        p_sim = cal_euclidean_distance(mat(pdu_i[0]), mat(pdu_j[0]))
        d_sim = cal_vector_similarity(mat(pdu_i[1]), mat(pdu_j[1]))
        u_sim = cal_vector_similarity(mat(pdu_i[2]), mat(pdu_j[2]))
        simlarities_by_pdu.append((p_sim, d_sim, u_sim, similarities[i][2]))
    # csv_file = open('input/simlarities_by_pdu.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['p_sim', 'd_sim', 'u_sim', 'sim'])
    # csv_writer.writerows(simlarities_by_pdu)
    # csv_file.close()


if __name__ == "__main__":
    solve_data()
