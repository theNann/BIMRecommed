import os
import sys
import string
import csv


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
    file_paths = ["TestData/2016-01-27.txt", "TestData/2016-02-25.txt", "TestData/full2half.txt",
                  "TestData/full2half2.0.txt", "TestData/inandturn.txt", "TestData/inandturn2.0.txt",
                  "TestData/out2in.txt"]
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


def cal_similarity(r1, r2):
    set_r1 = set(r1)
    set_r2 = set(r2)
    intersection = set_r1 & set_r2
    union = set_r1 | set_r2
    if len(intersection) == 0:
        return 0
    else:
        return 1.0*len(intersection)/len(union)


def solve_data():
    queries, results = generate_data()
    similarities = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            # print("i : ", i, "j : ", j)
            sim = cal_similarity(results[i], results[j])
            similarities.append((i, j, sim))

    # write csv file
    # csv_file = open('similarities.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['query_i', 'query_j', 'similarity'])
    # csv_writer.writerows(similarities)
    # csv_file.close()

    # csv_file = open('queries.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['mode', 'fov', 'near', 'far', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
    #                      '12', '13', '14', '15'])
    # csv_writer.writerows(queries)
    # csv_file.close()

    # csv_file = open('results.csv', 'w', newline='')
    # csv_writer = csv.writer(csv_file)
    # csv_writer.writerow(['result'])
    # csv_writer.writerows(results)
    # csv_file.close()
    return queries, results, similarities


if __name__ == "__main__":
    solve_data()
