import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import myparser
import csv
cos = {}
cos[0] = (0,0)
cos[2] = (2,2)
cos[1] = (1,1)
cos[3] = (2,3)
sort_cos = sorted(cos.items(), key=lambda item: (item[1][0], -item[1][1]), reverse=True)
print(sort_cos[0][0])
# samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
# neigh = NearestNeighbors(radius=5.0)
# neigh.fit(samples)
# rng = neigh.radius_neighbors([[1., 1., 1.]])
# print(rng)
# print(np.asarray(rng[0][0]))
# print(np.asarray(rng[1][0]))
# dists =np.asarray(rng[0][0])
# inds = np.asarray(rng[1][0])
# print(len(dists), inds.shape)
# print(dists[0])
# print(inds[0])

# arr1 = np.array([[4,5]])
# print(type(arr1), arr1.shape)
# from sklearn.model_selection import train_test_split
# x, y = np.arange(10).reshape(5, 2), range(5)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# print(X_train)
# print(X_test)

# samples = [[1, 0, 0], [1, 1, 0], [1, -1, 0], [0, 1, 0], [0, -1, 0], [-1, 1, 0], [-1, -1, 0], [-1, 0, 0]]
# neigh = NearestNeighbors(radius=1-math.cos(math.pi/3), metric='cosine', algorithm='brute')
# # neigh = NearestNeighbors(n_neighbors=8, algorithm='brute', metric='cosine')
# neigh.fit(samples)
# rng = neigh.radius_neighbors([[1, 0, 0]])
# dists = np.asarray(rng[0])
# inds = np.asarray(rng[1])
# print(type(dists), type(inds))
# print(dists)
# print(inds)

# for i in range(len(dists)):
#     print(i, ":")
#     for j in range(len(dists[i])):
#         print(dists[i][j])
#
# for i in range(len(inds)):
#     print(i, ":")
#     for j in range(len(inds[i])):
#         print(inds[i][j])


# dists, inds = neigh.kneighbors([[1, 0, 0]], return_distance= True)
# print(type(dists), type(inds))
# print(dists)
# print(inds)