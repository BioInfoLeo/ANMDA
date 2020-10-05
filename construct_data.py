import xlrd
import numpy as np
from sklearn.cluster import KMeans
import random
import pandas as pd

associations = np.zeros((383, 495))
semantic_similarity = np.zeros((383, 383))
semantic_weight = np.zeros((383, 383))
functional_similarity = np.zeros((495, 495))
functional_weight = np.zeros((495, 495))

associations_number = xlrd.open_workbook('data/disease-miRNA_associations_number.xlsx')
associations_number = associations_number.sheets()[0]
for i in range(5430):
    an = associations_number.row_values(i)
    m = int(an[0])
    n = int(an[1])
    associations[n - 1][m - 1] = 1

semantic_similarity_1 = xlrd.open_workbook('data/semantic_similarity_1.xlsx')
semantic_similarity_2 = xlrd.open_workbook('data/semantic_similarity_2.xlsx')
semantic_similarity_1 = semantic_similarity_1.sheets()[0]
semantic_similarity_2 = semantic_similarity_2.sheets()[0]
for i in range(383):
    for j in range(383):
        s1 = semantic_similarity_1.row_values(i)[j]
        s2 = semantic_similarity_2.row_values(i)[j]
        semantic_similarity[i][j] = float(s1 + s2) / 2

semantic_similarity_weight = xlrd.open_workbook('data/semantic_similarity_weight.xlsx')
semantic_similarity_weight = semantic_similarity_weight.sheets()[0]
for i in range(383):
    for j in range(383):
        sw = semantic_similarity_weight.row_values(i)
        semantic_weight[i][j] = sw[j]

functional_similarity_1 = xlrd.open_workbook('data/functional_similarity.xlsx')
functional_similarity_1 = functional_similarity_1.sheets()[0]
for i in range(495):
    for j in range(495):
        fs = functional_similarity_1.row_values(i)
        functional_similarity[i][j] = fs[j]

functional_similarity_weight = xlrd.open_workbook('data/functional_similarity_weight.xlsx')
functional_similarity_weight = functional_similarity_weight.sheets()[0]
for i in range(495):
    for j in range(495):
        fw = functional_similarity_weight.row_values(i)
        functional_weight[i][j] = fw[j]

Kernel_D = np.zeros((383, 383))
Disease = np.zeros((383, 383))
Kernel_M = np.zeros((495, 495))
MiRNA = np.zeros((495, 495))

A = np.asmatrix(associations)

gama_d = 383 / (np.linalg.norm(A, 'fro') ** 2)
kernel_d = np.mat(np.zeros((383, 383)))
A_D = A * A.T
for i in range(383):
    for j in range(i, 383):
        kernel_d[j, i] = np.exp(-gama_d * (A_D[i, i] + A_D[j, j] - 2 * A_D[i, j]))

kernel_d = kernel_d + kernel_d.T - np.diag(np.diag(kernel_d))
Kernel_D = np.asarray(kernel_d)
Disease = np.multiply(semantic_similarity, semantic_weight) + np.multiply(Kernel_D, (1 - semantic_weight))
Disease = np.asarray(Disease)

gama_m = 495 / (np.linalg.norm(A, 'fro') ** 2)
kernel_m = np.mat(np.zeros((495, 495)))
A_M = A.T * A

for i in range(495):
    for j in range(i, 495):
        kernel_m[i, j] = np.exp(-gama_m * (A_M[i, i] + A_M[j, j] - 2 * A_M[i, j]))

kernel_m = kernel_m + kernel_m.T - np.diag(np.diag(kernel_m))
Kernel_M = np.asarray(kernel_m)
MiRNA = np.multiply(functional_similarity, functional_weight) + np.multiply(Kernel_M, (1 - functional_weight))
MiRNA = np.asarray(MiRNA)

unknown = []
known = []
for x in range(383):
    for y in range(495):
        if associations[x][y] == 0:
            unknown.append((x, y))
        else:
            known.append((x, y))

unknown_samples = []
for i in range(184155):
    sample = Disease[unknown[i][0], :].tolist() + MiRNA[unknown[i][1], :].tolist()
    unknown_samples.append(sample)

kmeans = KMeans(n_clusters=23, random_state=0).fit(unknown_samples)
centers = kmeans.cluster_centers_
center_x = []
center_y = []

for i in range(len(centers)):
    center_x.append(centers[i][0])
    center_y.append(centers[i][1])

labels = kmeans.labels_

pairs = [[] for i in range(23)]
for i in range(len(labels)):
    pairs[labels[i]].append((unknown[i][0], unknown[i][1]))

random_pairs = [[] for i in range(23)]
for i in range(23):
    random_pairs[i] = random.sample(pairs[i], 240)

data = []
for i in range(23):
    data += random_pairs[i]

for i in range(383):
    for j in range(495):
        if associations[i][j] == 1:
            data.append((i, j))

data_csv = []
target = []
for pair in data:
    feature = Disease[pair[0], :].tolist() + MiRNA[pair[1], :].tolist()
    data_csv.append(feature)
    if (pair[0], pair[1]) in known:
        target.append(1)
    else:
        target.append(0)

pd.DataFrame(data=data_csv).to_csv('data.csv')
pd.DataFrame(data=target).to_csv('target.csv')
