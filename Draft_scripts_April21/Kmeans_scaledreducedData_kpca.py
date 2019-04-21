# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 00:10:37 2019

@author: Tamal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from scipy import stats
import random
from sklearn import metrics
plt.rc('font', size = 8)
sns.set_style('darkgrid')
sns.set_palette(sns.color_palette('Paired'))
sns.set()

data = pd.read_csv('curated_dataset_6PCs_200.csv')

# =============================================================================
# Lets put the data as Array to make calculation less complicated
# =============================================================================

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

sample_list = sorted(['L1T1', 'L1T2','L1T3', 'L1T4', 'L3T2', 'L3T3', 'L3T4', 'L5T3', 'L5T4', 'L7T4'] * 3)

sc = StandardScaler()
sc_X = sc.fit_transform(X)

#plotting the raw data
plt.scatter(sc_X[:,0], sc_X[:, 1])
plt.show()

kpca = KernelPCA(n_components = 3, kernel = 'rbf')
X_train = kpca.fit_transform(sc_X)

kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 100)
y_kpcaData = kmeans.fit_predict(X_train)
X_kpca = kmeans.fit_transform(X_train)


plt.rc('font', size = 10)
fig, ax = plt.subplots(figsize = (14,10))
plt.scatter(X_train[:,0], X_train[:,1], c = y, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.8)
for i, txt in enumerate(sample_list):
    ax.annotate(txt, (X_train[:,0][i], X_train[:,1][i]))
plt.colorbar(ticks = range(10), label = 'clusters')
plt.title('KMeans')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

plt.rc('font', size = 10)
fig, ax = plt.subplots(figsize = (14,10))
plt.scatter(X_train[:,0], X_train[:,1], c = y_kpcaData, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.8)
for i, txt in enumerate(sample_list):
    ax.annotate(txt, (X_train[:,0][i], X_train[:,1][i]))
plt.colorbar(ticks = range(10), label = 'clusters')
plt.title('KMeans')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


adjusted_rand_score = metrics.adjusted_rand_score(y, y_kpcaData)
a = print('Adjusted rand score for KMeans using KernelPCA {}.'.format(adjusted_rand_score))

homogeneity_score, completeness_score, V_score = metrics.homogeneity_completeness_v_measure(y, y_kpcaData)
h = print('Homogeneity score for KMeans using KernelPCA {}.'.format(homogeneity_score))
c = print('Completeness info score for KMeans using KernelPCA {}.'.format(completeness_score))