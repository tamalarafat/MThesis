# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:27:06 2019

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

def nFakeSamples(n, class1, class2):
    
    test_sample = np.zeros((n, class1.shape[1]))
    
    for j in test_sample:    
        for i in range(class1.shape[1]):
            rand = random.random()
            if rand <= 0.20:
                j[i] = random.uniform(np.amin(class2[:,i]), np.amax(class2[:,i]))
            else:
                j[i] = random.uniform(np.amin(class1[:,i]), np.amax(class1[:,i]))
            
    test_sample = test_sample.reshape(n, -1)
    return test_sample

class_1 = X[[0, 1, 2], :]
class_2 = X[[3 , 4, 5], :]
test_sample = nFakeSamples(2, class_1, class_2)

#creating test sample label
test_sample_label = []
for i in range(1, len(test_sample) + 1):
    test_sample_label.append('T_S_{}'.format(i))
    
sc = StandardScaler()
sc_X = sc.fit_transform(X)
sc_test = sc.transform(test_sample)

combined_data = np.append(sc_X, sc_test, axis = 0)

#plotting the raw data
plt.scatter(sc_X[:,0], sc_X[:, 1])
plt.show()

tsne = (TSNE(n_components = 2, n_iter = 200000, perplexity = 9, init = 'pca', 
             n_iter_without_progress = 300))
tsne_transformed = tsne.fit_transform(combined_data)

X_train = tsne_transformed[:30,:]
X_test = tsne_transformed[30:,:]
#a = tsne.embedding_ #Basically the transformed data

kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 100)

y_tsneData = kmeans.fit_predict(X_train)
y_test = kmeans.predict(X_test)


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
plt.scatter(X_train[:,0], X_train[:,1], c = y_tsneData, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.8)
for i, txt in enumerate(sample_list):
    ax.annotate(txt, (X_train[:,0][i], X_train[:,1][i]))
plt.colorbar(ticks = range(10), label = 'clusters')
plt.title('KMeans')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


plt.rc('font', size = 10)
fig, ax = plt.subplots(figsize = (14,10))
plt.scatter(X_train[:,0], X_train[:,1], c = y_tsneData, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.8)
plt.scatter(X_test[:,0], X_test[:,1], c = y_test, s = 200, alpha = 0.8)
for i, txt in enumerate(sample_list):
    ax.annotate(txt, (X_train[:,0][i], X_train[:,1][i]))    
for j, txt in enumerate(test_sample_label):
    ax.annotate(txt, (X_test[:,0][j], X_test[:,1][j]))   
plt.title('KMeans')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


sample = []
y_tsneData_lst = y_tsneData.tolist()
y_test_lst = y_test.tolist()
for i in range(len(y_tsneData_lst)):
    for j in range(len(y_test_lst)):
        if y_tsneData_lst[i] == y_test_lst[j]:
            sample.append(sample_list[i])
    break

for i in range(len(sample)):
    print('{} similar to {}.'.format(test_sample_label[i], sample[i]))
            
adjusted_rand_score = metrics.adjusted_rand_score(y, y_tsneData)
a = print('Adjusted rand score for KMeans using KernelPCA {}.'.format(adjusted_rand_score))

homogeneity_score, completeness_score, V_score = metrics.homogeneity_completeness_v_measure(y, y_tsneData)
h = print('Homogeneity score for KMeans using KernelPCA {}.'.format(homogeneity_score))
