# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 01:51:42 2019

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

#plotting the raw data
plt.scatter(sc_X[:,0], sc_X[:, 1])
plt.show()

kpca = KernelPCA(n_components = 3, kernel = 'rbf')
X_train = kpca.fit_transform(sc_X)
X_test = kpca.transform(sc_test)

# =============================================================================
# #Eigenvalues of the centered kernel matrix in decreasing order. If n_components and 
# #remove_zero_eig are not set, then all values are stored.
# l = kpca.lambdas_ #(n_components,)
# 
# #Eigenvectors of the centered kernel matrix. If n_components and remove_zero_eig are not set, 
# #then all components are stored.
# a = kpca.alphas_ #array, (n_samples, n_components)
# 
# #Inverse transform matrix. Only available when fit_inverse_transform is True.
# l1 = kpca.dual_coef_ #array, (n_samples, n_features)
# 
# #The data used to fit the model. If copy_X=False, then X_fit_ is a reference. 
# #This attribute is used for the calls to transform.
# a2 = kpca.X_fit_ # (n_samples, n_features), basically the main data(here sc_X)
# 
# #Projection of the fitted data on the kernel principal components. Only available 
# #when fit_inverse_transform is True.
# #This is the transformed data by KPCA
# a1 = kpca.X_transformed_fit_ #array, (n_samples, n_components)
# 
# 
# a3 = kpca.inverse_transform(X_train)
# =============================================================================

kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 100)

y_kpcaData = kmeans.fit_predict(X_train)
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
plt.scatter(X_train[:,0], X_train[:,1], c = y_kpcaData, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.8)
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
plt.scatter(X_test[:,0], X_test[:,1], c = y_test, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.8)
for i, txt in enumerate(sample_list):
    ax.annotate(txt, (X_train[:,0][i], X_train[:,1][i]))    
for j, txt in enumerate(test_sample_label):
    ax.annotate(txt, (X_test[:,0][j], X_test[:,1][j]))   
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