# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:57:29 2019

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

dataset_read = pd.read_excel('pilot_experiment_TPM_WTonly.xlsx')

data = pd.read_csv('Mydata.csv')
data = data.loc[:,(data != 0).any(axis = 0)]
unprocessed_dataset = data

y = sorted(list(range(10)) *3)
sample_list = sorted(['L1T1', 'L1T2','L1T3', 'L1T4', 'L3T2', 'L3T3', 'L3T4', 'L5T3', 'L5T4', 'L7T4'] * 3)
sample_arr = np.array(sample_list)

unprocessed_dataset['y'] = y
dataset = unprocessed_dataset
X = dataset.iloc[:, :-1].values 
y_true = dataset.iloc[:, -1].values

#plotting the raw data
plt.scatter(data['gene_1'], data['gene_29451'])
plt.show()

kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 100)
y_rawData = kmeans.fit_predict(X)

adjusted_rand_score = metrics.adjusted_rand_score(y_true, y_rawData)
a = print('Adjusted rand score for KMeans using KernelPCA {}.'.format(adjusted_rand_score))

homogeneity_score, completeness_score, V_score = metrics.homogeneity_completeness_v_measure(y_true, y_rawData)
h = print('Homogeneity score for KMeans using KernelPCA {}.'.format(homogeneity_score))
c = print('Completeness info score for KMeans using KernelPCA {}.'.format(completeness_score))