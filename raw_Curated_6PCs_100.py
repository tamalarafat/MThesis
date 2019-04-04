# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:22:02 2019

@author: Tamal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
# =============================================================================
# Lets Import whatever and whatever I want
# =============================================================================

data = pd.read_csv('curated_dataset_6PCs_100.csv')

# =============================================================================
# Lets put the data as Array to make calculation less complicated
# =============================================================================

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

sample_list = sorted(['L1T1', 'L1T2','L1T3', 'L1T4', 'L3T2', 'L3T3', 'L3T4', 'L5T3', 'L5T4', 'L7T4'] * 3)

# =============================================================================
# Lets plot some variabels from the data
# =============================================================================

def plotRawData(input_data, x_cor, y_cor):
    """data has to be an array"""
    plt.figure()
    plt.scatter(input_data[:, x_cor], input_data[:, y_cor])
    plt.xlabel('any index/gene')
    plt.ylabel('any index/gene')
    plt.title('Plotting data')
    return plt.show()

#remember to give input as an array
plotRawData(X, 4, 200)

# =============================================================================
# Lets Scale the data
# =============================================================================

SC = StandardScaler()
scaled_data = SC.fit_transform(X)

train_data, test_data, y_train, y_test = train_test_split(scaled_data, y, test_size = 0)

sample_label = []
for i in y_train:
    if i == 0:
        sample_label.append('L1T1')
    elif i == 1:
        sample_label.append('L1T2')
    elif i == 2:
        sample_label.append('L1T3')
    elif i == 3:
        sample_label.append('L1T4')
    elif i == 4:
        sample_label.append('L3T2')
    elif i == 5:
        sample_label.append('L3T3')
    elif i == 6:
        sample_label.append('L3T4')
    elif i == 7:
        sample_label.append('L5T3')
    elif i == 8:
        sample_label.append('L5T4')
    elif i == 9:
        sample_label.append('L7T4')
    
plotRawData(scaled_data, 4, 200)
# =============================================================================
# How about t-SNE now?
# =============================================================================
pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
axes_pos = np.arange(len(pcs))
# =============================================================================
# 
# components = [4,5,6,7]
# 
# fig, ax = plt.subplots(2, 2, figsize=(14, 10))
# ax = np.ravel(ax)
# fig.subplots_adjust(hspace = 0.01, wspace = 0.1)
# for i in range(4):
#     Kmeans_opt = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 100)
#     kpca = KernelPCA(n_components = components[i], kernel = 'rbf')
#     kpca_transformed = kpca.fit_transform(train_data)
#     y_kmeans_opt = Kmeans_opt.fit_predict(kpca_transformed)
#     
#     ax[i].scatter(kpca_transformed[:, 0], kpca_transformed[:, 1], c = y_kmeans_opt, s = 30, cmap='tab10')
#     ax[i].title.set_text('KPCA with {} PCs'.format(components[i]))
# #plt.savefig('KPCA_26to29PCS_6pcs_100.png')   
# plt.tight_layout()  
# 
# =============================================================================
kpca = KernelPCA(n_components = 5, kernel = 'rbf')
kpca_transformed = kpca.fit_transform(train_data)

kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 1000)
y_pred = kmeans.fit_predict(kpca_transformed)

fig, ax = plt.subplots(figsize = (12,8))
plt.rc('font', size = 8)
plt.scatter(kpca_transformed[:,0], kpca_transformed[:,1], c = y_train, s = 200, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KPCA transformed data')
plt.colorbar()
for i, txt in enumerate(sample_label):
    ax.annotate(txt, (kpca_transformed[:,0][i], kpca_transformed[:,1][i]))
plt.savefig('2. KPCA_Transformed_with5Pcs_from6PCAs_100genes.png')
plt.show()
    
fig, ax = plt.subplots(figsize = (12,8))
plt.rc('font', size = 8)
plt.scatter(kpca_transformed[:,0], kpca_transformed[:,1], c = y_pred, s = 200, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KPCA transformed clustered data')
plt.colorbar()
for i, txt in enumerate(sample_label):
    ax.annotate(txt, (kpca_transformed[:,0][i], kpca_transformed[:,1][i]))
plt.savefig('2. KPCA_KMeans_with5Pcs_from6PCAs_100genes.png')
plt.show()

fig = plt.figure(figsize = (20, 10))

ax1 = plt.subplot(1, 2, 1)
plt.rc('font', size = 8)
plt.scatter(kpca_transformed[:,0], kpca_transformed[:,1], c = y_train, s = 200, cmap = 'tab10', alpha = 0.9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KPCA transformed data')
for i, txt in enumerate(sample_label):
    ax1.annotate(txt, (kpca_transformed[:,0][i], kpca_transformed[:,1][i]))
    
ax2 = plt.subplot(1, 2, 2)
plt.rc('font', size = 8)
plt.scatter(kpca_transformed[:,0], kpca_transformed[:,1], c = y_pred, s = 200, cmap = 'tab10', alpha = 0.9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KPCA transformed clustered data')
for i, txt in enumerate(sample_label):
    ax2.annotate(txt, (kpca_transformed[:,0][i], kpca_transformed[:,1][i]))
    
fig.suptitle('KernelPCA with 5PCs from 6 PCAs and 600 genes')
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
plt.tight_layout()
fig.savefig('2. KPCA_with6Pcs_from6PCAs_100genes.png')
plt.show()


tsne = TSNE(n_components = 2, n_iter = 20000, perplexity = 4, init = 'pca')
tsne_transformed = tsne.fit_transform(train_data)

fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(tsne_transformed[:,0], tsne_transformed[:,1], c = y_train, s = 200, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('TSNE transformed data')
plt.colorbar()
for i, txt in enumerate(sample_label):
    ax.annotate(txt, (tsne_transformed[:,0][i], tsne_transformed[:,1][i]))
plt.savefig('2. TSNE_Transformed_with2Pcs_from6PCAs_100genes.png')
plt.show()
    
kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 1000)
y_pred = kmeans.fit_predict(tsne_transformed)
    
fig, ax = plt.subplots(figsize = (12,8))
plt.scatter(tsne_transformed[:,0], tsne_transformed[:,1], c = y_pred, s = 200, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('TSNE transformed clustered data')
plt.colorbar()
for i, txt in enumerate(sample_label):
    ax.annotate(txt, (tsne_transformed[:,0][i], tsne_transformed[:,1][i]))
plt.savefig('2. TSNE_clustered_with2Pcs_from6PCAs_100genes.png')
plt.show()

fig = plt.figure(figsize = (20, 10))

ax1 = plt.subplot(1, 2, 1)
plt.rc('font', size = 8)
plt.scatter(tsne_transformed[:,0], tsne_transformed[:,1], c = y_train, s = 200, cmap = 'tab10', alpha = 0.9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('TSNE transformed data')
for i, txt in enumerate(sample_label):
    ax1.annotate(txt, (tsne_transformed[:,0][i], tsne_transformed[:,1][i]))
    
ax2 = plt.subplot(1, 2, 2)
plt.rc('font', size = 8)
plt.scatter(tsne_transformed[:,0], tsne_transformed[:,1], c = y_pred, s = 200, cmap = 'tab10', alpha = 0.9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('TSNE transformed clustered data')
for i, txt in enumerate(sample_label):
    ax2.annotate(txt, (tsne_transformed[:,0][i], tsne_transformed[:,1][i]))
    
fig.suptitle('TSNE with 2PCs from 6 PCAs and 600 genes')
fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
plt.tight_layout()
fig.savefig('2. TSNE_with2Pcs_from6PCAs_100genes.png')
plt.show()

