# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:51:15 2019

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

data = pd.read_csv('curated_dataset.csv')

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

def dataTransformation(input_data):
    """output: Scaled, train_data, y_train, sample_label
    The input_data is the raw data that will be transformed.
    Step_1: Scaling the data or Normalization
    step_2: Shuffling the normalized data to break the order of the samples"""
    SC = StandardScaler()
    scaled_data = SC.fit_transform(X)
    
    train_data, test_data, y_train, y_test = train_test_split(scaled_data, y, test_size = 0)
      
    return scaled_data, train_data, y_train

scaled_data, train_data, y_train = dataTransformation(X)

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

def dataReduction(trained_data):
    """returns KPCA, TSNE transformed reduced dimension
    Please provide the input data as scaled/normalized data/train_data in case 
    you have performed train_test_split on the normalized data"""
    kpca = KernelPCA(n_components = 3, kernel = 'rbf')
    kpca_transformed = kpca.fit_transform(trained_data)
    
    tsne = TSNE(n_components = 2, n_iter = 20000, perplexity = 9, init = 'pca')
    tsne_transformed = tsne.fit_transform(trained_data)
    
    return kpca_transformed, tsne_transformed

def kMeans(reduced_data):
    
    kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 1000)
    y_pred = kmeans.fit_predict(reduced_data)
    
    return y_pred

def pcsVisualization(reduced_data, y_val, pc1_ind, pc2_ind):
    
    fig, ax = plt.subplots(figsize = (12,8))
    plt.scatter(reduced_data[:,pc1_ind], reduced_data[:,pc2_ind], c = y_val, s = 200, cmap = 'tab10', alpha = 0.7)
    plt.xlabel(pcs[pc1_ind])
    plt.ylabel(pcs[pc2_ind])
    plt.title(pcs[pc1_ind] + ' v ' + pcs[pc2_ind])
    plt.colorbar()
    for i, txt in enumerate(sample_label):
        ax.annotate(txt, (reduced_data[:,pc1_ind][i], reduced_data[:,pc2_ind][i]))
    return plt.show()

kpca_trans, tsne_trans = dataReduction(train_data)
y_pred = kMeans(tsne_trans)
pcsVisualization(tsne_trans, y_pred, 0, 1)
