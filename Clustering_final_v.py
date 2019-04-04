# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:48:26 2019

@author: Tamal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
sample_label = dataset_read.columns.values.tolist()

data = pd.read_csv('Mydata.csv')
data = data.loc[:,(data != 0).any(axis = 0)]
unprocessed_dataset = data.loc[:,(data != 0).any(axis = 0)]

plt.scatter(data['gene_1'], data['gene_29451'])
plt.show()

stats.pearsonr(data['gene_1'], data['gene_6'])

y = sorted(list(range(10)) *3)
sample_list = sorted(['L1T1', 'L1T2','L1T3', 'L1T4', 'L3T2', 'L3T3', 'L3T4', 'L5T3', 'L5T4', 'L7T4'] * 3)
sample_arr = np.array(sample_list)

unprocessed_dataset['y'] = y
dataset = unprocessed_dataset
X = dataset.iloc[:, :-1].values 
y_true = dataset.iloc[:, -1].values

sc = StandardScaler()
sc_X = sc.fit_transform(X)

pca = PCA(.9999) #retaining 99.99 variaance of the data
X_trans = pca.fit_transform(sc_X)
num_components = pca.n_components_
print('{} components retain 99.99% variance of the data and shape {}.'.format(num_components, X_trans.shape))

# =============================================================================
# Plotting PC1 vs other PCs
# =============================================================================

fig = plt.figure(figsize = (12, 8))

plt.subplot(2, 2, 1)
plt.scatter(X_trans[:,0], X_trans[:,1], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(2, 2, 2)
plt.scatter(X_trans[:,0], X_trans[:,2], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC3')

plt.subplot(2, 2, 3)
plt.scatter(X_trans[:,0], X_trans[:,3], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC4')

plt.subplot(2, 2, 4)
plt.scatter(X_trans[:,0], X_trans[:,4], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC1')
plt.ylabel('PC5')

fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
plt.tight_layout()
plt.savefig('PC1_vs_PCs.png')

# =============================================================================
# plotting PC2 vs other pcs
# =============================================================================

fig = plt.figure(figsize = (12, 8))

plt.subplot(2, 2, 1)
plt.scatter(X_trans[:,1], X_trans[:,2], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC2')
plt.ylabel('PC3')

plt.subplot(2, 2, 2)
plt.scatter(X_trans[:,1], X_trans[:,3], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC2')
plt.ylabel('PC4')

plt.subplot(2, 2, 3)
plt.scatter(X_trans[:,1], X_trans[:,4], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC2')
plt.ylabel('PC5')

plt.subplot(2, 2, 4)
plt.scatter(X_trans[:,1], X_trans[:,5], c = y, s = 100, cmap = 'tab10', alpha = 0.7)
plt.xlabel('PC2')
plt.ylabel('PC6')

fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
plt.tight_layout()
plt.savefig('PC2_vs_PCs.png')

# =============================================================================
# Scree Plot
# =============================================================================

pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
axes_pos = np.arange(len(pcs))

eigen_val = (pca.explained_variance_ratio_[:10]) * 100

fig, ax = plt.subplots(figsize = (12,8))
#plt.rc('font', size = 7)
#fig = plt.figure(figsize = (20, 10)) #for subplots in one figure
#ax1 = plt.subplot(1, 2, 1)
plt.bar(axes_pos, eigen_val, align='center', alpha = 0.5, label = eigen_val)
for ind, val in enumerate(eigen_val):
    ax.text(ind - 0.12, val + .5, ("{} %".format(round(val, 1))), color = 'r', fontweight = 'bold')
plt.xticks(axes_pos, pcs)
plt.xlabel('Principal Components')
plt.ylabel('Eigen Value (%)')
plt.title('Scree Plot')
plt.savefig('ScreePlot.png')
plt.show()

# =============================================================================
# Explained Variance Ratio, total Variance of the PCs
# =============================================================================

cumulative_var = np.cumsum(pca.explained_variance_ratio_[:10]) * 100

fig, ax = plt.subplots(figsize = (12,8))
#ax2 = plt.subplot(1, 2, 2)
plt.bar(axes_pos, cumulative_var, align = 'center', alpha = 0.5, label = eigen_val)
for ind, val in enumerate(cumulative_var):
    ax.text(ind - 0.12, val + .6, ("{} %".format(round(val, 1))), color = 'r', fontweight = 'bold')
plt.xticks(axes_pos, pcs)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variance (%)')
plt.title('Cumulative sum of variance')
plt.savefig('explainedVariance.png')
plt.show()

loadings = pd.DataFrame(pca.components_[:6,:],columns = data.columns, index = pcs[:6])

# =============================================================================
# Plotting histogram of the loadings of PCS
# =============================================================================
from scipy.stats import norm

fig = plt.figure()

plt.subplot(2, 3, 1)
sns.distplot(loadings.loc['PC1'], fit = norm, bins = 10, axlabel = 'PC1', rug = False)

plt.subplot(2, 3, 2)
sns.distplot(loadings.loc['PC2'], fit = norm, bins = 10, axlabel = 'PC2', rug = False)

plt.subplot(2, 3, 3)
sns.distplot(loadings.loc['PC3'], fit = norm, bins = 10, axlabel = 'PC3', rug = False)

plt.subplot(2, 3, 4)
sns.distplot(loadings.loc['PC4'], fit = norm, bins = 10, axlabel = 'PC4', rug = False)

plt.subplot(2, 3, 5)
sns.distplot(loadings.loc['PC5'], fit = norm, bins = 10, axlabel = 'PC5', rug = False)

plt.subplot(2, 3, 6)
sns.distplot(loadings.loc['PC6'], fit = norm, bins = 10, axlabel = 'PC6', rug = False)

fig.subplots_adjust(hspace = 0.1, wspace = 0.1)
fig.suptitle('Distribution of Principal components', fontsize = 12, color = 'r')
plt.tight_layout()
plt.savefig('PCs_distribution.png')

# =============================================================================
# Finding the subset of genes those contribute the most on PCs
# =============================================================================

def variable_contrib(index_n):
    pc_loadings = loadings.loc[loadings.index[index_n]].to_frame().abs()
    sorted_loadings = pc_loadings.sort_values(pcs[index_n], ascending = False)
    gene_contrib = sorted_loadings.loc[(sorted_loadings[pcs[index_n]] >= 0.01)].T
    genes = gene_contrib.columns.values.tolist()
    
    return gene_contrib, genes

fig = plt.figure(figsize = (12, 8))

plt.subplot(2, 2, 1)
gene_contrib = variable_contrib(0)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 2, 2)
gene_contrib = variable_contrib(1)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 2, 3)
gene_contrib = variable_contrib(2)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 2, 4)
gene_contrib = variable_contrib(3)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

fig.subplots_adjust(hspace = 0.05, wspace = 0.1)
plt.tight_layout()
plt.savefig('GenesContribution_onPCs.png')
plt.show()

# =============================================================================
# Creating dataframe using only those genes contributed the  most on the PCs
# =============================================================================

genes_pc1 = variable_contrib(0)[1][:100]
genes_pc2 = variable_contrib(1)[1][:100]
genes_pc3 = variable_contrib(2)[1][:100]
genes_pc4 = variable_contrib(3)[1][:100]
genes_pc5 = variable_contrib(4)[1][:100]
genes_pc6 = variable_contrib(5)[1][:100]

#total_genes_onPCS = list(set(genes_pc1 + genes_pc2 + genes_pc3 + genes_pc4 + genes_pc5 + genes_pc6))
#for 5PCs
total_genes_onPCS = list(set(genes_pc1 + genes_pc2 + genes_pc3 + genes_pc4 + genes_pc5 + genes_pc6))

import re

def stringSplitByNumbers(x):
    num_expression = re.compile('(\d+)')
    variable_split = num_expression.split(x)
    
    return [int(i) if i.isdigit() else i for i in variable_split]

total_genes_onPCS = sorted(total_genes_onPCS, key = stringSplitByNumbers)

new_dataFrame = data[total_genes_onPCS]
new_dataFrame['y'] = y

curated_data = new_dataFrame.to_csv('curated_dataset_6PCs_100.csv', index = False)
#new_dataFrame = new_dataFrame.sort_index(axis = 1)
#new_dataFrame = new_dataFrame.reindex(sorted(new_dataFrame.columns), axis=1)

def data_transformation(input_data):
    #data normalization or standardization
    sc = StandardScaler()
    normalized_data = sc.fit_transform(input_data)
    
    #Linear dimensionality reduction using PCA
    pca = PCA(0.9999)
    pca_transformed = pca.fit_transform(normalized_data)
    
    #Non-linear dimensionality reduction using KPCA
    kpca = KernelPCA(n_components = 5, kernel = 'rbf')
    kpca_transformed = kpca.fit_transform(normalized_data)
    
    #Non-linear dimensionality reduction using TSNE
    tsne = TSNE(n_components = 2, n_iter = 200000, perplexity = 9, init = 'pca')
    tsne_transformed = tsne.fit_transform(normalized_data)
    
    return normalized_data, pca_transformed, kpca_transformed, tsne_transformed

# =============================================================================
# Function for the visualization of the PCs
# =============================================================================

def pcs_visualization(pc1_ind, pc2_ind, input_data):
    fig, ax = plt.subplots(figsize = (12,8))
    plt.scatter(input_data[:,pc1_ind], input_data[:,pc2_ind], c = y_true, s = 200, cmap = 'tab10', alpha = 0.7)
    plt.xlabel(pcs[pc1_ind])
    plt.ylabel(pcs[pc2_ind])
    plt.title(pcs[pc1_ind] + ' v ' + pcs[pc2_ind])
    plt.colorbar()
    for i, txt in enumerate(sample_list):
        ax.annotate(txt, (input_data[:,pc1_ind][i], input_data[:,pc2_ind][i]))
    return plt.show()

# =============================================================================
# KMeans algorithm
# =============================================================================

def k_means(m_data):
    kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 100)
    y_km = kmeans.fit_predict(m_data)
    
    return y_km

# =============================================================================
# Visualization of the cluster
# =============================================================================
    
def cluster_visualization(input_data, y_result):
    plt.rc('font', size = 10)
    fig, ax = plt.subplots(figsize = (14,10))
    plt.scatter(input_data[:,0], input_data[:,1], c = y_result, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.8)
    for i, txt in enumerate(sample_list):
        ax.annotate(txt, (input_data[:,0][i], input_data[:,1][i]))
    plt.colorbar(ticks = range(10), label = 'clusters')
    plt.title('KMeans')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    return plt.show()

normalized_data, pca_trans, kpca_trans, tsne_trans = data_transformation(X)
y_result = k_means(kpca_trans)
cluster_visualization(kpca_trans, y_result)

adjusted_rand_score = metrics.adjusted_rand_score(y_true, y_result)
print('Adjusted rand score for KMeans using KernelPCA {}.'.format(adjusted_rand_score))

homogeneity_score, completeness_score, V_score = metrics.homogeneity_completeness_v_measure(y_true, y_result)
print('Homogeneity score for KMeans using KernelPCA {}.'.format(homogeneity_score))
print('Completeness info score for KMeans using KernelPCA {}.'.format(completeness_score))

# =============================================================================
# Creating new sample/fake sample from the data
# =============================================================================

class_1 = X[[9,10,11], :]
class_2 = X[[15,16,17], :]

test_sample = np.zeros((2, class_1.shape[1]))

for j in test_sample:    
    for i in range(class_1.shape[1]):
        rand = random.random()
        if rand <= 0.40:
            j[i] = random.uniform(np.amin(class_2[:,i]), np.amax(class_2[:,i]))
        else:
            j[i] = random.uniform(np.amin(class_1[:,i]), np.amax(class_1[:,i]))
        
test_sample = test_sample.reshape(2, -1)

test_sample_label = []
for i in range(1, len(test_sample) + 1):
    test_sample_label.append('T_S_{}'.format(i))

# =============================================================================
# Creating Pipeline for train and test data transformation and Clustering
# =============================================================================

def pca_pipeline(input_data, test_sample):
    pipeline = (Pipeline([('Scalar', StandardScaler()), ('PCA', PCA(.9999))]))
    input_data_trans = pipeline.fit_transform(input_data)
    test_sample_trans = pipeline.transform(test_sample)

    return input_data_trans, test_sample_trans

def kpca_pipeline(input_data, test_sample):
    pipeline = (Pipeline([('Scalar', StandardScaler()), ('KernelPCA', KernelPCA(n_components = 5, kernel = 'rbf'))]))
    input_data_trans = pipeline.fit_transform(input_data)
    test_sample_trans = pipeline.transform(test_sample)
        
    return input_data_trans, test_sample_trans

def tsne_pipeline(input_data, test_sample):
    pipeline = (Pipeline([('Scalar', StandardScaler()), ('TSNE', TSNE(n_components = 2, n_iter = 200000, perplexity = 9, init = 'pca')),
                          ('KMeans', KMeans(n_clusters = 10, n_init = 100, max_iter = 300, init = 'k-means++'))]))
    input_data_trans = pipeline.fit_transform(input_data)
    test_sample_trans = pipeline.fit_transform(test_sample)
    
    return input_data_trans, test_sample_trans

# =============================================================================
# Call the reduction method you want to implement
# =============================================================================

def train_test_result(train_data, test_data):
    kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 100)
    y_train = kmeans.fit_predict(train_data)
    y_test = kmeans.predict(test_data)
    
    plt.rc('font', size = 10)    
    fig, ax = plt.subplots(figsize = (14,10))
    plt.scatter(train_data[:,0], train_data[:,1], c = y_train, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.7)
    plt.scatter(test_data[:,0], test_data[:,1], c = y_test, s = 200, cmap = plt.get_cmap('tab10', 10), alpha = 0.7)
    
    for i, txt in enumerate(sample_list):
        ax.annotate(txt, (train_data[:,0][i], train_data[:,1][i]))
        
    for j, txt in enumerate(test_sample_label):
        ax.annotate(txt, (test_data[:,0][j], test_data[:,1][j]))
        
    plt.xlabel('PC_1')
    plt.ylabel('PC_2')
    plt.colorbar(ticks = range(10), label = 'Clusters')
    plt.clim(0,10)
    plt.title('KMeans with reduced dimension of normalized data by KPCA')
    return y_train, y_test, plt.show()


tr_train, tr_test = kpca_pipeline(X, test_sample)
train_test_result(tr_train, tr_test)