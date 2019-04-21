# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 23:12:51 2019

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

data = pd.read_csv('Mydata.csv')
data = data.loc[:,(data != 0).any(axis = 0)]
unprocessed_dataset = data.loc[:,(data != 0).any(axis = 0)]

y = sorted(list(range(10)) *3)
sample_list = sorted(['L1T1', 'L1T2','L1T3', 'L1T4', 'L3T2', 'L3T3', 'L3T4', 'L5T3', 'L5T4', 'L7T4'] * 3)
sample_arr = np.array(sample_list)

unprocessed_dataset['y'] = y
dataset = unprocessed_dataset
X = dataset.iloc[:, :-1].values 
y_true = dataset.iloc[:, -1].values

sc = StandardScaler()
sc_X = sc.fit_transform(X)

pca = PCA(10) #retaining 99.99 variaance of the data
X_trans = pca.fit_transform(sc_X)
num_components = pca.n_components_

pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
loadings = pd.DataFrame(pca.components_,columns = data.columns, index = pcs)
axes_pos = np.arange(len(pcs))
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
#plt.savefig('PC1_vs_PCs.png')
plt.show()

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
#plt.savefig('PC2_vs_PCs.png')
plt.show()

# =============================================================================
# Scree Plot
# =============================================================================

eigen_val = (pca.explained_variance_ratio_[:10]) * 100

fig, ax = plt.subplots(figsize = (12,8))

fig = plt.figure(figsize = (24, 12)) #for subplots in one figure
plt.rc('font', size = 10)

ax = plt.subplot(1, 2, 1)
plt.bar(axes_pos, eigen_val, align='center', alpha = 0.5, label = eigen_val)
for ind, val in enumerate(eigen_val):
    ax.text(ind - 0.12, val + .5, ("{} %".format(round(val, 1))), color = 'r', fontweight = 'bold')
plt.xticks(axes_pos, pcs)
plt.xlabel('Principal Components')
plt.ylabel('Eigen Value (%)')
plt.title('Scree Plot')
#plt.savefig('ScreePlot.png')
#plt.show()

# =============================================================================
# Explained Variance Ratio, total Variance of the PCs
# =============================================================================

cumulative_var = np.cumsum(pca.explained_variance_ratio_ ) * 100

#fig, ax = plt.subplots(figsize = (12,8))
ax2 = plt.subplot(1, 2, 2)
plt.bar(axes_pos, cumulative_var, align = 'center', alpha = 0.5, label = eigen_val)
for ind, val in enumerate(cumulative_var):
    ax2.text(ind - 0.12, val + .6, ("{} %".format(round(val, 1))), color = 'r', fontweight = 'bold')
plt.xticks(axes_pos, pcs)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variance (%)')
plt.title('Cumulative sum of variance')
fig.suptitle('Scree plot & Cumulative variance')
fig.savefig('Scree_CumulativeVariance.png')
#plt.savefig('CumulativeExplainedVariance.png')
plt.show()


# =============================================================================
# Plotting histogram of the loadings of PCS
# =============================================================================
from scipy.stats import norm

fig = plt.figure(figsize = (20, 10))
plt.rc('font', size = 8)

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
#fig.suptitle('Distribution of Principal components', fontsize = 14, color = 'r')
plt.tight_layout()
#plt.savefig('PCs_distribution.png')
plt.show()

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

plt.subplot(2, 3, 1)
gene_contrib = variable_contrib(0)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 3, 2)
gene_contrib = variable_contrib(1)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 3, 3)
gene_contrib = variable_contrib(2)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 3, 4)
gene_contrib = variable_contrib(3)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 3, 5)
gene_contrib = variable_contrib(4)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

plt.subplot(2, 3, 6)
gene_contrib = variable_contrib(5)[0]
gene_contrib.iloc[0, 0:10].plot(kind = 'bar', title = '% Contribution of features on {}.'.format(list(gene_contrib.index)[0]))

fig.subplots_adjust(hspace = 0.05, wspace = 0.1)
plt.tight_layout()
#plt.savefig('GenesContribution_onPCs.png')
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