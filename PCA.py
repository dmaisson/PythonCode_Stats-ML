# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:17:40 2020

@author: david
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#%%
rng = np.random.RandomState(0)
W = rng.randn(1000,2)
plt.scatter(W[:,0],W[:,1])
plt.axis('equal')
plt.grid('on')

#%%
A = rng.rand(2,2)
X = np.dot(W,A)
plt.scatter(X[:,0],X[:,1])
plt.axis('equal')
plt.grid('on')

#%% Caluclate Cov matrix
C = np.cov(X.T)
e, V = np.linalg.eig(C)

#%%
u = V[:,1]
Z = np.dot(X,u)
Y = np.outer(Z,u.T)
plt.scatter(X[:,0],X[:,1])
plt.scatter(Y[:,0],Y[:,1])
plt.axis('equal')
plt.grid('on')

#%% Scikit learn
from sklearn import datasets
from sklearn.decomposition import PCA

#%% Data
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#%% PCA
pca = PCA(n_components=2)
pca.fit(X)

#%% transform/project
Z = pca.transform(X)

#%% plot
plt.scatter(Z[:,0],Z[:,1])

#%% another example
digits = datasets.load_digits()
pca = PCA(n_components=10)
pca.fit(digits.data)
Z = pca.transform(digits.data)
plt.scatter(Z[:,0],Z[:,1],c=digits.target,cmap='tab10',alpha=0.7)
plt.colorbar()
plt.grid('on')