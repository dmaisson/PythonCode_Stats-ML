# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:27:57 2023

@author: dm40124
"""
import numpy as np
from scipy import stats

# %%
# Record ALL Collinearities
def collinearities(data):
    n = np.size(data,axis=1)
    iA = 0
    iB = 0
    collinearity_matrix = np.zeros((n+1,n+1)); collinearity_matrix[:] = np.nan
    collinearity_matrix[:] = np.nan
    while iA < n:
        while iB < n:
            x = data[:,iA]
            y = data[:,iB]
            check_x = len(np.unique(x))
            check_y = len(np.unique(y))
            if check_x <= 8:
                nas = np.logical_or(np.isnan(x), np.isnan(y))
                r, p = stats.spearmanr(x[~nas], y[~nas])
            elif check_y <= 8:
                nas = np.logical_or(np.isnan(x), np.isnan(y))
                r, p = stats.spearmanr(x[~nas], y[~nas])
            else:
                nas = np.logical_or(np.isnan(x), np.isnan(y))
                r, p = stats.pearsonr(x[~nas], y[~nas])
            if p < 0.05:
                collinearity_matrix[iA,iB] = r
            else:
                collinearity_matrix[iA,iB] = 'nan'
            del(x,y,r,p,nas,check_x,check_y)
            iB += 1
        if iB == n:
            iB = 0
        iA += 1
    return collinearity_matrix
