# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:06:19 2023

@author: dm40124
"""

import numpy as np
def filter(data):
    filt = int(input("Which column is being filtered? "))
    filt_val = int(input("What value are you filtering for? "))
    
    x = np.empty(np.shape(data))
    x[:] = np.nan
    y = data[:,filt]
    iA = 0
    while iA < len(y):
        if y[iA] == filt_val:
            x[iA,:] = data[iA,:]
        iA += 1
    
    output = x[~np.isnan(x).any(axis=1),:]
    return output