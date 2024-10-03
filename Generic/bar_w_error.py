# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:22:00 2023

@author: dm40124
"""

def bar_w_error(Ms,SEMs):
    import matplotlib.pyplot as plt
    import numpy as np

    b = np.zeros((1,np.size(Ms,axis=1)))
    c = np.zeros((1,np.size(Ms,axis=1)))
    iA = 0
    while iA < np.size(Ms,axis=1):
        b[0,iA] = np.mean(Ms[:,iA])
        c[0,iA] = np.mean(SEMs[:,iA])
        iA += 1
    a = np.zeros((1,np.size(Ms,axis=1)))
    a[0,:] = np.arange(0,np.size(b,axis=1))
    
     
    # Plot bar here
    plt.bar(a[0], b[0])
     
    plt.errorbar(a[0], b[0], yerr=c[0], fmt="o", color="r")
     
    plt.show()