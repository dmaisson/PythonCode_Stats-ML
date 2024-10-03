# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:33:35 2024

@author: david maisson 
"""

# import libraries
import numpy as np
# import matplotlib.pyplot as plt
import statsmodels.api as sm
# from scipy import stats
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score

def gen_baseline():
    baseline = np.zeros((1500,2))
    n = len(baseline)
    iA = 0
    while iA < n:
        baseline[iA,0] = np.random.randint(0,2)
        baseline[iA,1] = np.random.randint(-101,101)
        iA += 1
    log_reg = sm.Logit(baseline[:,0], baseline[:,1]).fit()
    X = np.linspace(-100,100,num=1500)
    logit = log_reg.predict(X[:])
    baseline_logit = np.zeros((1500,2))
    baseline_logit[:,0] = logit; baseline_logit[:,1] = X
    # plt.plot(baseline_logit[:,1],baseline_logit[:,0])
    temp = np.zeros((1499)); iA = 0
    while iA < (n-1):
        a = baseline_logit[iA+1,1]-baseline_logit[iA,1]
        b = baseline_logit[iA+1,0]-baseline_logit[iA,0]
        temp[iA] = np.sqrt((a*a)+(b*b))
        iA += 1
    slopiness = np.mean(temp)
    # del(a,b,baseline,iA,log_reg,logit,n,temp,X)
    return baseline_logit,slopiness

def gen_ideal():
    baseline = np.zeros((1500,2))
    n = len(baseline)
    iA = 0
    while iA < n:
        baseline[iA,1] = np.random.randint(-101,101)
        if baseline[iA,1] <= -25:
            baseline[iA,0] = 0
        elif baseline[iA,1] >= 25:
            baseline[iA,0] = 1
        else:
            baseline[iA,0] = np.random.randint(0,2)
        iA += 1
    log_reg = sm.Logit(baseline[:,0], baseline[:,1]).fit()
    X = np.linspace(-100,100,num=1500)
    logit = log_reg.predict(X[:])
    baseline_logit = np.zeros((1500,2))
    baseline_logit[:,0] = logit; baseline_logit[:,1] = X
    # plt.plot(baseline_logit[:,1],baseline_logit[:,0])
    temp = np.zeros((1499)); iA = 0
    while iA < (n-1):
        a = baseline_logit[iA+1,1]-baseline_logit[iA,1]
        b = baseline_logit[iA+1,0]-baseline_logit[iA,0]
        temp[iA] = np.sqrt((a*a)+(b*b))
        iA += 1
    slopiness = np.mean(temp)
    # del(a,b,baseline,iA,log_reg,logit,n,temp,X)
    return baseline_logit,slopiness

def logistic_projection(factor1,factor2,choice):
    n = len(choice)
    delta = factor1 - factor2
    log_reg = sm.Logit(choice[:], delta[:]).fit()
    X = np.linspace(-100,100,num=1500)
    logit = log_reg.predict(X[:])
    projected = np.zeros((1500,2))
    projected[:,0] = logit; projected[:,1] = X
    # plt.plot(baseline_logit[:,1],baseline_logit[:,0])
    temp = np.zeros((1499)); iA = 0
    while iA < (n-1):
        a = projected[iA+1,1]-projected[iA,1]
        b = projected[iA+1,0]-projected[iA,0]
        temp[iA] = np.sqrt((a*a)+(b*b))
        iA += 1
    slopiness = np.mean(temp)
    # del(a,b,baseline,iA,log_reg,logit,n,temp,X)
    return projected,slopiness

def logistic_transform(factor1,factor2,choice):
    n = len(choice)
    delta = factor1 - factor2
    log_reg = sm.Logit(choice[:], delta[:]).fit()
    logit = log_reg.predict(delta[:])
    transformed = np.zeros((n,2));
    transformed[:,0] = logit; transformed[:,1] = delta
    transformed = np.sort(transformed, axis = 0)
    temp = np.zeros((n)); iA = 0
    while iA < (n-1):
        a = transformed[iA+1,1]-transformed[iA,1]
        b = transformed[iA+1,0]-transformed[iA,0]
        temp[iA] = np.sqrt((a*a)+(b*b))
        iA += 1
    slopiness = np.mean(temp)
    # del(a,b,baseline,iA,log_reg,logit,n,temp,X)
    return transformed,slopiness