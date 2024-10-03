# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:17:21 2023

@author: dm40124
"""

# %% 

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# import sklearn.impute as impute

def Z_Score_dm(data):

    m = np.mean(data)
    sigma = np.std(data)

    output = np.zeros(np.shape(data))
    ii = 0
    for d in data:
        x = (d - m)/sigma
        output[ii] = x
        ii += 1
    del(ii, m, sigma)
    return output

# Define single predictors
def OptimalModel(predictors, response, Imp):
    
    # median-derived imputation for NaNs
    if Imp == 1:
        i = impute.SimpleImputer(missing_values=np.nan, strategy='median')
        predictors = i.fit_transform(predictors)
        del(i)
    
    # ranked prediction
    t = np.size(predictors, axis = 1)
    R2 = np.zeros((t,4)) # initialize matrix
    iA = 0
    while iA < t:
        predictor = predictors[:,iA]
        model = sm.OLS(predictor,response,missing='drop')
        result = model.fit()
        R2[iA,0] = result.rsquared
        R2[iA,2] = result.f_pvalue
        R2[iA,3] = result.params
        iA += 1
    R2[:,1] = np.array(range(0,t))
    R2 = R2[~np.isnan(R2).any(axis=1)]
    x = R2[:, 0].argsort()
    iA = 0
    t = np.size(R2, axis = 0)
    R2_sorted = np.zeros((t,4))
    while iA < t:
        R2_sorted[iA,:] = R2[int(x[iA]),:]
        iA += 1
    x = (R2_sorted[:,2] >= (0.05/t))
    R2_sorted = np.delete(R2_sorted,x,axis=0)
    pred_order = np.flipud(R2_sorted[:,1])
    plt.figure()
    plt.plot(R2_sorted[:,0])
    del (iA, model, t, predictor, result, x)
    if np.size(pred_order) == 0:
        results = {"no good predictors"}
        return results

    # Record ALL Collinearities
    n = int(max(pred_order))
    iA = 0
    iB = 0
    collinearities = np.zeros((n+1,n+1)); collinearities[:] = np.nan
    while iA < len(pred_order):
        while iB < len(pred_order):
            x = predictors[:,int(pred_order[iA,])]
            y = predictors[:,int(pred_order[iB,])]
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
                collinearities[int(pred_order[iA]),int(pred_order[iB])] = r
            else:
                collinearities[int(pred_order[iA]),int(pred_order[iB])] = 'nan'
            del(x,y,r,p,nas,check_x,check_y)
            iB += 1
        if iB == len(pred_order):
            iB = 0
        iA += 1
    
    # Systematically Remove Collinearities
    n = len(pred_order)
    n_prev = n
    iA = 0
    while iA < n:
        iB = iA + 1
        while iB < n:
            x = predictors[:,int(pred_order[iA,])]
            y = predictors[:,int(pred_order[iB,])]
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
                pred_order = np.delete(pred_order, iB)
            else:
                n_prev = n
            del(x,y,r,p,nas,check_x,check_y)
            n = len(pred_order)
            if n < n_prev:
                iB = iA
            iB += 1
        iA += 1
    del(iA, iB, n, n_prev, R2_sorted)
    
    # Restructure vars to prep for glm
    n = len(pred_order)    
    iA = 0
    predictors_n = np.zeros((len(predictors),n))
    while iA < n:
        predictors_n[:,iA] = predictors[:,int(pred_order[iA])]
        iA += 1
    predictors = predictors_n
    del(iA,predictors_n)
    
    # Normalize for comparison
    predictors = Z_Score_dm(predictors)
    
    # Define best multivariate combination
    pred_order_opt = pred_order
    predictors_opt = predictors
    iA = 0
    while iA < n:
        model = linear_model.LinearRegression()
        X = predictors_opt[:,0:iA+1]
        model.fit(X,response)
        pred = model.predict(predictors_opt[:,0:iA+1])
        R2a = r2_score(response,pred)
        
        params = np.append(model.intercept_,model.coef_)
        new_X = np.append(np.ones((len(X),1)), X, axis = 1)
        mse  = mean_squared_error(response, pred)
        v_b = mse*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
        s_b = np.sqrt(v_b)
        t_b = params/ s_b
        p_val = [2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_b]
        p_val = np.round(p_val,3)
        multivar_coefs = model.coef_
        multivar_intercept = model.intercept_
        del(X, model, params, new_X, mse, v_b, s_b, t_b)
        
        if iA == 0:
            best_model = iA
            R2_best = R2a
            p_best = p_val
            best_multvar_coefs = multivar_coefs
            best_multvar_intercept = multivar_intercept
        elif len(pred_order_opt) < 2:
            if R2a > R2_best:
                best_model = iA
                R2_best = R2a
                p_best = p_val
                best_multvar_coefs = multivar_coefs
                best_multvar_intercept = multivar_intercept
        else:
            if p_val[-1] > 0.05:
                pred_order_opt = np.delete(pred_order_opt, iA)
                predictors_opt = np.delete(predictors_opt, iA, axis = 1)
                iA -= 1
            else:
                if R2a > R2_best:
                    best_model = iA
                    R2_best = R2a
                    p_best = p_val
                    best_multvar_coefs = multivar_coefs
                    best_multvar_intercept = multivar_intercept
        # del(R2a,p_val)
        iA += 1
    del(iA,n)
    
   
    # % Clean-up
    results = {
      "R2 all vars": R2,
      "p_best": p_best,
      "pred_order": pred_order,
      "pred_order_opt": pred_order_opt,
      "R2_best": R2_best,
      "Collinearities": collinearities,
      "best_multvar_coefs": best_multvar_coefs,
      "best_multivar_intercept": best_multvar_intercept,
    }
    del (best_model,predictors,predictors_opt,response,
         p_best,pred_order,pred_order_opt,R2_best,R2)
    return results

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

# %%
data = np.genfromtxt('interim_data.csv')

AA_outcomes = data[:,0:5]
FFs = data[:,5:52]
selfRep = data[:,52:60]
codedVars = data[:,60:65]
raw_LLM = data[:,65:99]
roll_LLM = data[:,99:129]
pers_LLM = data[:,129:164]
del(data)
allVars = np.concatenate((FFs,selfRep,codedVars,raw_LLM,roll_LLM,pers_LLM),axis = 1)

# %%
response = AA_outcomes[:,0]

FFs_results_1 = OptimalModel(FFs, response, 1)
codedVars_results_1 = OptimalModel(codedVars, response, 1)
raw_results_1 = OptimalModel(raw_LLM, response, 1)
roll_results_1 = OptimalModel(roll_LLM, response, 1)
pers_results_1 = OptimalModel(pers_LLM, response, 1)
selfRep_results_1 = OptimalModel(selfRep, response, 1)
ALL_results_1 = OptimalModel(allVars, response, 1)
ALL_alt_results_1 = OptimalModel(allVars_alt, response, 1)

response = AA_outcomes[:,1]

FFs_results_2 = OptimalModel(FFs, response, 1)
codedVars_results_2 = OptimalModel(codedVars, response, 1)
raw_results_2 = OptimalModel(raw_LLM, response, 1)
roll_results_2 = OptimalModel(roll_LLM, response, 1)
pers_results_2 = OptimalModel(pers_LLM, response, 1)
selfRep_results_2 = OptimalModel(selfRep, response, 1)
ALL_results_2 = OptimalModel(allVars, response, 1)
ALL_alt_results_2 = OptimalModel(allVars_alt, response, 1)

response = AA_outcomes[:,2]

FFs_results_3 = OptimalModel(FFs, response, 1)
codedVars_results_3 = OptimalModel(codedVars, response, 1)
raw_results_3 = OptimalModel(raw_LLM, response, 1)
roll_results_3 = OptimalModel(roll_LLM, response, 1)
pers_results_3 = OptimalModel(pers_LLM, response, 1)
selfRep_results_3 = OptimalModel(selfRep, response, 1)
ALL_results_3 = OptimalModel(allVars, response, 1)
ALL_alt_results_3 = OptimalModel(allVars_alt, response, 1)

del(response)

ALL_alt2_results = OptimalModel(allVars_alt2, y, 1)

# %% Clustering (KNN and KMeans)
# just use base code to "predict" and extract "probability" for class assign
a = allVars[:,4]
b = allVars[:,15]
c = allVars[:,16]
d = allVars[:,47]
e = allVars[:,55]
f = allVars[:,57]
g = allVars[:,58]
h = allVars[:,60]
i = allVars[:,62]
j = allVars[:,65]
k = allVars[:,76]
l = allVars[:,81]
m = allVars[:,88]
n = allVars[:,97]
o = allVars[:,112]
p = allVars[:,114]
q = allVars[:,122]
r = allVars[:,127]
s = allVars[:,148]

optimal_iso = np.zeros((43,19))
iA = 0
while iA < 43:
    optimal_iso[iA,0] = a[iA]
    optimal_iso[iA,1] = b[iA]
    optimal_iso[iA,2] = c[iA]
    optimal_iso[iA,3] = d[iA]
    optimal_iso[iA,4] = e[iA]
    optimal_iso[iA,5] = f[iA]
    optimal_iso[iA,6] = g[iA]
    optimal_iso[iA,7] = h[iA]
    optimal_iso[iA,8] = i[iA]
    optimal_iso[iA,9] = j[iA]
    optimal_iso[iA,10] = k[iA]
    optimal_iso[iA,11] = l[iA]
    optimal_iso[iA,12] = m[iA]
    optimal_iso[iA,13] = n[iA]
    optimal_iso[iA,14] = o[iA]
    optimal_iso[iA,15] = p[iA]
    optimal_iso[iA,16] = q[iA]
    optimal_iso[iA,17] = r[iA]
    optimal_iso[iA,18] = s[iA]
    iA += 1
del(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,iA)

# %%
# data = codedVars
data = optimal_iso[:,1:19]

import sklearn.impute as impute
from sklearn import cluster

i = impute.SimpleImputer(missing_values=np.nan, strategy='median')
data_imp = i.fit_transform(data)
del(i)

n_clust = 3
kmeans = cluster.KMeans(n_clusters = n_clust).fit(data_imp)
kmeans_labels = kmeans.labels_
GoF = kmeans.inertia_
plt.hist(kmeans_labels)

# %%
x = np.zeros((43,1))
iA = 0
while iA < 43:
    x[iA,0] = kmeans_labels[iA]
    iA += 1
optimal_iso = np.concatenate((optimal_iso,x), axis = 1)
del (iA,x)

optimal_iso_c1_man = filter(optimal_iso) #21,0
optimal_iso_c2_man = filter(optimal_iso) #21,1
optimal_iso_c3_man = filter(optimal_iso) #21,2

c1_opt_means = np.zeros((1,21))
c2_opt_means = np.zeros((1,21))
c3_opt_means = np.zeros((1,21))
c1_opt_SEMs = np.zeros((1,21))
c2_opt_SEMs = np.zeros((1,21))
c3_opt_SEMs = np.zeros((1,21))

iA = 0
while iA < 21:
    c1_opt_means[0,iA] = np.mean(optimal_iso_c1_man[:,iA])
    c2_opt_means[0,iA] = np.mean(optimal_iso_c2_man[:,iA])
    c3_opt_means[0,iA] = np.mean(optimal_iso_c3_man[:,iA])
    c1_opt_SEMs[0,iA] = np.std(optimal_iso_c1_man[:,iA])/np.sqrt(13)
    c2_opt_SEMs[0,iA] = np.std(optimal_iso_c2_man[:,iA])/np.sqrt(15)
    c3_opt_SEMs[0,iA] = np.std(optimal_iso_c3_man[:,iA])/np.sqrt(13)
    iA += 1

bar_w_error(c1_opt_means[:,1:19],c1_opt_SEMs[:,1:19])
bar_w_error(c2_opt_means[:,1:19],c2_opt_SEMs[:,1:19])
bar_w_error(c3_opt_means[:,1:19],c3_opt_SEMs[:,1:19])

# %%

codedVars_c1 = filter(codedVars_orig) #7,0
codedVars_c2 = filter(codedVars_orig) #7,1
codedVars_c3 = filter(codedVars_orig) #7,2

c1_opt_means = np.zeros((1,7))
c2_opt_means = np.zeros((1,7))
c3_opt_means = np.zeros((1,7))
c1_opt_SEMs = np.zeros((1,7))
c2_opt_SEMs = np.zeros((1,7))
c3_opt_SEMs = np.zeros((1,7))

iA = 0
while iA < 7:
    c1_opt_means[0,iA] = np.mean(codedVars_c1[:,iA])
    c2_opt_means[0,iA] = np.mean(codedVars_c2[:,iA])
    c3_opt_means[0,iA] = np.mean(codedVars_c3[:,iA])
    c1_opt_SEMs[0,iA] = np.std(codedVars_c1[:,iA])/np.sqrt(13)
    c2_opt_SEMs[0,iA] = np.std(codedVars_c2[:,iA])/np.sqrt(15)
    c3_opt_SEMs[0,iA] = np.std(codedVars_c3[:,iA])/np.sqrt(13)
    iA += 1