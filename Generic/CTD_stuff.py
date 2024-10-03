# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:53:44 2023

@author: dm40124
"""

import numpy as np

data = np.loadtxt('DATA_ATU_MAY.csv')
# iA = 0
# while iA < n:
#     iA += 1

# %% Define single predictors
# •	Determine empirically the best predictors independent after accounting for collinearities (1-hot design matrix with systematic addition of vars until model is “unimprovable’)
def OptimalModel(data):
    import numpy as np
    from itertools import combinations
    import statsmodels.api as sm
    from matplotlib import pyplot as plt
    from scipy import stats
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    
    n = np.size(data, axis = 1)
    R2 = np.zeros((n,2)) # initialize matrix
    com_set = combinations(range(1,n), 1)
    for i in com_set:
        predictor = data[:,int(i[0])]
        response = data[:,0]
        model = sm.OLS(predictor,response)
        result = model.fit()
        R2[int(i[0])] = result.rsquared
    plt.plot(R2[1:n])
    R2[:,1] = np.array(range(0,n))
    x = R2[:, 0].argsort()
    iA = 0
    R2_sorted = np.zeros((n,2))
    while iA < n:
        R2_sorted[iA,:] = R2[int(x[iA]),:]
        iA += 1
    R2_sorted = np.delete(R2_sorted,0,0)
    pred_order = np.flipud(R2_sorted[:,1])
    del (com_set, i, iA, model, n, predictor, result, x)

    # Record ALL Collinearities
    n = len(pred_order)
    n_prev = n
    iA = 0
    iB = 0
    collinearities = np.zeros((n+1,n+1))
    while iA < n:
        while iB < n:
            x = data[:,int(pred_order[iA,])]
            y = data[:,int(pred_order[iB,])]
            r, p = stats.pearsonr(x,y)
            if p < 0.05:
                collinearities[int(pred_order[iA]),int(pred_order[iB])] = r
            else:
                collinearities[int(pred_order[iA]),int(pred_order[iB])] = 'nan'
            del(x,y,r,p)
            iB += 1
        if iB == n:
            iB = 0
        iA += 1
    
    # Systematically Remove Collinearities
    n = len(pred_order)
    n_prev = n
    iA = 0
    while iA < n:
        iB = iA + 1
        while iB < n:
            x = data[:,int(pred_order[iA,])]
            y = data[:,int(pred_order[iB,])]
            r, p = stats.pearsonr(x,y)
            if p < 0.05:
                pred_order = np.delete(pred_order, iB)
            else:
                n_prev = n
            del(x,y,r,p)
            n = len(pred_order)
            if n < n_prev:
                iB = iA
            iB += 1
        iA += 1
    del(iA, iB, n, n_prev, R2_sorted)
    
    # Restructure vars to prep for glm
    n = len(data)
    iA = 0
    response = np.zeros((len(data),1))
    while iA < n:
        response[iA,0] = data[iA,0]
        iA += 1
    
    iA = 0
    n = len(pred_order)
    predictors = np.zeros((len(data),n))
    while iA < n:
        predictors[:,iA] = data[:,int(pred_order[iA])]
        iA += 1
    del(iA)
    
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
        del(X, model, params, new_X, mse, v_b, s_b, t_b)
        
        if iA == 0:
            best_model = iA
            R2_best = R2a
            p_best = p_val
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
        del(R2a,p_val)
        iA += 1
    del(iA,n)
    
    #
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,4],data[:,7],data[:,18])
    plt.show()
    
    # % Clustering
    # •	There will be a select few that make model best – these select few will lend themselves to optimal intelligent clustering
    from sklearn.mixture import GaussianMixture
    from scipy import stats
    
    n = len(data)
    n2 = len(pred_order)
    components = list(range(1,n2+1))
    BIC = np.zeros((n2-1,1))
    # pred_norm = predictors
    pred_norm = stats.zscore(predictors)
    iA = 0
    while iA < (n2-1):
        gmm = GaussianMixture(n_components = int(components[iA]), covariance_type = "full", n_init = 10)
        gmm.fit(pred_norm)
        BIC[iA] = gmm.bic(pred_norm)
        iA += 1
    plt.figure()
    plt.plot(BIC)
    
    best_n = np.where(BIC > 0) #== min(BIC) and BIC > 0)
    best_n = best_n[0]
    best_n = np.where(BIC == min(BIC[0:max(best_n)+1]))
    best_n = best_n[0]+1
    gmm = GaussianMixture(n_components = int(best_n)).fit(pred_norm)
    cluster = gmm.predict(pred_norm)
    postProb = gmm.predict_proba(pred_norm)
    plt.figure()
    plt.hist(cluster)
    pred_norm = np.append(np.zeros((n,1)), pred_norm, axis=1)
    iA = 0
    test = np.zeros((n,1))
    while iA < n:
        test[iA] = cluster[iA]
        iA += 1
    stat, p = stats.f_oneway(test,response)
    plt.figure()
    plt.scatter(test, response)
    
    # %
    # •	Test if intelligent clustering by HUI/cog vars out performs demographic
    
    cluster  = test
    iA = 0
    race = np.zeros((len(cluster),1))
    while iA < len(cluster):
        race[iA,0] = data[iA,23]
        iA += 1
    factors = np.append(race, cluster, axis = 1)
    factors = np.append(factors, response, axis = 1)
    X = np.zeros((len(factors),1))
    y = np.zeros((len(factors),1))
    iA = 0
    while iA < len(X):
        X[iA] = factors[iA,0]
        y[iA] = factors[iA,2]
        iA += 1
    model = linear_model.LinearRegression()
    model = model.fit(X,y)
    pred = model.predict(X)
    R2_clust = model.score(X,y)
    # R2_clust = r2_score(pred,y_test)
    params = np.append(model.intercept_,model.coef_)
    new_X = np.append(np.ones((len(X),1)), X, axis = 1)
    mse  = mean_squared_error(y, pred)
    v_b = mse*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
    s_b = np.sqrt(v_b)
    t_b = params/ s_b
    p_clust = [2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_b]
    p_clust = np.round(p_clust,3)
    del (model, params, new_X, mse, v_b, s_b, t_b, pred, X, y)
    
    X = np.zeros((len(factors),1))
    y = np.zeros((len(factors),1))
    iA = 0
    while iA < len(X):
        X[iA] = factors[iA,1]
        y[iA] = factors[iA,2]
        iA += 1
    model = linear_model.LinearRegression()
    model = model.fit(X,y)
    pred = model.predict(X)
    R2_race = model.score(X,y)
    # R2_clust = r2_score(pred,y_test)
    params = np.append(model.intercept_,model.coef_)
    new_X = np.append(np.ones((len(X),1)), X, axis = 1)
    mse  = mean_squared_error(y, pred)
    v_b = mse*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
    s_b = np.sqrt(v_b)
    t_b = params/ s_b
    p_race = [2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_b]
    p_race = np.round(p_clust,3)
    del (model, params, new_X, mse, v_b, s_b, t_b, pred, X, y)
    
    # % Clean-up
    results = {
      "R2 all vars": R2,
      "clusters": cluster,
      "p_best": p_best,
      "p_clust": p_clust,
      "p_race": p_race,
      "pred_order": pred_order,
      "pred_order_opt": pred_order_opt,
      "R2_best": R2_best,
      "R2_clust": R2_clust,
      "R2_race": R2_race,
      "Collinearities": collinearities,
    }
    del (ax,best_model,best_n,BIC,components,factors,fig,gmm,iA,n2,postProb,
         pred_norm,predictors,predictors_opt,race,response,stats,test,cluster,
         p_best,p_clust,p_race,pred_order,pred_order_opt,R2_best,R2_clust,
         R2_race,stat,p,R2)
    return results

# %% Define Optimal Predictive Model
All_results = OptimalModel(data)

# %% none of this may be necessary
# start with pre-filtered data by demog, then see model differences

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

# White = filter(data) # value = 1
# White = np.delete(White, 23, axis = 1)
Black = filter(data) # value = 2
Black = np.delete(Black, 23, axis = 1)
# Hisp = filter(data) # value = 3
# Hisp = np.delete(Hisp, 23, axis = 1)
# Asian = filter(data) # value = 4
# Asian = np.delete(Asian, 23, axis = 1)
# Native = filter(data) # value = 5
# Native = np.delete(Native, 23, axis = 1)
Mixed = filter(data) # value = 6
Mixed = np.delete(Mixed, 23, axis = 1)

Vaxed = filter(data)
Vaxed = np.delete(Vaxed, 19, axis = 1)
HIV = filter(data)
HIV = np.delete(HIV, 32, axis = 1)
ConStat = filter(data) # value = 2
ConStat = np.delete(ConStat, 31, axis = 1)

HIV_Black = filter(Black)
HIV_Black = np.delete(HIV_Black, 30, axis = 1)

del (data)

# %%
Black_results = OptimalModel(Black); del (Black)
Mixed_results = OptimalModel(Mixed); del (Mixed)
ConStat_results = OptimalModel(ConStat); del (ConStat)
HIV_results = OptimalModel(HIV); del (HIV)
HIV_Black_results = OptimalModel(HIV_Black); del (HIV_Black)
Vaxed_results = OptimalModel(Vaxed); del (Vaxed)