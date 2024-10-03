# -*- coding: utf-8 -*-
"""
Spyder Editor

Copyright
David Maisson - 10/3/2024
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def OptimalModel(predictors, response):
    
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