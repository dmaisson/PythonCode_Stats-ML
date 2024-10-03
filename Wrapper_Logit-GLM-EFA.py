# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#%% Define Baseline and Transform Functions
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

def collinearities(matrix):
    n = np.size(matrix, axis = 1)
    iA = 0
    iB = 0
    collinearities_matrix = np.zeros((n+1,n+1)); collinearities_matrix[:] = np.nan
    while iA < n:
        while iB < n:
            x = matrix[:,iA]
            y = matrix[:,iB]
            nas = np.logical_or(np.isnan(x), np.isnan(y))
            r, p = stats.pearsonr(x[~nas], y[~nas])
            if p < 0.05:
                collinearities_matrix[iA,iB] = r
            else:
                collinearities_matrix[iA,iB] = 'nan'
            del(x,y,r,p,nas)
            iB += 1
        if iB == n:
            iB = 0
        iA += 1
    collinearities_matrix = np.delete(collinearities_matrix,-1,axis = 0)
    collinearities_matrix = np.delete(collinearities_matrix,-1,axis = 1)
    return collinearities_matrix

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

#%% Generate Baselines
# Ideal curve
baseline,slopiness = gen_ideal()
# plt.plot(baseline_ideal[:,1],baseline_ideal[:,0])
baseline = {"rate": baseline, "slopiness": slopiness};
results = {"ideal": baseline}

# Random curve
baseline,slopiness = gen_baseline()
plt.plot(baseline[:,1],baseline[:,0])
baseline = {"rate": baseline, "slopiness": slopiness};
results["random"] = baseline
del(baseline, slopiness)

#% data read
data = np.genfromtxt('Sample_data.csv',delimiter=",")
n = len(data); results["sample size"] = n;

#%% variable computations
f1a = data[:,0]*data[:,1]
f1b = data[:,2]*data[:,3]
c1 = data[:,4]
f2a = data[:,5]*data[:,6]
f2b = data[:,7]*data[:,8]
c2 = data[:,9]

#%% logistic projections
projected, slopiness = logistic_projection(f1a,f1b,c1)
# plt.plot(projected[:,1],projected[:,0])
Factor = {"projected": projected, "slopiness": slopiness}
projected_results = {"Factor 1": Factor}

projected, slopiness = logistic_projection(f2a,f2b,c2)
# plt.plot(projected[:,1],projected[:,0])
Factor = {"projected": projected, "slopiness": slopiness}
projected_results["Factor 2"] = Factor

results["projected"] = projected_results; del(projected_results, projected, Factor, slopiness)

#%% logistic transform
transformed, slopiness = logistic_transform(f1a,f1b,c1); del(f1a,f1b,c1)
Factor = {"transform": transformed, "slopiness": slopiness}
transformed_results = {"Factor 1": Factor};

transformed, slopiness = logistic_transform(f2a,f2b,c2); del(f2a,f2b,c2)
# plt.plot(f1_transformed[:,1],f1_transformed[:,0])
Factor = {"transform": transformed, "slopiness": slopiness}
transformed_results["Factor 2"] = Factor

results["transformed"] = transformed_results
del(Factor,slopiness, transformed, transformed_results)

#%% collinearity matrix (descriptive and informative, not stat test)
#       I have this function built. Just need to import and adjust to match
# Record ALL Collinearities
temp_set = results["transformed"]
temp_keys = list(temp_set.keys())
temp_n = len(temp_set)
iA = 0
while iA < temp_n:
    key = temp_keys[iA]
    temp_f = temp_set[key]
    temp_data = temp_f["transform"]
    if iA == 0:
        transform_agg = np.zeros((n,1))
        transform_agg[:,0] = temp_data[:,0]
    else:
        temp_vector = np.zeros((n,1))
        temp_vector[:,0] = temp_data[:,0]
        transform_agg = np.concatenate((transform_agg,temp_vector),axis = 1)
        del(temp_vector)
    del(key,temp_f,temp_data)
    iA += 1
collinearities_matrix = collinearities(transform_agg)
results["Aggregated Factors"] = transform_agg
results["Collinearities"] = collinearities_matrix
del(temp_set,temp_keys,temp_n,iA,collinearities_matrix,transform_agg)

#%% calculate drift from ideal

temp_set = results["projected"]
ideal = results["ideal"]["rate"]
random = results["random"]["rate"]
temp_keys = list(temp_set.keys())
temp_n = len(temp_set)
iA = 0
while iA < temp_n:
    key = temp_keys[iA]
    temp_f = temp_set[key]
    temp_data = temp_f["projected"]
    
    # average distange from projected point to projected point
    results["projected"][key]["MSE_Ideal"] = np.mean((temp_data[:,0] - ideal[:,0])**2)*100
    results["projected"][key]["MSE_random"] = np.mean((temp_data[:,0] - random[:,0])**2)*100
    
    # KS test on full shape
    coeff, p = stats.kstest(temp_data[:,0], ideal[:,0])
    results["projected"][key]["KS_coeff_ideal"] = coeff
    results["projected"][key]["KS_pval_ideal"] = p
    coeff, p = stats.kstest(temp_data[:,0], random[:,0])
    results["projected"][key]["KS_coeff_random"] = coeff
    results["projected"][key]["KS_pval_random"] = p

    del(key,temp_f,temp_data,coeff,p)
    iA += 1
del(temp_keys,temp_n,temp_set,iA,random,ideal)

# distribution of differences between the two sets of curves
    # including keeping directionality consistent


#%% General group differentiation (ANOVA) - across mental ops
agg = results["Aggregated Factors"]
F, p = stats.f_oneway(agg[:,0],agg[:,1])
ANOVA = {"Coeff": F, "p-value": p}
results["ANOVA"] = ANOVA; del(ANOVA,F,p,agg)

#%% model for global choice
predictors = results["Aggregated Factors"]
choice = np.zeros((n,1)); #nX1 vector of choice rate for NUCALA across all patient profile choices

# Global choice should be taken from patient-profile and other direct nucala choices
    # should be calculated as rate (across choices) of choosing nucala
    # then model for the influence of Mental Op effects on Global Nuc Choice
    # Collinearity removals within modeling
results["Global Model"] = OptimalModel(predictors, choice);

#%% Decision Tree
# iteratively build

# check those specific questions for order of consideration and compare

# (qualitative - not here) draw map and compare to voice note on algo

# might be a way to actually run a Dec Tree (or iterative SVM?) but n is small
