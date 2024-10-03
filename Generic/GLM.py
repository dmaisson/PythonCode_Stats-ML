def GLM_OLS(train, test):
    # This code will run a OLS GLM on your data, given n predictors and 1 outcome.
    # the script makes the following assumptions:
        # 1) that your data is structed as observations (rows) X variables (cols)
        # 2) that your data is already pre-divided into training and test sets
            # -if you want to run the simple regression on your data alone, simply use the same data for training and testing
            #  and recognize that predicted model will be overly well-fitted.
            # -additionally, you can simply ignore the forecast variable and consider only the fitted model parameters

    import numpy as np
    import sklearn.linear_model as linmod

    n_var = int(input("How many predictors will be included in your model? "))
    predictor = np.zeros(n_var)
    ii = 0
    while ii < n_var:
        predictor[ii] = input("Which variable is the predictor? ")
        ii += 1
    outcome = int(input("Which variable is the outcome? "))
    del (n_var, ii)

    s1 = np.shape(train)
    s1 = s1[0]
    s2 = np.shape(predictor)
    s2 = s2[0]
    ii = 0
    x = np.zeros([s1, s2])
    while ii < s2:
        i = 0
        while i < s1:
            pred = int(predictor[ii])
            x[i, ii] = train[i, pred]
            i += 1
        ii += 1
    y = train[:, outcome]
    reg = linmod.LinearRegression().fit(x, y)

    s1 = np.shape(test)
    s1 = s1[0]
    s2 = np.shape(predictor)
    s2 = s2[0]
    ii = 0
    z = np.zeros([s1, s2])
    while ii < s2:
        i = 0
        while i < s1:
            pred = int(predictor[ii])
            z[i, ii] = test[i, pred]
            i += 1
        ii += 1
    forecast = reg.predict(z)
    return reg, forecast
