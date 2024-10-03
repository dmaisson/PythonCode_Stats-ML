def KNN(train, test):
    # This code will run a KNN on your data, given n features and m neighbors.
    # the script makes the following assumptions:
        # 1) that your data is structed as observations (rows) X variables (cols)
        # 2) that your data is already pre-divided into training and test sets
            # -if you want to run the simple regression on your data alone, 
            #    simply use the same data for training and testing
            #    and recognize that predicted model will be overly well-fitted.
            # -additionally, you can simply ignore the forecast variable and 
            #    consider only the fitted model parameters

    import numpy as np
    import sklearn.neighbors as KNC

    n_var = int(input("How many features will be included in your model?"))
    m = input("How many neighbors to use as minimum? (You can also type 'd' for default n = 5)")
    if m == 'd':
        m = 5
    else:
        m = int(m)
    features = np.zeros(n_var)
    ii = 0
    while ii < n_var:
        features[ii] = input("Which variable is an included feature?")
        ii += 1
    classLabel = int(input("Which variable is the class label?"))
    del (n_var, ii)

    s1 = np.shape(train)
    s1 = s1[0]
    s2 = np.shape(features)
    s2 = s2[0]
    ii = 0
    x = np.zeros([s1, s2])
    while ii < s2:
        i = 0
        while i < s1:
            feat = int(features[ii])
            x[i, ii] = train[i, feat]
            i += 1
        ii += 1
    y = train[:, classLabel]
    neigh = KNC.KNeighborsClassifier(n_neighbors=m)
    neigh.fit(x,y)

    s1 = np.shape(test)
    s1 = s1[0]
    s2 = np.shape(features)
    s2 = s2[0]
    ii = 0
    z = np.zeros([s1, s2])

    class Object(object):
        pass
    classify = Object()
    while ii < s2:
        i = 0
        while i < s1:
            feat = int(features[ii])
            z[i, ii] = test[i, feat]
            i += 1
        ii += 1
    classify.predictions = neigh.predict(z)
    classify.post_probs = neigh.predict_proba(z)

    classify.accuracy = np.zeros(s1)
    a = train[:,-1]
    ii = 0
    while ii < s1:
        if classify.predictions[ii] == a[ii]:
            classify.accuracy[ii] = 1
        ii += 1

    classify.accuracyRate = sum(classify.accuracy)/s1

    return neigh, classify
