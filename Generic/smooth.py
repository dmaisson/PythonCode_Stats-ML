def smooth(data):
    import scipy.ndimage as im
    import numpy as np
    import scipy.stats as stats

    check = input("Is your data structured as observations (row) X variables (col): y/n?")
    if check == 'y':
        ii = int(input("How many variables do you want to smooth?"))
        i = 0
        n = np.shape(data)
        n = n[0]
        if ii > 1:
            output = np.zeros([n, ii])
        while i < ii:
            var = int(input("Which variables do you want to smooth?"))
            sigma = int(input("What's the respective sigma for smoothing?"))
            orig = data[:, var]
            x = im.gaussian_filter(orig, sigma)
            ks, p = stats.kstest(orig, x)
            if ii > 1:
                output[:, i] = x
            else:
                output = x
            if p < 0.05:
                print("WARNING! You may have smoothed the data too much: p = " + str(p) + '. Consider reducing sigma and resmoothing.')
            i += 1
    else:
        foo = 1
    return output
