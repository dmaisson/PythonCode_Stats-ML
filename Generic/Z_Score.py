def Z_Score(data):
    import numpy as np

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