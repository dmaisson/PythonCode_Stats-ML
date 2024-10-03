def shuffle_col(in_var):
    import numpy as np
    [y,x] = np.shape(in_var)
    out_var = np.zeros((y,x))
    ii = 0
    while ii < y:
        r = np.random.permutation(x)
        out_var[ii,:] = in_var[ii,r]
        ii += 1
    return out_var

def shuffle_row(in_var):
    import numpy as np
    [y,x] = np.shape(in_var)
    out_var = np.zeros((y,x))
    ii = 0
    while ii < x:
        r = np.random.permutation(y)
        out_var[:,ii] = in_var[r,ii]
        ii += 1
    return out_var