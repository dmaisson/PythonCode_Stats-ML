def ttests_applied(data1, data2):

# This script will apply the scipy stats package for TTests to your data structured as a matrix
    # columns are variables, rows are instances
# The script will first ask which variable along which you'd like to perform your comparison.
# Then, if you have 10 or fewer observations, it will run a simple comparison along that axis. If you have greater than
# 10 observations, it will perform a set of size-specific bootrapping with replacement and perform the comparison on only
# the bootstrapped sample. Then, it will aggregate the distribution of T-stats and p-values and provide metrics of central
# tendency as well as to define the proportion of bootstraps in which the ttest delivered a significant result.

    import scipy.stats as stats
    import numpy as np
    import matplotlib.pyplot as plt

    in_vars = int(input("Which variable? (integers only)"))
    a1 = np.shape(data1); a1 = a1[0]
    a2 = np.shape(data2); a2 = a2[0]
    if a1 <= a2:
        a = a1
    elif a2 < a1:
        a = a2
    del(a1, a2)

    if a <= 10:
        x = data1[:, in_vars]
        y = data2[:, in_vars]
        t, p = stats.ttest_ind(x, y)
    elif a > 10 and a <= 100:
        ii = 0
        t_dist = np.zeros(10)
        p_dist = np.zeros(10)
        while ii < 10:
            r = np.random.permutation(a)
            r = r[0:10]
            sub1 = data1[r,in_vars]
            sub2 = data2[r,in_vars]
            t_dist[ii], p_dist[ii] = stats.ttest_ind(sub1,sub2)
            ii += 1
        t_avg = np.mean(t_dist)
        t_sem = np.std(t_dist)/np.sqrt(ii-1)
        p_avg = np.mean(p_dist)
        p_sem = np.std(p_dist)/np.sqrt(ii-1)
        temp = np.where(p_dist < 0.05)
        sig_rate = np.size(temp)/np.size(p_dist)
        del(r, sub1, sub2, ii, temp)
    elif a > 100 and a <= 500:
        ii = 0
        t_dist = np.zeros(50)
        p_dist = np.zeros(50)
        while ii < 50:
            r = np.random.permutation(a)
            r = r[0:50]
            sub1 = data1[r,in_vars]
            sub2 = data2[r,in_vars]
            t_dist[ii], p_dist[ii] = stats.ttest_ind(sub1,sub2)
            ii += 1
        t_avg = np.mean(t_dist)
        t_sem = np.std(t_dist)/np.sqrt(ii-1)
        p_avg = np.mean(p_dist)
        p_sem = np.std(p_dist)/np.sqrt(ii-1)
        temp = np.where(p_dist < 0.05)
        sig_rate = np.size(temp)/np.size(p_dist)
        del(r, sub1, sub2, ii, temp)
    else:
        ii = 0
        t_dist = np.zeros(100)
        p_dist = np.zeros(100)
        while ii < 100:
            r = np.random.permutation(a)
            r = r[0:100]
            sub1 = data1[r,in_vars]
            sub2 = data2[r,in_vars]
            t_dist[ii], p_dist[ii] = stats.ttest_ind(sub1,sub2)
            ii += 1
        t_avg = np.mean(t_dist)
        t_sem = np.std(t_dist)/np.sqrt(ii-1)
        p_avg = np.mean(p_dist)
        p_sem = np.std(p_dist)/np.sqrt(ii-1)
        temp = np.where(p_dist < 0.05)
        sig_rate = np.size(temp)/np.size(p_dist)
        del(r, sub1, sub2, ii, temp)

    x = list(range(np.size(p_dist)))

    fig = plt.hist(p_dist)
    return t_avg, t_sem, p_avg, p_sem, sig_rate