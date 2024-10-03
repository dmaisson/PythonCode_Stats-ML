def conditional_ops(data):
    check_dvs = input("Do you want to perform operations on multiple variables? (y or n): ")
    if check_dvs == 'y':
        no_dvs = input("On how many variables will you perform operations? (enter integer): ")
        try:
            no_dvs = int(no_dvs)
        except:
            print("Invalid entry. Please enter integer.")
            quit()
    elif check_dvs == 'n':
        no_dvs = 1
    else:
        print("Invalid entry. Please enter y or n.")
        quit()
        ##########
    import numpy as np
    import matplotlib.pyplot as plt
    dvs = np.zeros(no_dvs)
    n = 0
    while n < np.size(dvs):
        t = input("On which columns do you want to perform operations? (enter these individuall): ")
        dvs[n] = float(t)
        n += 1
        del t
    del (check_dvs, no_dvs, n)
    ##########
    no_cons = int(input("How many conditional variables will there be? (enter integer only): "))
    cons = np.zeros([2, no_cons])
    n = 0
    a = np.shape(cons)
    while n < a[1]:
        t = input("Which columns will offer conditions? ")
        cons[0, n] = float(t)
        del t
        t = input("What is the conditional value? ")
        cons[1, n] = float(t)
        n += 1
        del t
    del (no_cons, n, a)
    ##########
    a = np.shape(cons)
    ii = 0
    con_sum = np.zeros(np.size(dvs))
    con_avg = np.zeros(np.size(dvs))
    con_rate = np.zeros(np.size(dvs))
    con_sem = np.zeros(np.size(dvs))
    while ii < a[1]:
        if ii == 0:
            idx = np.where(data[:, int(cons[0, ii])] == cons[1, ii])
            temp = data[idx, :]
            del idx
        else:
            idx = np.where(temp[:, int(cons[0, ii])] == cons[1, ii])
            temp = temp[idx, :]
            del idx
        ii += 1
    filtered = temp[0, :, :]
    del (temp, ii)
    ii = 0
    while ii < np.size(dvs):
        con_sum[ii] = np.sum(filtered[:, int(dvs[ii])])
        con_rate[ii] = ((np.sum(filtered[:, int(dvs[ii])])) / np.sum(data[:, int(dvs[ii])])) * 100
        con_avg[ii] = np.mean(filtered[:, int(dvs[ii])])
        s = np.shape(filtered)
        con_sem[ii] = np.std(filtered[:, int(dvs[ii])]) / np.sqrt(s[0])
        ii += 1

    fig, ax = plt.subplots()
    ax.errorbar(1, con_avg, con_sem, fmt='o', linewidth=2, capsize=6)

    return con_sum, con_avg, con_sem, con_rate, filtered
