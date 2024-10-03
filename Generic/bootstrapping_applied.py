def bootstrapping_w(data):
    import numpy as np

    #in_vars = int(input("Which variable? (integers only)"))
    a = np.shape(data); a1 = a[0]; a2 = a[1]
    if a1 > 10 and a1 <= 100:
        ii = 0
        boots = np.zeros((10,10,a2))
        while ii < 10:
            r = np.random.permutation(a1)
            r = r[0:10]
            boots[ii,:,:] = data[r,:]
            ii += 1
        del(r, ii)
    elif a1 > 100 and a1 <= 500:
        ii = 0
        boots = np.zeros((50,50,a2))
        while ii < 50:
            r = np.random.permutation(a1)
            r = r[0:50]
            boots[ii,:,:] = data[r,:]
            ii += 1
        del(r, ii)
    elif a1 <= 10:
        print("Sample size too small. No bootstrapping.")
    else:
        ii = 0
        boots = np.zeros((100,100,a2))
        while ii < 100:
            r = np.random.permutation(a1)
            r = r[0:100]
            boots[ii,:,:] = data[r,:]
            ii += 1
        del(r, ii)
        return boots

def bootstrapping_wo(data):
    import numpy as np

    #in_vars = int(input("Which variable? (integers only)"))
    a = np.shape(data); a1 = a[0]; a2 = a[1]
    if a1 > 10 and a1 <= 100:
        ii = 0
        s = a1 % 10
        s = int(((a1-s)/10))
        boots = np.zeros((s,10,a2))
        r = np.random.permutation(a1)
        r = r[0:(s*10)]
        while ii < s:
            if ii == 0:
                r1 = r[ii:10]
            else:
                r1 = r[(ii*10):((ii+1)*10)]
            boots[ii,:,:] = data[r1,:]
            del(r1)
            ii += 1
        del(r, ii)
    elif a1 > 100 and a1 <= 500:
        ii = 0
        s = a1 % 50
        s = int(((a1-s)/50))
        boots = np.zeros((s,50,a2))
        r = np.random.permutation(a1)
        r = r[0:(s*50)]
        while ii < s:
            if ii == 0:
                r1 = r[ii:50]
            else:
                r1 = r[(ii*50):((ii+1)*50)]
            boots[ii,:,:] = data[r1,:]
            del(r1)
            ii += 1
        del(r, ii)
    elif a1 <= 10:
        print("Sample size too small. No bootstrapping.")
    else:
        ii = 0
        s = a1 % 100
        s = int(((a1-s)/100))
        boots = np.zeros((s,100,a2))
        r = np.random.permutation(a1)
        r = r[0:(s*100)]
        while ii < s:
            if ii == 0:
                r1 = r[ii:100]
            else:
                r1 = r[(ii*100):((ii+1)*100)]
            boots[ii,:,:] = data[r1,:]
            del(r1)
            ii += 1
        del(r, ii)
    return boots