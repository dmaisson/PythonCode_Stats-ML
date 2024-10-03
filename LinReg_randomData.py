# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:45:40 2022

@author: DavidMaisson
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

psth = np.random.randint(2, size=(1000,120))
x = list(range(0,119))
for xi in x:
    r = np.random.randint(0,len(psth),size=10)
    psth[r,x[xi]] = psth[r,x[xi]]+2

var = np.random.normal(np.random.random(1), abs(np.random.random(1)),size=(1000,120))

normed_psth = stats.zscore(psth)

mpsth = np.average(normed_psth,axis = 1)
pred = var[:,(2)]

reg1 = stats.linregress(pred,mpsth)

print("slope = " + str(reg1.slope))
print("p = " + str(reg1.pvalue))
print("r = " + str(reg1.rvalue))

x = np.linspace(min(pred),max(pred),num = len(pred))
y = x*reg1.slope

plt.figure()
plt.scatter(pred,mpsth)
# plt.hold(True)
plt.plot(x,y, color = 'red')
