# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:07:33 2023

@author: dm40124
"""

import numpy as np
import os
import matplotlib.pyplot as plt

dir_path = os.getcwd()
dir_list = os.listdir(dir_path)

liwc_m_prep = np.zeros((len(dir_list),106))
raw_m_prep = np.zeros((len(dir_list),34))
roll_m_prep = np.zeros((len(dir_list),30))
pers_m_prep = np.zeros((len(dir_list),35))

iC = 0
while iC < len(dir_list):
    data = np.genfromtxt(dir_list[iC],delimiter=',')
    
    # trim summary and ID vars, and header row
    data = np.delete(data,0,axis = 0)
    data = np.delete(data,np.arange(0,12), axis = 1)
    data = np.delete(data,np.arange(205,210), axis = 1)
    
    # norm interviewee against interviewer
    data_intr = np.zeros((int(len(data)/2),np.size(data,axis=1)))
    data_resp = np.zeros((int(len(data)/2),np.size(data,axis=1)))
    iB = 0
    while iB < int(len(data)/2):
        n = iB*2
        data_intr[iB,:] = data[n,:]
        data_resp[iB,:] = data[n+1,:]
        del (n)
        iB += 1
    
    data_intr = np.mean(data_intr,axis = 0)
    data_resp = np.mean(data_resp,axis = 0)
    # norm = abs((data_resp - data_intr) / (data_resp + data_intr))
    data = data_resp

    # separate LIWC
    liwc = data[82:188]
    data = np.delete(data,np.arange(82,188),axis = 0)
    
    # separate personality traits
    pers = data[0:35]
    data = np.delete(data,np.arange(0,35),axis = 0)
    
    # separate raw and wrap-up
    raw = data[13:47]
    roll = np.delete(data,np.arange(13,47),axis = 0)
    del(data,iB, data_intr,data_resp)
    
    liwc_m_prep[iC,:] = liwc 
    raw_m_prep[iC,:] = raw
    roll_m_prep[iC,:] = roll
    pers_m_prep[iC,:] = pers
    
    iC += 1
del(dir_list,dir_path,iC,liwc,pers,raw,roll)