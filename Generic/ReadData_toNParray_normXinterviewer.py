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

liwc_sem_prep = np.zeros((len(dir_list),106))
raw_sem_prep = np.zeros((len(dir_list),34))
roll_sem_prep = np.zeros((len(dir_list),30))
pers_sem_prep = np.zeros((len(dir_list),35))

iC = 0
while iC < len(dir_list):
    data = np.genfromtxt(dir_list[iC],delimiter=',')
    
    # trim summary and ID vars, and header row
    data = np.delete(data,0,axis = 0)
    data = np.delete(data,np.arange(0,12), axis = 1)
    data = np.delete(data,np.arange(205,210), axis = 1)
    
    # norm interviewee against interviewer
    data_normed = np.zeros((int(len(data)/2),np.size(data,axis=1)))
    data_new = np.zeros((int(len(data)/2),np.size(data,axis=1)))
    iB = 0
    while iB < int(len(data)/2):
        n = iB*2
        x = data[n,:]
        y = data[n+1,:]
        z = abs((y - x))/(y + x) # what proportion of total speech is interviewee
        z = np.transpose(z)
        data_normed[iB,:] = z
        data_new[iB,:] = data[n,:]
        del (x,y,z,n)
        iB += 1
    data_normed = np.nan_to_num(data_normed, copy=True, nan=0, posinf=None, 
                                neginf=None)
    data_new = data_normed*data_new
    data = data_new
    
    # separate LIWC
    liwc = data[:,82:188]
    data = np.delete(data,np.arange(82,188),axis = 1)
    
    # trim the nonsense
    pers = data[:,0:35]
    data = np.delete(data,np.arange(0,35),axis = 1)
    
    # separate raw and wrap-up
    raw = data[:,13:47]
    roll = np.delete(data,np.arange(13,47),axis = 1)
    del(data,data_normed,data_new)
    
    # person-wrap
    n = len(liwc)
    m_raw = np.zeros((1,np.size(raw,axis=1)))
    sem_raw = np.zeros((1,np.size(raw,axis=1)))
    iA = 0
    while iA < np.size(m_raw,axis=1):
        m_raw[0,iA] = np.mean(raw[:,iA])
        sem_raw[0,iA] = (np.std(raw[:,iA]))/(np.sqrt(n))
        iA += 1
    
    m_roll = np.zeros((1,np.size(roll,axis=1)))
    sem_roll = np.zeros((1,np.size(roll,axis=1)))
    iA = 0
    while iA < np.size(m_roll,axis=1):
        m_roll[0,iA] = np.mean(roll[:,iA])
        sem_roll[0,iA] = (np.std(roll[:,iA]))/(np.sqrt(n))
        iA += 1
    
    m_liwc = np.zeros((1,np.size(liwc,axis=1)))
    sem_liwc = np.zeros((1,np.size(liwc,axis=1)))
    iA = 0
    while iA < np.size(sem_liwc,axis=1):
        m_liwc[0,iA] = np.mean(liwc[:,iA])
        sem_liwc[0,iA] = (np.std(liwc[:,iA]))/(np.sqrt(n))
        iA += 1
    
    m_pers = np.zeros((1,np.size(pers,axis=1)))
    sem_pers = np.zeros((1,np.size(pers,axis=1)))
    iA = 0
    while iA < np.size(m_pers,axis=1):
        m_pers[0,iA] = np.mean(pers[:,iA])
        sem_pers[0,iA] = (np.std(pers[:,iA]))/(np.sqrt(n))
        iA += 1
    del(liwc,raw,roll,pers)
    
    liwc_m_prep[iC,:] = m_liwc 
    raw_m_prep[iC,:] = m_raw
    roll_m_prep[iC,:] = m_roll
    pers_m_prep[iC,:] = m_pers
    del(m_liwc,m_raw,m_roll,m_pers)
    liwc_sem_prep[iC,:] = sem_liwc 
    raw_sem_prep[iC,:] = sem_raw
    roll_sem_prep[iC,:] = sem_roll
    pers_sem_prep[iC,:] = sem_pers
    del(sem_liwc,sem_raw,sem_roll,sem_pers)
    
    iC += 1
del(dir_list,dir_path,n,iA,iB,iC)