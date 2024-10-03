# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:07:33 2023

@author: dm40124
"""

import numpy as np
import os

dir_path = os.getcwd()
dir_list = os.listdir(dir_path)

liwc_m_prep = np.zeros((len(dir_list),106))
raw_m_prep = np.zeros((len(dir_list),34))
roll_m_prep = np.zeros((len(dir_list),30))

liwc_sem_prep = np.zeros((len(dir_list),106))
raw_sem_prep = np.zeros((len(dir_list),34))
roll_sem_prep = np.zeros((len(dir_list),30))



iB = 0
while iB < len(dir_list):
    data = np.genfromtxt(dir_list[iB],delimiter=',')
    
    # trim summary and ID vars, and header row
    data = np.delete(data,0,axis = 0)
    data = np.delete(data,np.arange(0,12), axis = 1)
    data = np.delete(data,np.arange(205,210), axis = 1)
    
    # separate LIWC
    liwc = data[:,82:188]
    data = np.delete(data,np.arange(82,188),axis = 1)
    
    # trim the nonsense
    data = np.delete(data,np.arange(0,35),axis = 1)
    
    # separate raw and wrap-up
    raw = data[:,13:47]
    roll = np.delete(data,np.arange(13,47),axis = 1)
    del(data)
    
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
    del(liwc,raw,roll)
    
    liwc_m_prep[iB,:] = m_liwc 
    raw_m_prep[iB,:] = m_raw
    roll_m_prep[iB,:] = m_roll
    del(m_liwc,m_raw,m_roll)
    liwc_sem_prep[iB,:] = sem_liwc 
    raw_sem_prep[iB,:] = sem_raw
    roll_sem_prep[iB,:] = sem_roll
    del(sem_liwc,sem_raw,sem_roll)
    
    iB += 1
del(dir_list,dir_path,n,iA,iB)