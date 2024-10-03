# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:53:05 2023

@author: dm40124
"""

which = Tx_raw_results["pred_order"]
raw_vars = np.zeros((len(outcome_mast),len(which)))
raw_vars_sem = np.zeros((len(outcome_mast),len(which)))
raw_readME = np.zeros((len(which),2))
raw_readME[:,0] = which
raw_readME[:,1] = np.arange(0,len(which))
iA = 0
while iA < len(which):
    raw_vars[:,iA] = raw_m_prep[:,int(which[iA])]
    raw_vars_sem[:,iA] = raw_sem_prep[:,int(which[iA])]
    iA += 1
    
bar_w_error(raw_vars,raw_vars_sem)

data = np.concatenate((raw_vars,liwc_m_prep),axis=1); del(raw_vars, raw_vars_sem,iA)

rawLIWC_collinearity_matrix = collinearities(data); del (data,which)

which = Tx_roll_results["pred_order"]
roll_vars = np.zeros((len(outcome_mast),len(which)))
roll_vars_sem = np.zeros((len(outcome_mast),len(which)))
roll_readME = np.zeros((len(which),2))
roll_readME[:,0] = which
roll_readME[:,1] = np.arange(0,len(which))
iA = 0
while iA < len(which):
    roll_vars[:,iA] = roll_m_prep[:,int(which[iA])]
    roll_vars_sem[:,iA] = roll_sem_prep[:,int(which[iA])]
    iA += 1
    
bar_w_error(roll_vars,roll_vars_sem)

data = np.concatenate((roll_vars,liwc_m_prep),axis=1); del(roll_vars, roll_vars_sem,iA)

rollLIWC_collinearity_matrix = collinearities(data); del (data,which)