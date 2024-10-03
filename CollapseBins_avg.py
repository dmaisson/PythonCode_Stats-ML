# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:08:36 2022

@author: DavidMaisson
"""

import numpy as np

def CollapseBins_avg(input_data,resize_factor):
    n = max(np.shape(input_data))
    c = 0
    while c <= n:
        if c == 0:
            output_data = [np.average(input_data[0,c:c+resize_factor])]
        else:
            x = [np.average(input_data[0,c:c+resize_factor])]
            output_data = np.concatenate((output_data,x),axis=0)
        c += resize_factor
        
    output_data = np.delete(output_data,-1)
    return output_data

def CollapseBins_sum(input_data,resize_factor):
    n = max(np.shape(input_data))
    c = 0
    while c <= n:
        if c == 0:
            output_data = [np.average(input_data[0,c:c+resize_factor])]
        else:
            x = [sum(input_data[0,c:c+resize_factor])]
            output_data = np.concatenate((output_data,x),axis=0)
        c += resize_factor
        
    output_data = np.delete(output_data,-1)
    return output_data

