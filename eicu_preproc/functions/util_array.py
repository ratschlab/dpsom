"""
Array processing utilities
"""

import numpy as np
import scipy.stats as sp_stats

def nan_ratio(in_arr):
    return np.sum(in_arr)/in_arr.size

def print_nan_stats(in_arr):
    print("[0.0: {:.5f}, 1.0: {:.5f}, NAN: {:.5f}]".format(np.sum(in_arr==0.0)/in_arr.size,np.sum(in_arr==1.0)/in_arr.size,
                                                           np.sum(np.isnan(in_arr))/in_arr.size))

def pos_ratio(in_arr):
    return np.sum(in_arr==1.0/in_arr.size)

def column_statistics_median_std(in_arr):
    median_arr=np.median(in_arr,axis=0)
    std_arr=np.std(in_arr,axis=0)
    for i in range(median_arr.size):
        print("Col {}: MED: {:.3f}, STD: {:.3f}".format(i,median_arr[i],std_arr[i]))

def first_last_diff(in_arr):
    return in_arr[-1]-in_arr[0]

def print_pos_stats(in_arr,desc_str):
    if np.sum(in_arr==1.0)>0:
        print(desc_str)

def empty_nan(sz):
    arr=np.empty(sz)
    arr[:]=np.nan
    return arr

def value_empty(size, default_val, dtype=None):
    if dtype is not None:
        tmp_arr=np.empty(size, dtype=dtype)
    else:
        tmp_arr=np.empty(size)
    tmp_arr[:]=default_val
    return tmp_arr

def clip_to_val(mat, threshold):
    mat[mat>threshold] = threshold
    mat[mat<-threshold] = -threshold
    return mat

def time_diff(t1, t2=None):
    if t2 is None:
        tdiff = np.diff(t1)
    else:
        if type(t1) == list:
            t1 = np.array(t1)
        if type(t2) == list:
            t2 = np.array(t2)
        tdiff = t2 - t1
    return tdiff

def array_mode(in_arr):
    if in_arr.size==1:
        return in_arr[0]
    counts,edges = np.histogram(in_arr,bins=50)
    max_idx=counts.argmax()
    midpoint=(edges[max_idx]+edges[max_ids+1])/2
    return midpoint




    
