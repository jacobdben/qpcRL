import os
import re
import numpy as np
# [+-]?\d{1}.\d{2}

import matplotlib.pyplot as plt


path="optimization/scipy_results/"

def get_file_names(qpctype="pixelarrayQPC"):
    return list(os.walk(path+qpctype+"/"))[0][2]
    
def load_data(voltages,qpctype="pixelarrayQPC"):
    fname="Optimization/scipy_results/"+qpctype+"/"
    for V in voltages:
        fname+='{:.2f}_'.format(V)
    fname=fname[:-1]+'.npy'
    return np.load(fname)

def save_data(result,voltages,qpctype="pixelarrayQPC"):
    fname="Optimization/scipy_results/"+qpctype+"/"
    for V in voltages:
        fname+='{:.2f}_'.format(V)
    fname=fname[:-1]+'.npy'
    np.save(fname,result)
    
def load_closest_data(voltages,qpctype="pixelarrayQPC"):
    all_files=get_file_names(qpctype)
    arr=[]
    for fname in all_files:
        arr.append(fname[:-4].split('_'))
    arr=np.array(arr,dtype=float)
    idx=np.linalg.norm(arr-np.array(voltages),axis=1).argmin()
    return load_data(arr[idx,:],qpctype)
    



# FOR RENAMING
# for name in file_list:
#     os.rename(path+"pixelarrayQPC/"+name,path+"pixelarrayQPC/"+name[1])


    