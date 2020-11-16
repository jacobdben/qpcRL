import os
import re
import numpy as np
import pickle

import matplotlib.pyplot as plt
import sys
sys.path.append('../')

path="optimization/scipy_results/"

class datahandler():
    def __init__(self,qpctype="pixelarrayQPC"):
        self.qpctype=qpctype
        self.fname=path+self.qpctype+"/"

    def get_file_names(self):
        return list(os.walk(path+self.qpctype+"/"))[0][2]
        
    def check_data(self,voltages):
        fname=self.fname
        for V in voltages:
            fname+='{:.2f}_'.format(V)
        fname=fname[:-1]+'.npy'
        if os.path.isfile(fname):
            return True
        else:
            return False
           
    def load_data(self,voltages):
        fname=self.fname
        for V in voltages:
            fname+='{:.2f}_'.format(V)
        fname=fname[:-1]+'.npy'
        return np.load(fname)
    
    def save_data(self,result,voltages):
        fname=self.fname
        for V in voltages:
            fname+='{:.2f}_'.format(V)
        fname=fname[:-1]+'.npy' 
        np.save(fname,result)
        
    def load_closest_data(self,voltages):
        all_files=self.get_file_names(self.qpctype)
        arr=[]
        for fname in all_files:
            arr.append(fname[:-4].split('_'))
        arr=np.array(arr,dtype=float)
        idx=np.linalg.norm(arr-np.array(voltages),axis=1).argmin()
        return self.load_data(arr[idx,:],self.qpctype)
    
    def save_dict(self,dictionary,dictname):
        file=open(path+dictname+'.pkl',"wb")
        pickle.dump(dictionary,file)
        file.close

    def load_dict(self,dictname):
        file=open(path+dictname+'.pkl',"rb")
        output=pickle.load(file)
        file.close
        return output

# FOR RENAMING
# for name in file_list:
#     os.rename(path+"pixelarrayQPC/"+name,path+"pixelarrayQPC/"+name[1])


    