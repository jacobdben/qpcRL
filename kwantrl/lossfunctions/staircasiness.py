# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:34:37 2020

@author: Torbjørn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class staircasiness():
    def __init__(self,delta=0.05,last_step=20,favorite=100):
        self.delta=delta
        self.bins=[]
        self.last_step=last_step
        if isinstance(favorite,int):
            favorite=[favorite]
        self.favorite=favorite
        for i in range(last_step):
            self.bins.extend([i+1-delta,1+i+delta])
            
        self.arange=np.arange(1,last_step)
            

    
    def window_loss(self,staircase,p=0.2, noise_eps=0):
        upper_lim=7
        lower_lim=1e-2
        if staircase[0]>upper_lim:
            return 1
        if (staircase<upper_lim).all() or (staircase>lower_lim).all():
            return 1
        
        small = np.where(staircase < lower_lim)[0]
        large = np.where(staircase > upper_lim)[0]
        if small.shape[0] > 0:
            small = small[-1]
        else:
            small = 0
        
        if large.shape[0] > 0:
            large = large[0]
        else:
            large = staircase.shape[0]
        
        numel = large - small
        diff = (staircase[large - 1] - staircase[small])
        res = 0
        for i in range(small, large - 1):
            x = (staircase[i+1] - staircase[i])/diff
            res += np.abs(x+noise_eps/numel)**p #
            
        if numel==1:
            return 2
        else:
            return res * (1.0/(numel-1))**(1-p)
    
    
    def L_1_regularization(self,x,lamb):
        return lamb*np.sum(abs(np.array(x))) 
        
    def L_2_regularization(self,x,lamb):
        return lamb*np.sum(np.array(x)**2)
    

            
    def window_histogram(self,staircase,linear_factor=0,p=3,plot=False,ax=None):
        if not ((staircase>1e-2) & (staircase<11)).any():
            return 1e4
        
        if not ((staircase<1e-2)).any():
            return 1e4
        
        if not ((staircase>11)).any():
            return 1e4
        
        mask=(staircase>1e-2) & (staircase<11)
        staircase=staircase[mask]
        
        num_bins=100
    
        hist,bins=np.histogram(staircase,num_bins,density=True)
        
        width=bins[1]-bins[0]
        
        loss=np.sum(abs(np.diff(hist*width)+linear_factor)**p)
        if plot:
            if ax!=None:
                bin_mids=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
                ax.plot(bin_mids,hist*width,label="%.3f"%loss)
                
        return 1/loss


