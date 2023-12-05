# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:34:37 2020

@author: Torbjørn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class staircasiness():
    def __init__(self, cond_window=(1e-2, 11)):
        self.cond_window = cond_window
            

    
    def window_loss(self,staircase,p=0.2, noise_eps=0):
        upper_lim=self.cond_window[1]
        lower_lim=self.cond_window[0]
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
    

            
    def window_histogram(self,staircase,linear_factor=0,p=3):
        """
        Project conductance onto a histogram (ie. a density of y-values) between
        a minimum and maximum conductance, and punish uniform densities while
        rewarding large peaks in the histogram
        """
        
        
        # Check that there are datapoints between cond_window[0] 
        # and cond_window[1] conductance quanta
        if not ((staircase>self.cond_window[0]) & (staircase<self.cond_window[1])).any():
            return 1e4
        
        # Check that there are datapoints below cond_window[0] conductance quanta
        if not ((staircase<self.cond_window[0])).any():
            return 1e4
        
        # Check that there are datapoints above cond_window[1] conductance quanta
        if not ((staircase>self.cond_window[1])).any():
            return 1e4
        
        mask=(staircase>self.cond_window[0]) & (staircase<self.cond_window[1])
        staircase=staircase[mask]
        
        num_bins=100
    
        hist,bins=np.histogram(staircase,num_bins,density=False)
        
        width=bins[1]-bins[0]
        
        loss=np.sum(abs(np.diff(hist*width)+linear_factor)**p)
                
        return 1/loss
    
    
    def multiple_windows_histogram(self,staircase,linear_factor=0,p=3):
        
        # Check that there are datapoints between cond_window[0] 
        # and cond_window[1] conductance quanta
        if not ((staircase>self.cond_window[0]) & (staircase<self.cond_window[1])).any():
            return 1e4
        
        # Check that there are datapoints below cond_window[0] conductance quanta
        if not ((staircase<self.cond_window[0])).any():
            return 1e4
        
        # Check that there are datapoints above cond_window[1] conductance quanta
        if not ((staircase>self.cond_window[1])).any():
            return 1e4
        
        windows=np.arange(1,min(np.max(staircase),self.cond_window[1]+1),2)
        loss=0
        for i in range(len(windows)-1):

            mask=(staircase>=windows[i]) & (staircase<windows[i+1])
            if not mask.any():
                continue

            num_bins=20
        
            hist,bins=np.histogram(staircase[mask],num_bins,density=False)
            
            
            width=bins[1]-bins[0]
            
            loss+=np.sum(abs(np.diff(hist*width)+linear_factor)**p)


                
        return 1/loss


