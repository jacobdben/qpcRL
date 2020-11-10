# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:34:37 2020

@author: Torbjørn
"""

import numpy as np

class staircasiness():
    def __init__(self,delta=0.1,last_step=3):
        self.bins=[]
        for i in range(last_step):
            self.bins.extend([i+1-delta,1+i+delta])
            
    def histogram(self,staircase,favorite=100):
        res=1   
        # print(np.digitize(staircase,self.bins))
        if isinstance(favorite,int):
            favorite=[favorite]
        for binnr in np.digitize(staircase,self.bins):
            if (binnr%2) ==1:
                # print((binnr+1)/2)
                if (binnr+1)/2 in favorite:
                    # print("fav")
                    res+=0.3
                else:
                    res+=0.1
        return res
    
if __name__=="__main__":    
    t=staircasiness()
    test=np.linspace(0,5,15)
    test2=[0,0.4,0.95,1,1.5,1.5,1.5,1.5,2,2]
    # print(t.bins)
    
    # print(test)
    
    print(t.histogram(test2,favorite=2))