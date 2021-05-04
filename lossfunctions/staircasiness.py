# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:34:37 2020

@author: Torbjørn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def invdist_to_plateau(staircase,plateau):
    dists=np.abs(staircase-plateau)
    invdists=1-dists
    return np.fmax(0,invdists)

def make_gauss_fit(staircase,plateau,plot=False):
    y=invdist_to_plateau(staircase, plateau)
    x=np.arange(len(y))
    
    n = len(x)                          
    mean = sum(x*y)/n                   
    sigma = sum(y*(x-mean)**2)/n        
    
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])
    if plot:
        plt.plot(x,y,'b+:',label='data')
        plt.plot(x,gaus(x,*popt),'ro:',label='fit')
        plt.legend()
        plt.title('Fit for plateau {}'.format(plateau))
        plt.xlabel('index')
        plt.ylabel('max(0,dist_to_plateau)')
        plt.text(1,0.8,"sigma={:.2f}".format(popt[2]))
        # plt.savefig('Optimization/Fit for plateau {}.png'.format(plateau))
        plt.show()
    return abs(popt[2])
    




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
            
    def histogram(self,staircase):
                   
        test=np.histogram(staircase,self.bins)[0]
        multiplier=np.zeros(len(test))
        multiplier[range(0,len(multiplier),2)]=0.1
        score=sum(test*multiplier)+1
        
        return 1/score
    
    def gaussian_fit(self,staircase):
        score=1
        highest_plateau=int(np.floor(np.max(staircase)))
        for plateau in np.arange(1,highest_plateau):
            score+=make_gauss_fit(staircase,plateau,plot=True)
        return 1/score
    
    def deriv_metric(self,staircase):
        res=0
        for i in range(len(staircase)-1):
            res += np.sqrt(np.abs(staircase[i+1]-staircase[i]))
        res /= np.sqrt(np.max(staircase)-np.min(staircase))
        return res
    
    def step_loss(self,last_transmission,transmission):
        indexs=np.digitize(np.array([last_transmission,transmission]),self.bins)
        # print(indexs)
        if indexs[0]%2==1:
            if indexs[1]==indexs[0]:
                # print("same")
                return abs(np.round(last_transmission)-transmission)

            elif indexs[1]>=indexs[0]:
                # print("above")
                return abs((np.round(last_transmission)+1-transmission))
            
            else:
                # print("below")
                return 10
        else:
            ceil=np.ceil(last_transmission)
            if transmission>=(ceil+self.delta):
                return 10
            else:
                return abs(ceil-transmission)

            
    

    



if __name__=="__main__":   
        
    y=np.array([0.97460391, 0.96824428, 0.97574253, 0.9857496 , 1.0071998 ,
                  1.38298989, 1.97786701, 1.98512252, 1.97990789, 1.98450759,
                  2.02895031, 2.93185717, 2.97778235, 2.97963773, 2.98609815,
                  3.42125114, 3.9582292 , 3.9812698 , 4.03072006, 4.9354933 ])
    




    # plot_nearest(1,1)    
    # x=[0.19334913, 0.12763539] #really good
    # x=[0.2,0.1]
    # fname="qcodesCHECK/scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(x[0],x[1])
    # fname=plot_nearest(x[0],x[1])
    # staircase=np.load(fname)
    staircase=y
    t=staircasiness(delta=0.1,last_step=int(np.ceil(np.max(staircase))))
    test = t.histogram(staircase)
    test2= t.gaussian_fit(staircase)

    # results=[]
    # for i in np.arange(0,len(y)-1):
    #     results.append(t.step_loss(reee[i], reee[i+1]))

    # plt.figure()
    # plt.plot(results,'*')
    # plt.plot(reee,'*')
    
    # print(t.step_loss(1, 0.9))
    plot=True
    if plot:
        plt.figure()
        for binline in t.bins:
            plt.plot([0,len(staircase)],[binline,binline],'k--',alpha=0.5)
        plt.plot(staircase,'-*')
        plt.xticks([])
        plt.yticks([1,2,3,4],['1','2','3','4'])
        plt.title("histogram={:.2f}, Gaussian fit={:.2f}".format(test,test2))
    # plt.savefig('Optimization/staircase_for_2_optimizations.png')
              # ,fontdict={'color':'blue','size':16})
