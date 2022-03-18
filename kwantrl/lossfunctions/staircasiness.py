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
    


def make_staircase_fit(staircase,plot=False):
    def staircase_func(h,w,a,x):
        return h*(1/2*np.cosh(a/2)/np.sinh(a/2)*np.tanh(a*((x/w-np.floor(x/w))-0.5)) + 1/2 + np.floor(x/w))
    
    def linear_func(a2,b,x):
        return a2*x+b
    
    def fit_func(x,height,width,a,a2,b,xs,xs2):
        if isinstance(x, np.ndarray):
            datalist=[]
            datalist.extend(np.zeros(len(x[x<xs])))
            datalist.extend(staircase_func(height,width,a,x[(xs<=x) & (x<xs2)]))
            datalist.extend(linear_func(a2,b,x[x>=xs2]))
            return datalist
        # elif isinstance(x,float):
        #     if x<xs:
        #         return staircase_func(height,width,a,x)
        #     else:
        #         return linear_func(a2,b,x)
            
    popt,pcov = curve_fit(fit_func,np.arange(len(staircase)),staircase,p0=[2.5,30,300,0.2,-15,25,175])
    if plot:
        fig,ax=plt.subplots()
        ax.plot(fit_func(np.arange(len(staircase)),*popt))
        ax.plot(staircase)
    return popt,pcov

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
    
    def deriv_metric_zeros1(self,staircase):
        res=0
        zero_count=0
        for i in range(len(staircase)-1):
            res += np.sqrt(np.abs(staircase[i+1]-staircase[i]))
            if staircase[i]<=1e-5: # added afterwards
                    zero_count+=1
                
        res /= np.sqrt(np.max(staircase)-np.min(staircase))
        res/=len(staircase)-zero_count
        return res
    
    def deriv_metric_cube_addsmall(self,staircase):
        res=0
        for i in range(len(staircase)-1):
            res += np.cbrt(np.abs(staircase[i+1]-staircase[i])+0.01)

        return res
    
    def stairLossFunk(self,staircase):
        Res=0
        for i in range(len(staircase)-1):
              Res+=np.cbrt(np.abs(staircase[i+1]-staircase[i])+0.01)
        
        P_zeros=len(np.where(staircase<1e-5)[0])/len(staircase)
        
        return Res*P_zeros
    
    def stairLossFunk2(self,staircase):
        res=0
        for i in range(len(staircase)-1):
              res+=np.cbrt(np.abs(staircase[i+1]-staircase[i])+0.01)
        
        P_not_zeros=len(np.where(staircase>1e-1)[0])/len(staircase)
        
        return res/(P_not_zeros+1)
    
    def deriv_metric_cube_addsmall_zeros(self,staircase):
        res=0
        # zero_count=0
        for i in range(len(staircase)-1):
            if staircase[i]<=1e-5: # added afterwards
                continue
            res += np.cbrt(np.abs(staircase[i+1]-staircase[i])+0.01)
        # res*=len(np.where(staircase<=1e-3)[0])/len(staircase)
        res/=np.cbrt(np.max(staircase))
        return res
    
    def deriv_metric_original(self,staircase):
        res=0
        for i in range(len(staircase)-1):
            res += np.sqrt(np.abs(staircase[i+1]-staircase[i]))
             
        res /= np.sqrt(np.max(staircase)-np.min(staircase))
        
        return res
    
    def deriv_metric_cube_zeros(self,staircase):
        res=0
        zero_count=0
        for i in range(len(staircase)-1):
            res += np.cbrt(np.abs(staircase[i+1]-staircase[i]))
            if staircase[i]<=1e-5: # added afterwards
                    zero_count+=1
                
        res /= np.sqrt(np.max(staircase)-np.min(staircase))
        res/=len(staircase)-zero_count
        return res
    
    def deriv_metric_cube_mask(self,staircase,maskvals=[0.1,20]):
        mask=np.where((maskvals[0]<=staircase) & (staircase<=maskvals[1]))[0]
        mstaircase=staircase[mask]
        res=0
        for i in range(len(mstaircase)-1):
            res += np.cbrt(np.abs(mstaircase[i+1]-mstaircase[i]))
             
        res /= np.sqrt(np.max(staircase)-np.min(staircase))
        
        return res
    
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

    def multiple_windows_histogram(self,staircase,linear_factor=0,p=3,plot=False,ax=None):
        if not ((staircase>1e-2) & (staircase<11)).any():
            return 1e4
        
        if not ((staircase<1e-2)).any():
            return 1e4
        
        if not ((staircase>11)).any():
            return 1e4
        
        windows=np.arange(1,min(np.max(staircase),12),2)
        loss=0
        for i in range(len(windows)-1):

            mask=(staircase>=windows[i]) & (staircase<windows[i+1])
            if not mask.any():
                continue

            num_bins=20
        
            hist,bins=np.histogram(staircase[mask],num_bins,density=True)
            
            
            width=bins[1]-bins[0]
            
            loss+=np.sum(abs(np.diff(hist*width)+linear_factor)**p)

        # if plot:
        #     if ax!=None:
        #         bin_mids=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        #         ax.plot(bin_mids,hist*width,label="%.3f"%loss)
                
        return 1/loss

    

    



if __name__=="__main__":   
        
    y=np.array([0,0.97460391, 0.96824428, 0.97574253, 0.9857496 , 1.0071998 ,
                  1.38298989, 1.97786701, 1.98512252, 1.97990789, 1.98450759,
                  2.02895031, 2.93185717, 2.97778235, 2.97963773, 2.98609815,
                  3.42125114, 3.9582292 , 3.9812698 , 4.03072006, 4.9354933,13 ])
    
    y=np.linspace(0,15,100)



    # plot_nearest(1,1)    
    # x=[0.19334913, 0.12763539] #really good
    # x=[0.2,0.1]
    # fname="qcodesCHECK/scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(x[0],x[1])
    # fname=plot_nearest(x[0],x[1])
    # staircase=np.load(fname)
    staircase=y
    t=staircasiness(delta=0.1,last_step=int(np.ceil(np.max(staircase))))
    test = t.window_histogram(staircase)
    test2= t.multiple_windows_histogram(staircase)

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
        # for binline in t.bins:
        #     plt.plot([0,len(staircase)],[binline,binline],'k--',alpha=0.5)
        plt.plot(staircase,'-*')
        plt.xticks([])
        plt.yticks([1,2,3,4],['1','2','3','4'])
        plt.title("histogram={:.2f}, Gaussian fit={:.2f}".format(test,test2))
    # plt.savefig('Optimization/staircase_for_2_optimizations.png')
              # ,fontdict={'color':'blue','size':16})
