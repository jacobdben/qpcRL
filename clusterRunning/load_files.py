# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:08:36 2020

@author: Torbjørn
"""

import numpy as np
import matplotlib.pyplot as plt
from ClusterPoisson.staircasiness import staircasiness

def staircasiness2(transmission, p=0.5,lam=1,penalty=1):
    pen=0
    for val in transmission[5:]:
        if val<0.1:
            pen+=penalty
        else: 
            break
    
    R = transmission[-1] - transmission[0]
    f=lam*R + np.sum( np.sqrt(np.abs(np.diff(transmission)))/np.sqrt(R))
    return f#+pen

t=staircasiness()
staircasiness3=t.histogram

from os import walk
prefix="ClusterPoisson/thirdresults/"
#test=walk()
plt.figure()
vals=np.zeros([20,20])
ii=0
for i in np.arange(-2,2,0.2):
    jj=0
    for j in np.arange(-2,2,0.2):
#        print(i)
#        if i==0.0:
        try:
            if round(i,1)==0:
                print("hej")
                i=-0.0
            if round(j,1)==0:
                j=-0.0
            filename=prefix+"V2_{:.1f}_tilt_{:.1f}.npy".format(i,j)
            result=np.load(filename)
        except:
            if round(i,1)==0:
                print("hej")
                i=0.0
            if round(j,1)==0:
                j=0.0
            filename=prefix+"V2_{:.1f}_tilt_{:.1f}.npy".format(i,j)
#            print(filename)
#            print("not found") #vals[ii,jj]=0
            result=np.load(filename)
            if ii==0:
                plt.plot(result)
        stairs=staircasiness3(result)
        vals[ii,jj]=1/stairs
        print(stairs)

        jj+=1
    ii+=1
    
np.save("staircasiness_result_third",vals)
plt.figure()
plt.imshow(vals,origin='lower',extent=[-2,2,-2,2])
plt.xlabel('tilt',fontsize=18)
plt.ylabel('V2',fontsize=18)
plt.colorbar()
plt.savefig("newcasinessmap.pdf")

#f = []
#for (dirpath, dirnames, filenames) in walk(prefix):
#    f.extend(filenames)
#    break

#common_mode_voltages=np.linspace(-5,0.5,30)
#for name in f[3:]:
#    test=np.load("results/"+name)
#    test2=(test-min(test))/(max(test)-min(test))
#    
#    plt.figure(1)
#    plt.plot(common_mode_voltages,test,label=str(staircasiness2(test)))
#    
#    plt.figure(2)
#    plt.plot(common_mode_voltages,test2,label=str(staircasiness2(test2)))
#    
#plt.figure(1)
#plt.title("not normalized")
#plt.legend()
#plt.figure(2)
#plt.title("normalized")
#plt.legend()


