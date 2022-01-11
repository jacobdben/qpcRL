# optimization
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
# general
import numpy as np
import matplotlib.pyplot as plt
import time
from qcodes.dataset.data_set import load_by_id
    
# import my functions
from qcodesCHECK import *
import sys
sys.path.append('../')
from staircasiness import staircasiness

import os.path
from os import path


# script
stairs=staircasiness(delta=0.1,last_step=5)

common_voltages=np.linspace(-5,-1,30)
global_ids=[]
def func_to_minimize(x):
    global global_ids
    # x[0]=V2 , X[1]=tilt
    fname="scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(x[0],x[1])
    if path.isfile(fname):
        result=np.load(fname)
        print("[{:.2f},{:.2f}] - already existing ".format(x[0],x[1]))
    else:
        result=qcmeas(x[0],x[1],common_voltages)[0]
        np.save(fname,result)
    res2=stairs.histogram(result)
    res=1/(res2)
    # print(res)
    global_ids.append(x)
    # print("V2:{:.2f} tilt:{:.2f} gives staircasiness:{:.2f}".format(x[0],x[1],res))
    return res

callback_steps=[]
def callbackf(x):
    global callback_steps
    callback_steps.append(x)
    fname="scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(x[0],x[1])
    if path.isfile(fname):
        result=np.load(fname)
        print("[{:.2f},{:.2f}] - already existing [CALLBACK] ".format(x[0],x[1]))
    else:
        result=qcmeas(x[0],x[1],common_voltages)[0]
    res2=stairs.histogram(result)
    if res2!=0:
        res=1/res2
        
    else:
        res=20
    # np.save("scipy_results/V2_{:.2f}_tilt_{:.2f}_score_{:.2f}_id_{}".format(x[0],x[1],res,dataid),result)
    # print("V2_{:.2f}_tilt_{:.2f}_score_{:.2f}_id_{}".format(x[0],x[1],res,dataid))

        
bounds=[(-0.5,0.5),(-0.5,0.5)]
make_map=True
if make_map:
    start_time=time.perf_counter()
    vals=np.empty([20,20])
    en=[]
    i=0
    
    for v2 in np.linspace(bounds[0][0],bounds[0][1],20):
        j=0
        for tilt in np.linspace(bounds[1][0],bounds[1][1],20):

            vals[i,j]=func_to_minimize((v2,tilt))
            en.append([v2,tilt])
            j+=1
        i+=1
            
            
    plt.imshow(vals,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
    plt.colorbar()
    plt.xlabel("tilt")
    plt.ylabel("V2")
    stop_time=time.perf_counter()
    print("time spent mappping: {:.2f}".format(stop_time-start_time))
    
    best=np.argmin(vals)
    
    fname="scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(en[best][0],en[best][1])
    result=np.load(fname)
    plt.figure()
    plt.plot(result)
    
def plot_nearest(V2,tilt,arr=np.array(en)):
    idx=(np.sqrt((arr[:,0]-V2)**2+(arr[:,1]-tilt)**2)).argmin()
    fname="scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(arr[idx,0],arr[idx,1])
    result=np.load(fname)
    plt.figure()
    plt.plot(result)
    
    

"""
global_ids=[]

start_V2=np.random.uniform(bounds[0][0],bounds[0][1])
start_tilt=np.random.uniform(bounds[1][0],bounds[1][1])
start_V2=-0.3
print("starting from point: {:.2f},{:.2f}".format(start_V2,start_tilt))
print("starting staircasiness: {:.2f}".format(func_to_minimize([start_V2,start_tilt])))



start_time=time.perf_counter()
result=minimize(fun=func_to_minimize,x0=np.array([start_V2,start_tilt]),method="L-BFGS-B",bounds=bounds,callback=callbackf,options={"maxfun":100,"eps":0.1})
# point,value,info=fmin_l_bfgs_b(func=func_to_minimize,x0=[start_V2,start_tilt],approx_grad=True,bounds=bounds,maxfun=100,epsilon=0.01,maxls=5,callback=callbackf)
stop_time=time.perf_counter()
print("time spent minimizing: {:.2f}".format(stop_time-start_time))

print(result)

def plots(ids):
    plt.figure()
    for dataid in ids:
        ds = load_by_id(dataid)
        x=ds.get_parameter_data('inst_transmission')['inst_transmission']['inst_transmission']
        plt.plot(x)
        
        
def converter(x):
    if isinstance(x,list):
        return str(x)
    if isinstance(x,str):
        temp=str(x[1:-1])
        # print(temp)
        # print(temp.split(sep=','))
        return [float(y) for y in temp.split()]
    
def plots2(ids):
    # plot everything
    # plt.figure(1)
    results=[]
    unique_ids=[]
    for x in ids:
        if not str(x) in unique_ids:
            fname="scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(x[0],x[1])
            results.append(np.load(fname))
            unique_ids.append(str(x))

    # print(unique_ids)
    # plot selected staircases 
    scores=[]
    for result in results:
        # plt.plot(result)
        invscore=stairs.histogram(result)
        scores.append(1/invscore)
        
    plt.figure(2)
    plt.plot(results[np.argmin(scores)],label='best')
    print(np.argmin(scores))
    print(converter(unique_ids[np.argmin(scores)]))
    plt.plot(results[np.argmax(scores)],label='worst')
    plt.plot(results[0],'--',label='first')
    plt.plot(results[-1],'--',label='last')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    

    
    
    plt.figure(3)
    plt.imshow(np.rot90(np.load('staircasiness_result_third.npy')),extent=(-2,2,-2,2),origin='lower')
    plt.xlabel('V2')
    plt.ylabel('tilt')
    plt.colorbar()
    # plot path in staircasinessmap
    for x in callback_steps:
        fname1="scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(x[0],x[1])
        plt.figure(4)
        plt.plot(np.load(fname1))
        plt.figure()
        plt.plot(x[0],x[1])

    
plots2(global_ids)


        
# print(point)
# print(value)
# print(info)
"""