# optimization
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
# general
import numpy as np
import matplotlib.pyplot as plt
import time
from qcodes.dataset.data_set import load_by_id
    
# import my functions

import sys
sys.path.append('../')
from pixel_array_sim import pixelarrayQPC
from staircasiness import staircasiness

import os.path
from os import path

# script
stairs=staircasiness(delta=0.1,last_step=5)

common_voltages=np.linspace(-8,-2,20)
global_ids=[]
prefix="Optimization/scipy_results/pixelarrayQPC/"

QPC=pixelarrayQPC(plot=False)
disorder=True
if disorder:
    QPC.U0=0.5

def measure(V2,V3,V4,V5,V6,V7,V8,V9,V10):
    QPC.V2=V2
    QPC.V3=V3
    QPC.V4=V4
    QPC.V5=V5
    QPC.V6=V6
    QPC.V7=V7
    QPC.V8=V8
    QPC.V9=V9
    QPC.V10=V10
    
    results=[]
    for v in common_voltages:
        QPC.V1=v
        QPC.V11=v
        results.append(QPC.transmission())
        
    return results

    
    
    
def func_to_minimize(x):
    global global_ids
    # x[0]=V2 , X[1]=tilt
    if len(global_ids)%10==0:
        print(len(global_ids))
    fname=prefix+"{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.npy".format(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])
    if path.isfile(fname):
        result=np.load(fname)
        print("[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}] - already existing ".format(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]))
    else:
        result=measure(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8])
        np.save(fname,result)
    res2=stairs.histogram(result)
    res=(res2)
    # print(res)
    global_ids.append(x)
    # print("V2:{:.2f} tilt:{:.2f} gives staircasiness:{:.2f}".format(x[0],x[1],res))
    return res

callback_steps=[]
def callbackf(x):
    global callback_steps
    callback_steps.append(x)



        
bound=(-1,1)
bounds=[bound for i in range(9)]


global_ids=[]

start=np.random.uniform(bound[0],bound[1],9)
print("starting from point:")
print(start)
# print("starting staircasiness: {:.2f}".format(func_to_minimize([start_V2,start_tilt])))



start_time=time.perf_counter()
result=minimize(fun=func_to_minimize,x0=start,method="L-BFGS-B",bounds=bounds,callback=callbackf,options={"maxfun":100,"eps":0.1})
# point,value,info=fmin_l_bfgs_b(func=func_to_minimize,x0=[start_V2,start_tilt],approx_grad=True,bounds=bounds,maxfun=100,epsilon=0.01,maxls=5,callback=callbackf)
stop_time=time.perf_counter()
print("time spent minimizing: {:.2f}".format(stop_time-start_time))

print(result)


        
def converter(x):
    if isinstance(x,list):
        return str(x)
    if isinstance(x,str):
        temp=str(x[1:-1])
        return [float(y) for y in temp.split()]
    
def plots2(ids):
    # plot everything
    # plt.figure(1)
    results=[]
    unique_ids=[]
    for x in ids:
        if not str(x) in unique_ids:
            fname=prefix+"_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}.npy".format(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[7])
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
    # plt.plot(results[np.argmin(scores)],label='best')
    print(np.argmin(scores))
    print(converter(unique_ids[np.argmin(scores)]))
    # plt.plot(results[np.argmax(scores)],label='worst')

    plt.xlabel("Outer Gates [V]")
    plt.ylabel('Conductance')
    plt.text(-3,1.2,"{:.2f}".format(scoreshist[0]),fontsize=18)
    plt.text(-4.1,3.5,"{:.2f}".format(scoreshist[1]),fontsize=18)
    for binline in hej.bins:
        plt.plot([-8,-2],[binline,binline],'k--',alpha=0.5)
    plt.plot(common_voltages,results[0],label='Start')
    plt.plot(common_voltages,results[-1],label='Finish')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    

    
    
    # plt.figure(3)
    # plt.imshow(np.rot90(np.load('staircasiness_result_third.npy')),extent=(-2,2,-2,2),origin='lower')
    # plt.xlabel('V2')
    # plt.ylabel('tilt')
    # plt.colorbar()
    # plot path in staircasinessmap
    # for x in callback_steps:
    #     fname1="scipy_results/V2_{:.2f}_tilt_{:.2f}.npy".format(x[0],x[1])
    #     plt.figure(4)
    #     plt.plot(np.load(fname1))
    #     plt.figure()
    #     plt.plot(x[0],x[1])

    
plots2(global_ids)


        
# print(point)
# print(value)
# print(info)
