
from scipy.optimize import minimize, LinearConstraint, fmin_l_bfgs_b, Bounds
import cvxpy as cp
import cma


# general
import numpy as np
import matplotlib.pyplot as plt
import time
import os
# import my functions
# import sys
# sys.path.append("../")
from simulations.pixel_array_sim_2 import pixelarrayQPC
from staircasiness import staircasiness
from datahandling import datahandler, save_optimization_dict
from newpoint import new_point

start=-2
stop=0
steps=30
disorder=0.1
outer_gates=-4
pfactor=0.001

stairs=staircasiness(delta=0.05,last_step=20)
common_voltages=np.linspace(start,stop,steps)

QPC=pixelarrayQPC(plot=False)

QPC.U0=disorder

QPC.V1=outer_gates
QPC.V11=outer_gates

dat=datahandler(experiment_name='cma-es3',QPC=QPC)


data_path=dat.data_path
counter={'new':0,'loaded':0}
def func_to_minimize(x):
    # new_point ensures points are valid within bounds and constraints.
    x,penalty=new_point(x,bounds=(-0.5,0.5))
    
    result=[]
    for avg_gates in common_voltages:
        if dat.check_measurement(x+avg_gates):
            result=dat.load_measurement(x+avg_gates)
            counter['loaded']+=1
        else:
            QPC.V2=x[0]+avg_gates
            QPC.V3=x[1]+avg_gates
            QPC.V4=x[2]+avg_gates
            QPC.V5=x[3]+avg_gates
            QPC.V6=x[4]+avg_gates
            QPC.V7=x[5]+avg_gates
            QPC.V8=x[6]+avg_gates
            QPC.V9=x[7]+avg_gates
            QPC.V10=x[8]+avg_gates
            result.append(QPC.transmission())
            dat.save_measurement(x+avg_gates, result)
            counter['new']+=1
    # plt.plot(result)
    return pfactor*penalty+stairs.histogram(result)
def folder_name():
    if not os.path.exists(data_path+"outcmaes"):
        os.mkdir(data_path+"outcmaes")
    folders=[x[1] for x in os.walk(data_path+"outcmaes/")]
    lis=[int(f) for f in folders[0]]
    lis.append(0)
    newfolder=data_path+'outcmaes/'+'{}/'.format(max(lis)+1)
    return newfolder

newfolder=folder_name()
x,es=cma.fmin2(func_to_minimize,np.zeros(9),0.5,options={'maxfevals':20,'verb_filenameprefix':newfolder})
dat.save_datahandler()


# x2=es.best.x
print(x)
# print(x2)
plot=False
if plot:
    x,p=new_point(x,bounds=(-0.5,0.5))
    
    print("result had penalty: {}".format(p))
    print(x)
    result=[]
    baseline=[]
    plt.figure()
    for avg_gates in common_voltages:
        QPC.set_all_pixels(x+avg_gates)
        result.append(QPC.transmission())
    plt.plot(common_voltages,result,label='Optimized:{:.3f}'.format(stairs.histogram(result)))

    
    for avg_gates in common_voltages:
        QPC.set_all_pixels(avg_gates)
        baseline.append(QPC.transmission())
    plt.plot(common_voltages,baseline,label='Non Optimized:{:.3f}'.format(stairs.histogram(baseline)))

    plt.xlabel('Avg Voltage [V]')
    plt.ylabel('Conductance')
    plt.grid()
    plt.legend()
    
    
    bounds=((19,41),(25,45))
    fig,ax=plt.subplots() 
    QPC.U0=disorder
    ax.set_title("Optimized") 
    QPC.set_all_pixels(x+common_voltages[0])
    t1,p1=QPC.plot_potential_section(bounds=bounds,ax=ax)
    plt.colorbar(p1)
    
    fig,ax=plt.subplots()
    ax.set_title("Non Optimized")
    QPC.U0=disorder
    QPC.set_all_pixels(common_voltages[0])
    t2,p2=QPC.plot_potential_section(bounds=bounds,ax=ax)
    plt.colorbar(p2)
    np.where(t1==t2)
    
    fig,ax=plt.subplots()
    ax.set_title("Non Optimized,no disorder")
    QPC.U0=0
    QPC.set_all_pixels(common_voltages[0])
    t2,p2=QPC.plot_potential_section(bounds=bounds,ax=ax)
    plt.colorbar(p2)
    np.where(t1==t2)
