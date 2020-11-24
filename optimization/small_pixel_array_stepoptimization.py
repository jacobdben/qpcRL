# optimization
from scipy.optimize import minimize, LinearConstraint, fmin_l_bfgs_b, Bounds
# from scipy.optimize import LinearConstraint
# from scipy.optimize import fmin_l_bfgs_b

# general
import numpy as np
import matplotlib.pyplot as plt
import time
    
# import my functions

import sys
sys.path.append('../')
from simulations.pixel_array_sim_2 import pixelarrayQPC
from staircasiness import staircasiness
from datahandling import datahandler, save_optimization_dict

def step_optimization(start=-1.5,stop=0,steps=50,experiment_name='small_pixel_array',outer_gates=-2,disorder=0.5):
    global last_trans
    # script
    stairs=staircasiness(delta=0.1,last_step=5)
    
    common_voltages=np.linspace(start,stop,steps)
    QPC=pixelarrayQPC(plot=False)
    QPC.V1=outer_gates
    QPC.V11=outer_gates
    
    

    QPC.U0=disorder
        
    dat=datahandler(experiment_name=experiment_name,QPC=QPC)
    # file_names=dat.get_file_names()
    counter={'already_measured':0,'new':0}
    def measure(x):
        # print(x)
        if dat.check_measurement(x):
            result=dat.load_measurement(x)
            counter['already_measured']+=1
        else:
            QPC.V2=x[0]
            QPC.V3=x[1]
            QPC.V4=x[2]
            QPC.V5=x[3]
            QPC.V6=x[4]
            QPC.V7=x[5]
            QPC.V8=x[6]
            QPC.V9=x[7]
            QPC.V10=x[8]
            result=QPC.transmission()
            dat.save_measurement(x,result)
            counter['new']+=1
        return result
    
        
    def func_to_minimize(x):
        global last_trans
        transmission=measure(x)
        return stairs.step_loss(last_trans, transmission)
        
    
    start=np.ones(9)*common_voltages[0]
    
    results={'x':[],'trans':[],'score':[],'optimizationresults':[]}
    test=[]
    
    bound=(-3,2)
    bounds=Bounds([bound[0] for i in range(9)],[bound[1] for i in range(9)])
    print("Point #,   Conductance,   Score,   Time")
    start_time=time.perf_counter()
    for i in range(len(common_voltages)):
        loop_time1=time.perf_counter()
        average_voltage=common_voltages[i]
        # constraint=LinearConstraint(np.ones([1,9])*1/9, average_voltage, average_voltage)
        # constraint2=LinearConstraint(np.eye(9),np.ones(9)*(-1),np.ones(9))
        
        if i==0:
            last_trans=measure(start)
            results['x'].append(start)
            results['trans'].append(last_trans)
            results['score'].append(1)
            results['optimizationresults'].append("point 0 not optimized")
            
            
        elif i==1:
            # result=minimize(fun=func_to_minimize,x0=start,method="trust-constr", constraints=constraint,bounds=bounds,options={'finite_diff_rel_step':0.01,'maxiter':30})
            result=minimize(fun=func_to_minimize,x0=start,method="SLSQP", constraints=[{'type':'eq','fun':lambda x: np.sum(x)/9-average_voltage}],bounds=bounds,options={'eps':0.01,'maxiter':10})
            last_trans=measure(result.x)
    
            results['x'].append(result.x)
            results['score'].append(result.fun)
            results['trans'].append(last_trans)
            results['optimizationresults'].append(result)
            
        else:
            # result=minimize(fun=func_to_minimize,x0=result.x,method="trust-constr", constraints=constraint,bounds=bounds,options={'finite_diff_rel_step':0.01,'maxiter':30})
            result=minimize(fun=func_to_minimize,x0=result.x,method="SLSQP", constraints=[{'type':'eq','fun':lambda x: np.sum(x)/9-average_voltage}],bounds=bounds,options={'eps':0.01,'maxiter':10})
            last_trans=measure(result.x)
    
            results['x'].append(result.x)
            results['score'].append(result.fun)
            results['trans'].append(last_trans)
            results['optimizationresults'].append(result)
            
        loop_time2=time.perf_counter()
        print("{} , {} , {} , {}".format(i,last_trans,results['score'][-1],(loop_time2-loop_time1)))
        # print("Conductance=%f "%last_trans)
        # print("Score=%f"%results['score'][-1])

        # print("time:%.2f"%(loop_time2-loop_time1))
    
    
    
    stop_time=time.perf_counter()
    print("total time spent optimizing: {:.1f}".format(stop_time-start_time))
    # fig,(ax1,ax2)=plt.subplots(2,1)
    basic=[]
    for V in common_voltages:
        basic.append(measure(V*np.ones(9)))
    fig,ax1=plt.subplots()
    ax1.plot(common_voltages,basic,label="Non Optimized")
    ax1.plot(common_voltages,results['trans'],label="Optimized")
    ax1.set_ylabel("Conductance")
    ax1.set_xlabel("Avg Pixel Voltage [V]")
    ax1.plot(common_voltages,results['trans'],'r*')
    ax1.grid('on')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.savefig('pixelstepoptimizationDISORDER5.png')
    
    
            
    save_optimization_dict(experiment_name,results)
        
    dat.save_datahandler()
    print(counter)
    plot=False
    save_plot=False
    if plot:
    
            
        num=0
        for x in results['x']:
            plt.figure()
            plt.title("avg V={:.2f}, step={}".format(sum(x)/9,num))
            plt.imshow(x.reshape(3,3),origin="lower",vmin=bound[0],vmax=bound[1])
            for i in range(3):
                for j in range(3):
                    plt.text(i,j,"{:.2f}".format(x.reshape(3,3).T[i,j]),horizontalalignment='center',color='white')
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.colorbar()
            if save_plot:
                plt.savefig('optimization/figures/small_pixel_array/{}.png'.format(num))
            num+=1
        
        
        
    
# def plot_dict(dictname):
#     results=dat.load_dict(dictname)
#     common_voltages=[]
#     for i in range(len(results['x'])):
#         common_voltages.append(sum(results['x'][i])/9)
        
        
#     basic=[]
#     for V in common_voltages:
#         basic.append(measure(V*np.ones(9)))
#     fig,ax1=plt.subplots()
#     ax1.plot(common_voltages,basic,label="Non Optimized")
#     ax1.plot(common_voltages,results['trans'],label="Optimized")
#     ax1.set_ylabel("Conductance")
#     ax1.set_xlabel("Avg Pixel Voltage [V]")
#     ax1.plot(common_voltages,results['trans'],'r*')
#     ax1.grid('on')
#     ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    
    
    