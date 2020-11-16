# optimization
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import fmin_l_bfgs_b
# general
import numpy as np
import matplotlib.pyplot as plt
import time
    
# import my functions

import sys
sys.path.append('../')
from simulations.pixel_array_sim import pixelarrayQPC
from staircasiness import staircasiness


# script
stairs=staircasiness(delta=0.1,last_step=5)

common_voltages=np.linspace(-1.5,0,20)
QPC=pixelarrayQPC(plot=False)
QPC.V1=-3
QPC.V11=-3

disorder=False
if disorder:
    QPC.U0=0.25

def measure(x,average_voltage):
    
    QPC.V2=average_voltage+x[0]
    QPC.V3=average_voltage+x[1]
    QPC.V4=average_voltage+x[2]
    QPC.V5=average_voltage+x[3]
    QPC.V6=average_voltage+x[4]
    QPC.V7=average_voltage+x[5]
    QPC.V8=average_voltage+x[6]
    QPC.V9=average_voltage+x[7]
    QPC.V10=average_voltage+x[8]
    
    return QPC.transmission()


    
def func_to_minimize(x):
    global average_voltage, last_trans

    transmission=measure(x, average_voltage)
    return stairs.step_loss(last_trans, transmission)
    







global_ids=[]

start=np.zeros(9)

results={'x':[],'trans':[]}
test=[]
constraint=LinearConstraint(np.ones([9,9]), np.zeros(9), np.zeros(9))
constraint2=LinearConstraint(np.eye(9),np.ones(9)*(-1),np.ones(9))

for i in range(len(common_voltages)):
    average_voltage=common_voltages[i]
    
    if i==0:
        last_trans=measure(start,common_voltages[0])
        results['x'].append(start)
        results['trans'].append(last_trans)
        
    elif i==1:
        result=minimize(fun=func_to_minimize,x0=start,method="trust-constr", constraints=constraint)
        last_trans=result.fun
        results['x'].append(result.x)
        results['trans'].append(result.fun)
        test.append(result)
    else:
        result=minimize(fun=func_to_minimize,x0=result.x,method="trust-constr", constraints=constraint)
        last_trans=result.fun
        results['x'].append(result.x)
        results['trans'].append(result.fun)
        test.append(result)
        
        
plt.plot(common_voltages,results['trans'])
        
        
        
        
    
    
    