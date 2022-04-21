

import numpy as np
import matplotlib.pyplot as plt
import time
import os


from simulations.pixel_array_sim_2 import pixelarrayQPC
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler
from optimization.newpoint import new_point
from optimization.cma2 import optimize_cma
from optimization.gradientdescent import optimize_gradient
# from plotting.plotting import plot_run,plot_potentials
# Common voltage sweep
start=-3
stop=0
steps=30

# start2=-2
# stop2=0
# steps2=60

# Parameters for QPC
disorder=0.1
outer_gates=-4
B_field=0

# disorder2=0.1
# outer_gates2=-4
# B_field2=0.05

# Parameters for optimization algorithm
bounds=(-1,1)
pfactor=0.001

# Initialize loss function
stairs=staircasiness(delta=0.05,last_step=20)

# Set common voltage sweep
common_voltages=np.linspace(start,stop,steps)
# common_voltages2=np.linspace(start2,stop2,steps2)


# Initialize QPC instance and set parameters
QPC=pixelarrayQPC(plot=False,disorder_type='regular')
QPC.U0=disorder
QPC.V1=outer_gates
QPC.V11=outer_gates
QPC.phi=B_field

# QPC2=pixelarrayQPC(plot=False,disorder_type='regular')
# QPC2.U0=disorder2
# QPC2.V1=outer_gates2
# QPC2.V11=outer_gates2
# QPC2.phi=B_field2



# Initialize Datahandler
dat=datahandler('_________',QPC=QPC)


    
# Define the function we want to minimize, 
# in this case, its the voltage on individual pixels, constrained by their average=0
# and bounded so each pixel must be within avg_gates-/+ bounds[0/1]
def func_to_minimize(x,table): #x len 8

    voltages,penalty=new_point(x,bounds)
    
    result=[]
    for avg in common_voltages:
        QPC.set_all_pixels(voltages+avg)
        result.append(QPC.transmission())
    
    
    val=stairs.deriv_metric_zeros1(result)+stairs.L_1_regularization(voltages,0.01)
    
    key=table['next_key']
    table['next_key']+=1
    
    table['measurements'][key]={'val':val+penalty*pfactor,'staircase':result,'voltages':voltages.tolist()}
    
    return val+penalty*pfactor

baseline={'next_key':0,'measurements':{}}
func_to_minimize(np.zeros(9),baseline)
#%%

xbest,es,run_id=optimize_cma(func_to_minimize,dat,start_point=np.zeros(9),stop_time=44*3600,options={'tolx':1e-3})

