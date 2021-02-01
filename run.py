

import numpy as np
import matplotlib.pyplot as plt
import time
import os


from simulations.pixel_array_sim_2 import pixelarrayQPC
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler, load_cma_output
from optimization.newpoint import new_point
from optimization.cma import optimize_cma
from optimization.gradientdescent import optimize_gradient
from plotting.plotting import plot_run,plot_potentials
# Common voltage sweep
start=-2
stop=0
steps=30

start2=-2
stop2=0
steps2=60

# Parameters for QPC
disorder=0.1
outer_gates=-4
B_field=0

disorder2=0.1
outer_gates2=-4
B_field2=0.05

# Parameters for optimization algorithm
bounds=(-1,1)
pfactor=0.001

# Initialize loss function
stairs=staircasiness(delta=0.05,last_step=20)

# Set common voltage sweep
common_voltages=np.linspace(start,stop,steps)
common_voltages2=np.linspace(start2,stop2,steps2)


# Initialize QPC instance and set parameters
QPC=pixelarrayQPC(plot=False,disorder_type='regular')
QPC.U0=disorder
QPC.V1=outer_gates
QPC.V11=outer_gates
QPC.phi=B_field

QPC2=pixelarrayQPC(plot=False,disorder_type='regular')
QPC2.U0=disorder2
QPC2.V1=outer_gates2
QPC2.V11=outer_gates2
QPC2.phi=B_field2



# Initialize Datahandler
dat=datahandler('pixel_disorder',QPC=QPC)


# Define the function we want to minimize, 
# in this case, its the voltage on individual pixels, constrained by their average=0
# and bounded so each pixel must be within avg_gates-/+ bounds[0/1]
def func_to_minimize(x):
    # new_point ensures points are valid within bounds and constraints.
    x_projected,penalty=new_point(x,bounds=bounds)
    
    result=[]
    for avg_gates in common_voltages:
        QPC.set_all_pixels(x_projected+avg_gates)
        result.append(QPC.transmission())
    return pfactor*penalty+stairs.histogram(result)


def func_to_minimize2(x):
    # new_point ensures points are valid within bounds and constraints.
    x_projected,penalty=new_point(x,bounds=bounds)
    
    result=[]
    for avg_gates in common_voltages2:
        QPC2.set_all_pixels(x_projected+avg_gates)
        result.append(QPC2.transmission())
    return pfactor*penalty+stairs.histogram(result)



def fitness_check(reported_fitness,reported_xs):
    new_fitness=[]
    for i in range(len(reported_fitness)):
        x_projected,penalty=new_point(reported_xs[i,:],bounds=bounds)
        
        result=[]
        for avg_gates in common_voltages:
            QPC.set_all_pixels(x_projected+avg_gates)
            result.append(QPC.transmission())
            
        new_fitness.append(pfactor*penalty+stairs.histogram(result))
    new_fitness=np.array(new_fitness)
    if not (new_fitness==reported_fitness).all():
        print("error in reported vs new fitness") 
    path=dat.data_path
    np.savetxt(path+"/fitness_check.txt",np.vstack([reported_fitness,new_fitness]).T)


if __name__=="__main__":
    
    #optimize with cma
    xbest,es=optimize_cma(func_to_minimize,dat,maxfevals=10)
    
    
    #optimize with gradient descent
    # xbest2=optimize_gradient(func_to_minimize,dat,bounds=bounds,maxiter=5)
    
    
    # alternate_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data/clusterruns"
    # plot_run(QPC, alternate_path,7, common_voltages, bounds, stairs,pfactor)
    # plot_potentials(QPC, dat.data_path, 21, common_voltages, bounds, staircasiness=stairs, pfactor=pfactor)
    # QPC.set_all_pixels(0)
    # QPC.plot_potential()
    # fitness,recentbestxs,xbest=load_cma_output(alternate_path,7)
    # fitness_check(fitness,recentbestxs)
    
    
    
    
