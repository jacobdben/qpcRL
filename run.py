

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

# Common voltage sweep
start=-2
stop=0
steps=30

# Parameters for QPC
disorder=0.1
outer_gates=-4

# Parameters for optimization algorithm
bounds=(-1,1)
pfactor=0.001

# Initialize loss function
stairs=staircasiness(delta=0.05,last_step=20)

# Set common voltage sweep
common_voltages=np.linspace(start,stop,steps)

# Initialize QPC instance and set parameters
QPC=pixelarrayQPC(plot=False)
QPC.U0=disorder
QPC.V1=outer_gates
QPC.V11=outer_gates

# Initialize Datahandler
dat=datahandler('restructure_runs',QPC=QPC)


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


#optimize with cma
xbest=optimize_cma(func_to_minimize,dat,maxfevals=500)
# xbest_projected=new_point(xbest,bounds)

#optimize with gradient descent
# xbest2=optimize_gradient(func_to_minimize,dat,bounds=bounds,maxiter=5)


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

# alternate_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data/clusterruns"
fitness,recentbestxs=load_cma_output()
fitness_check(fitness,recentbestxs)
