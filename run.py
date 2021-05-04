

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
dat=datahandler('ML_data_disorder_seed_3',QPC=QPC)

def make_ml_data(data_points):
    np.random.seed(2)
    for i in range(data_points):
        x=np.random.uniform(-1,1,9)
        x_projected,p=new_point(x,bounds=bounds)
        x_projected+=np.random.choice(common_voltages)
        if dat.check_measurement(x_projected):
            print("h")
        else:
            QPC.set_all_pixels(x_projected)
            dat.save_measurement(x_projected,QPC.transmission())
        if i%(data_points/20)==0:
            print("%.2f %%"%(i/data_points*100))
    
    

# Define the function we want to minimize, 
# in this case, its the voltage on individual pixels, constrained by their average=0
# and bounded so each pixel must be within avg_gates-/+ bounds[0/1]
def func_to_minimize(x):
    # new_point ensures points are valid within bounds and constraints.
    x_projected,penalty=new_point(x,bounds=bounds)
    
    result=[]
    for avg_gates in common_voltages:
        QPC.set_all_pixels(x_projected+avg_gates)
        res=QPC.transmission()
        result.append(res)
        dat.save_measurement(x_projected+avg_gates, res)
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
    # xbest,es=optimize_cma(func_to_minimize,dat,maxfevals=10)
    # dat.save_datahandler()
    
    #optimize with gradient descent
    # xbest2=optimize_gradient(func_to_minimize,dat,bounds=bounds,maxiter=5)
    
    data_points=5000
    start=time.perf_counter()
    make_ml_data(data_points)
    stop=time.perf_counter()
    print("total time: %.3f seconds"%(stop-start))
    dat.save_datahandler()
    
    
    # alternate_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data/clusterruns"
    # plot_run(QPC, alternate_path,9, common_voltages, bounds, stairs,pfactor)
    # plot_potentials(QPC, dat.data_path, 21, common_voltages, bounds, staircasiness=stairs, pfactor=pfactor)
    # QPC.set_all_pixels(0)
    # QPC.plot_potential()
    # fitness,recentbestxs,xbest=load_cma_output(alternate_path,7)
    # fitness_check(fitness,recentbestxs)
    
    
    
    
    #optimize with cma without disorder
    # QPC.disorder=0
    # xbest,es=optimize_cma(func_to_minimize,dat,maxfevals=10)
    # xbest_projected,penalty=new_point(xbest,bounds=bounds)
    
    #optimize with cma with solution from without disorder, with disorder
    # QPC.disorder=disorder
    # xbest2,es2=optimize_cma(func_to_minimize,dat,maxfevals=10,start_point=xbest_projected)
    # xbest2_projected,penalty2=new_point(xbest2,bounds=bounds)
    
    # with open(dat.data_path+"disorder_test.txt",mode='w') as file_object:
    #     print("no disorder xbest_projected",file=file_object)
    #     print(xbest_projected,file=file_object)
    #     print("with disorder xbest2_projected",file=file_object)
    #     print(xbest2_projected,file=file_object)
    #     print("difference xbest2_projected-xbest_projected",file=file_object)
    #     print(xbest2_projected-xbest_projected,file=file_object)
    