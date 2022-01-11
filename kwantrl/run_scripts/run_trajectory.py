#%%
from simulations.pixel_array import pixelarrayQPC
from optimization.trajectory_optimization import trajectory_func_to_optimize
from optimization.cma2 import optimize_cma
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler
#%%
import numpy as np
from functools import partial
import sys

start=-7
stop=2
steps=100

# Parameters for QPC
disorder=0.3
outer_gates=-13
B_field=0
energy=3

# Parameters for optimization algorithm
# bounds=(-1,1)
# pfactor=0.001

# Initialize loss function
stairs=staircasiness(delta=0.05,last_step=20)

# Set common voltage sweep
common_voltages=np.linspace(start,stop,steps)


# Initialize QPC instance and set parameters
QPC=pixelarrayQPC(plot=False)

QPC.U0=disorder
QPC.energy=energy
QPC.V1=outer_gates
QPC.V11=outer_gates
QPC.phi=B_field

dat=datahandler('fourier_modes',QPC=QPC)

order=2
start_point=np.zeros(shape=(order,8)).ravel()
kwargs={'common_mode':common_voltages,
        'QPC_instance':QPC,
        'order':order,
        'loss_function':stairs.stairLossFunk2,
        'bounds':(-9,4),
        'pfactor':0.001,
        'num_cpus':4}
actual_func_to_minimize=partial(trajectory_func_to_optimize,**kwargs)
result=optimize_cma(actual_func_to_minimize,dat,start_point,maxfevals=99999,sigma=0.5,stop_time=sys.argv[1]*3600)