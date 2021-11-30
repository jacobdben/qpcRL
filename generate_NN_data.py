
from simulations.pixel_array_sim_2 import pixelarrayQPC
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler, save_optimization_dict, load_optimization_dict
from optimization.fourier.fourier_modes_hardcoded import fourier_to_potential, plot_fourier_modes
from optimization.cma2 import optimize_cma, resume_cma
from optimization.newpoint import new_point


import numpy as np
import matplotlib.pyplot as plt
import time
import sys

start=-7
stop=2
steps=300

# Parameters for QPC
disorder=0.3
outer_gates=-13
B_field=0
energy=3

# Parameters for optimization algorithm
bounds=(-1,1)
pfactor=0.001

# Set common voltage sweep
common_voltages=np.linspace(start,stop,steps)


# Initialize QPC instance and set parameters
QPC=pixelarrayQPC(plot=True,disorder_type='regular')
QPC.U0=disorder
QPC.energy=energy
QPC.V1=outer_gates
QPC.V11=outer_gates
QPC.phi=B_field


import json
Results={'next_key':0}
start_time=time.perf_counter()
run_time=3600*float(sys.argv[2])
while (time.perf_counter()-start_time)<run_time:
    # print((time.perf_counter()-start_time))
    Randoms=np.random.uniform(bounds[0],bounds[1],size=8)
    x0=[0]
    x0.extend(Randoms)
    _,voltages=fourier_to_potential(x0)
    voltages,penalty=new_point(voltages.ravel(),bounds)

    for avg in common_voltages:
        QPC.set_all_pixels(voltages+avg)
        Results[str(Results['next_key'])]={'random_vals':Randoms.tolist(),'voltages':voltages.tolist(),'conductance':QPC.transmission(),'common_mode':avg}
        Results['next_key']+=1

with open('../NNdata/run{}.json'.format(int(sys.argv[1])),'w') as file:
    json.dump(Results,file,indent=6)
