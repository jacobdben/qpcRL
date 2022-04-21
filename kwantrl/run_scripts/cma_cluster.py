from kwantrl.simulations.pixel_array import pixelarrayQPC
from kwantrl.optimization.trajectory_optimization import trajectory_func_to_optimize2
from kwantrl.optimization.cma import cma_p
from kwantrl.lossfunctions.staircasiness import staircasiness
from kwantrl.datahandling.datahandling import datahandler
import numpy as np
from functools import partial
from multiprocessing import cpu_count
import sys

num_cpu = int(sys.argv[1])
timeout = float(sys.argv[2])

W=int(sys.argv[3])
L=int(sys.argv[4])
scale=int(sys.argv[5])


start=-3
stop=5
steps=300

# Parameters for QPC
disorder=0.15
outer_gates=-3
B_field=0
energy=1

# Initialize loss function
stairs=staircasiness(delta=0.05,last_step=20)

# Set common voltage sweep
common_voltages=np.linspace(start,stop,steps)


# Initialize QPC instance and set parameters
QPC=pixelarrayQPC(W=W,L=L,plot=False,gate_kwargs={'scale':scale})

QPC.U0=disorder
QPC.energy=energy
QPC.V1=outer_gates
QPC.V11=outer_gates
QPC.phi=B_field

dat=datahandler()

order=2
start_point=np.zeros(shape=(order,8)).ravel()
kwargs={'common_mode':common_voltages,'QPC_instance':QPC,'order':order,'loss_function':stairs.window_loss,'bounds':(-5,7),'pfactor':0.001}
# actual_func_to_minimize=partial(trajectory_func_to_optimize2,**kwargs)
test=cma_p(trajectory_func_to_optimize2,function_args=kwargs,datahandler=datahandler(),starting_point=start_point,QPC=QPC, options=dict({'timeout':timeout, 'popsize':num_cpu}) )
# b=test.run()
