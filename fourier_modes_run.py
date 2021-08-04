from simulations.pixel_array_sim_2 import pixelarrayQPC
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler, save_optimization_dict, load_optimization_dict
from optimization.fourier.fourier_modes_hardcoded import fourier_to_potential, plot_fourier_modes
from optimization.cma2 import optimize_cma, resume_cma
from optimization.newpoint import new_point


import numpy as np
import matplotlib.pyplot as plt
import time

start=-3
stop=0
steps=30

# Parameters for QPC
disorder=0.1
outer_gates=-4
B_field=0
energy=2

# Parameters for optimization algorithm
bounds=(-1,1)
pfactor=0.001

# Initialize loss function
stairs=staircasiness(delta=0.05,last_step=20)

# Set common voltage sweep
common_voltages=np.linspace(start,stop,steps)


# Initialize QPC instance and set parameters
QPC=pixelarrayQPC(plot=True,disorder_type='regular')
QPC.U0=disorder
QPC.energy=energy
QPC.V1=outer_gates
QPC.V11=outer_gates
QPC.phi=B_field

dat=datahandler('fourier_modes',QPC=QPC)

def func_to_minimize(x,table): #x len 8
    x0=[0]
    x0.extend(x)
    _,voltages=fourier_to_potential(x0)
    voltages,penalty=new_point(voltages.ravel(),bounds)
    
    result=[]
    for avg in common_voltages:
        QPC.set_all_pixels(voltages+avg)
        result.append(QPC.transmission())
    
    
    # val=stairs.deriv_metric_zeros1(result)+stairs.L_1_regularization(x0, 0.001)+stairs.L_2_regularization(x0, 0.001)
    val=stairs.simple_plateau(result,2)+stairs.L_1_regularization(x0, 0.001)+stairs.L_2_regularization(x0, 0.001)
    
    key=table['next_key']
    table['next_key']+=1
    
    table['measurements'][key]={'val':val+penalty*pfactor,'staircase':result,'x':x0,'voltages':voltages.ravel().tolist()}
    
    return val+penalty*pfactor



#%% start the optimization

xbest,es,run_id=optimize_cma(func_to_minimize,dat,start_point=np.random.uniform(-0.5,0.5,8),stop_time=3600,options={'tolx':1e-3})
   

#%% resume the optimization

# xbest,es,run_id=resume_cma(func_to_minimize,run_id=run_id,datahandler=dat,stop_time=14*3600)


#%% plot some results
"""
from datahandling.datahandling import load_cma_data

datadict=load_cma_data(run_id=49) #fourier modes run
datadict2=load_cma_data(run_id=54) #normal run

loss=[]
staircases=[]
voltages=[]
xs=[]
for key in datadict.keys():
    if not key=='next_key':
        loss.append(datadict[key]['val'])
        staircases.append(datadict[key]['staircase'])
        voltages.append(datadict[key]['voltages'])
        xs.append(datadict[key]['x'])

loss2=[]
staircases2=[]
voltages2=[]
for key in datadict2.keys():
    if key!='next_key' and key!='starting_point':
        loss2.append(datadict2[key]['val'])
        staircases2.append(datadict2[key]['staircase'])
        voltages2.append(datadict2[key]['voltages'])
        
baseline=datadict2['starting_point']
    
best_loss=np.argmin(loss)  
best_loss2=np.argmin(loss2)  
    
plt.figure()
plt.plot(loss,label='fourier')
plt.plot(loss2,label='normal')
plt.ylabel('loss',fontsize=18)
plt.xlabel('function call',fontsize=18)
plt.legend()


plt.figure()
plt.plot(common_voltages,staircases[best_loss],label='fourier: {:.3f}'.format(loss[best_loss]))
plt.plot(common_voltages,staircases2[best_loss2],label='normal: {:.3f}'.format(loss2[best_loss2]))
plt.plot(common_voltages,baseline['0']['staircase'],label='no opt: {:.3f}'.format(baseline['0']['val']))
plt.ylabel('avg voltage',fontsize=18)
plt.xlabel('conductance',fontsize=18)

plt.grid()
plt.legend()

plt.figure()
plt.title('fourier voltages',fontsize=18)
plt.imshow(np.array(voltages[best_loss]).reshape((3,3)),origin='lower')
plt.colorbar(label='V')


plt.figure()
plt.title('normal voltages',fontsize=18)
plt.imshow(np.array(voltages2[best_loss2]).reshape((3,3)),origin='lower')
plt.colorbar(label='V')
plt.show()

fig,axes=plot_fourier_modes(xs[best_loss])
"""
