#%%
from simulations.pixel_array_sim_2 import pixelarrayQPC
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler, save_optimization_dict, load_optimization_dict
from optimization.fourier.fourier_modes_hardcoded import fourier_to_potential, plot_fourier_modes
from optimization.cma2 import optimize_cma, resume_cma
from optimization.newpoint import new_point


import numpy as np
import matplotlib.pyplot as plt
import time

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
    
    
    val=stairs.stairLossFunk2(np.array(result))
    # val=stairs.simple_plateau(result,2)+stairs.L_1_regularization(x0, 0.001)+stairs.L_2_regularization(x0, 0.001)
    
    key=table['next_key']
    table['next_key']+=1
    
    table['measurements'][key]={'loss':val+penalty*pfactor,'staircase':result,'x':x0,'voltages':voltages.ravel().tolist()}
    
    return val+penalty*pfactor

# start_time=time.perf_counter()
# result=[]
# for avg in common_voltages:
#     QPC.set_all_pixels(avg)
#     result.append(QPC.transmission())

# print('time %.3f'%(time.perf_counter()-start_time))
# plt.figure()
# plt.plot(common_voltages,result)

#%%
import json
Results={'next_key':0}
start_time=time.perf_counter()
run_time=100
while (time.perf_counter()-start_time)<run_time:
    # print((time.perf_counter()-start_time))
    Randoms=np.random.uniform(bounds[0],bounds[1],size=8)
    x0=[0]
    x0.extend(Randoms)
    # print(x0)
    _,voltages=fourier_to_potential(x0)
    voltages,penalty=new_point(voltages.ravel(),bounds)
    # print(voltages)

    for avg in common_voltages:
        QPC.set_all_pixels(voltages+avg)
        Results[str(Results['next_key'])]={'random_vals':Randoms.tolist(),'voltages':voltages.tolist(),'conductance':QPC.transmission(),'common_mode':avg}
        Results['next_key']+=1
save_folder=r'C:\Users\Torbjørn\Google Drev\UNI\MastersProject\EverythingkwantRL\NN_test_data/'
with open(save_folder+'test.json','w') as file:
    json.dump(Results,file,indent=6)

#%% start the optimization

# xbest,es,run_id=optimize_cma(func_to_minimize,dat,start_point=np.zeros(8),stop_time=48*3600,options={'tolx':1e-3})
   

#%% resume the optimization

# xbest,es,run_id=resume_cma(func_to_minimize,run_id=run_id,datahandler=dat,stop_time=14*3600)


#%% plot some results
#%% plot some results
# import matplotlib
# matplotlib.rcParams['figure.dpi']=200

# import json
# def load_dataids(outcmaes_run):
#     with open("C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data/outcmaes/{}/datadict.txt".format(outcmaes_run)) as file:
#         test=json.load(file)
#     return test
         
# data_dict=load_dataids(68) #run id
# losses=[]
# dataids_list=[]
# voltages=[]
# staircases=[]
# for key in range(len(data_dict['measurements'])):
#     losses.append(data_dict['measurements'][str(key)]['val'])
#     # dataids_list.append(data_dict['measurements'][str(key)]['dataid'])
#     voltages.append(data_dict['measurements'][str(key)]['voltages'])
#     staircases.append(data_dict['measurements'][str(key)]['staircase'])
# # loss=[dataiddict[key]['loss'] for key in dataiddict.keys()]
# # iter_loss=iter_loss(losses,runs_per_iteration=9)
# #%%
# #plot the pure losses
# plt.figure()
# plt.title('losses')
# plt.xlabel('run')
# plt.ylabel('loss')
# plt.plot(losses)


# #plot the starting point run, and the best achieving run
# fig,ax=plt.subplots()
# # plot_by_id(dataids_list[np.argmin(losses)],axes=ax,label='best - #' + str(dataids_list[np.argmin(losses)]) + ' - %.3f'%np.min(losses))
# ax.plot(common_voltages,data_dict['starting_point']['measurements']['0']['staircase'],label='start') 
# ax.plot(common_voltages,staircases[np.argmin(losses)],label='end')
# fig.legend()
# ax.set_yticks(np.arange(0,14,2))
# ax.grid(axis='y')

# #%%

# def plot_voltages(voltages=np.arange(9),sweep_gates=None):
#     fig,ax=plt.subplots()
#     h=ax.imshow(voltages.reshape((3,3)))
#     plt.colorbar(h,label='V')
    
#     ax.set_title('resulting voltages - sweep gates are white, ID: 7941')
#     ax.set_yticks([0.5,1.5])
#     ax.set_xticks([0.5,1.5])
#     ax.grid(color='black',linewidth=2)
    
#     #this removes the ticks but keep the gridlines
#     ax.xaxis.set_ticklabels([])
#     ax.yaxis.set_ticklabels([])
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
    
    
#     #set names
#     ax.text(0-0.5,0-0.4,'BNC12:%.3f'%voltages[0],color='r')
#     ax.text(1-0.5,0-0.4,'BNC15:%.3f'%voltages[1],color='r')
#     ax.text(2-0.5,0-0.4,'BNC19:%.3f'%voltages[2],color='r')
#     ax.text(0-0.5,1-0.4,'BNC5:%.3f'%voltages[3],color='r')
#     ax.text(1-0.5,1-0.4,'BNC18:%.3f'%voltages[4],color='r')
#     ax.text(2-0.5,1-0.4,'BNC48:%.3f'%voltages[5],color='r')
#     ax.text(0-0.5,2-0.4,'BNC3:%.3f'%voltages[6],color='r')
#     ax.text(1-0.5,2-0.4,'BNC2:%.3f'%voltages[7],color='r')
#     ax.text(2-0.5,2-0.4,'BNC50:%.3f'%voltages[8],color='r')

# # voltages_plot is a list with indexes [12,15,19,5,18,48,3,2,50]
# voltages_plot=np.empty(9)

# voltages_plot[0]=voltages[np.argmin(losses)][0]
# voltages_plot[1]=np.nan
# voltages_plot[2]=voltages[np.argmin(losses)][1]
# voltages_plot[3]=voltages[np.argmin(losses)][2]
# voltages_plot[4]=voltages[np.argmin(losses)][3]
# voltages_plot[5]=voltages[np.argmin(losses)][4]
# voltages_plot[6]=voltages[np.argmin(losses)][5]
# voltages_plot[7]=np.nan
# voltages_plot[8]=voltages[np.argmin(losses)][6]


# plot_voltages(voltages_plot)

# #%%
# import matplotlib as mpl

# iterations=[0,1,2]
# mpl.rcParams['figure.dpi']=300
# for iteration in iterations:
#     fig,ax=plt.subplots()
#     for num in np.arange(iteration*10,iteration*10+5):
#         ax.plot(common_voltages,staircases[num],label=' : %.3f'%losses[num])
#     fig.legend(bbox_to_anchor=(0.4,1))
#     ax.set_title('iteration:%i'%iteration)
#     ax.set_yticks(np.arange(0,14,2))
#     ax.grid(axis='y')
#     plt.tight_layout()
    # plt.tight_layout()
