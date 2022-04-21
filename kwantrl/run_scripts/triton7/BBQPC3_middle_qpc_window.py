import numpy as np

from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler, save_optimization_dict, load_optimization_dict
from optimization.fourier.fourier_modes_hardcoded import fourier_to_potential, plot_fourier_modes
from optimization.cma2 import optimize_cma, resume_cma
from optimization.newpoint import new_point

import matplotlib.pyplot as plt
import time


from triton7.pixel_sweep import sweep_gates
from optimization.newpoint import new_point, simple_new_point


def outer_gates_set(val):
    qdac.BNC13(val)
    qdac.BNC16(val)
    qdac.BNC17(val)
    qdac.BNC20(val)
    
    qdac.BNC6(val)
    qdac.BNC4(val)
    qdac.BNC1(val)
    qdac.BNC49(val)

outer_gates = qc.Parameter(name='outer_gates',label='outer gates pixel device',unit='v', set_cmd=outer_gates_set)

pixel_gates_list=[qdac.BNC12,
                  qdac.BNC19,
                  qdac.BNC5,
                  qdac.BNC18,
                  qdac.BNC48,
                  qdac.BNC3,
                  qdac.BNC50]

def parameter_pixels_set(val):
    qdac.BNC12(val)
    qdac.BNC19(val)
    qdac.BNC5(val)
    qdac.BNC18(val)
    qdac.BNC48(val)
    qdac.BNC3(val)
    qdac.BNC50(val)
    
parameter_pixels = qc.Parameter(name='BNC_12_19_5_18_48_3_50',label='BNC_12_19_5_18_48_3_50',unit='V', set_cmd=parameter_pixels_set)

def set_15_2(val):
    qdac.BNC15(val)
    qdac.BNC2(val)
  
gate_15_2 = qc.Parameter(name='BNC_15_and_2',label='BNC15,2',unit='V', set_cmd=set_15_2)



def Conductance_get():
    voltage=lockin2.X()/100
    current=lockin3.X()*1e-7
    if current==0:
        return 0
    return 1/((voltage/current)/25.8125e3)

Conductance = qc.Parameter(name='g',label='Conductance',unit=r'$e^2/h$', get_cmd=Conductance_get)



#%%
bounds=(-1,0.3)
pfactor = 0.001

start = -0.8
window_size = 0.8
window_bounds=(-0.4,0.79) #this gives a window of 800mv, that can be moved between 0V and -2V

points = 400
wait = 0.05



vals = np.linspace(start, start-window_size, points)


stairs = staircasiness(delta=0.05, last_step=30)

dat = datahandler('BBQPC3')


def func_to_minimize(x,table): #x len 8

    voltages,penalty=simple_new_point(x[:-1],bounds) #new_point fixes the sum to 0, simple_new_point just sets values beyond bounds to the bounds    

    window_move,penalty2=simple_new_point([x[-1]],bounds=window_bounds)
    voltages_send=vals+window_move[0]
    
    #implement going to 0 and waiting
    gate_15_2(0)
    parameter_pixels(0)
    time.sleep(20)
    # set the pixels
    for i in range(len(pixel_gates_list)):
        pixel_gates_list[i](voltages[i])
    gate_15_2(voltages_send[0])
    time.sleep(20)
    
    
    
    result,dataid=sweep_gates([gate_15_2],voltages_send,wait,Conductance)
    
    #np.flip result here because we measure towards pinch off
    val=stairs.window_loss(np.flip(result))#+stairs.L_1_regularization(voltages, 0.001)+stairs.L_2_regularization(voltages, 0.001)
    
    key=table['next_key']
    table['next_key']+=1
    
    table['measurements'][key]={'loss':val+penalty*pfactor,'staircase':result,'x':x.tolist(),'voltages':voltages.tolist(),'dataid':dataid,'deriv_metric':val,'xaxis':voltages_send.tolist()}
    
    return val+penalty*pfactor+penalty2*pfactor


#%%

# set bias on 1 ohmic with qdac
qdac.BNC8(0.000125)
lockin2.amplitude(0.06) #40uV 
outer_gates(-2)
time.sleep(10)

xbest,es,run_id=optimize_cma(func_to_minimize,dat,start_point=np.zeros(8),stop_time=20*3600,options={'tolx':1e-3})

#%%
xbest,es,run_id=resume_cma(func_to_minimize,46,dat,stop_time=14*3600+1253*60,options={'tolx':1e-3})
  
#%% plot some results
import matplotlib
matplotlib.rcParams['figure.dpi']=200

import json
def load_dataids(outcmaes_run):
    with open("F:/qcodes_data/BBQPC3/saved_data/outcmaes/{}/datadict.txt".format(outcmaes_run)) as file:
        test=json.load(file)
    return test

def iter_loss(loss,runs_per_iteration=7):
    return [np.min(loss[i*7:i*7+7]) for i in range(int(len(loss)/runs_per_iteration))]
                   
data_dict=load_dataids(46) #run id
losses=[]
dataids_list=[]
voltages=[]
staircases=[]
for key in range(len(data_dict['measurements'])):
    losses.append(data_dict['measurements'][str(key)]['loss'])
    dataids_list.append(data_dict['measurements'][str(key)]['dataid'])
    voltages.append(data_dict['measurements'][str(key)]['voltages'])
    staircases.append(data_dict['measurements'][str(key)]['staircase'])
# loss=[dataiddict[key]['loss'] for key in dataiddict.keys()]
iter_loss=iter_loss(losses,runs_per_iteration=9)
#%%
#plot the pure losses
plt.figure()
plt.title('losses')
plt.xlabel('run')
plt.ylabel('loss')
plt.plot(losses)

#plot the loss per iteration
plt.figure()
plt.title('iteration loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.plot(iter_loss)

#plot the starting point run, and the best achieving run
fig,ax=plt.subplots()
plot_by_id(dataids_list[np.argmin(losses)],axes=ax,label='best - #' + str(dataids_list[np.argmin(losses)]) + ' - %.3f'%np.min(losses))
ax.plot(vals,data_dict['starting_point']['measurements']['0']['staircase'],label='start - #'+ str(data_dict['starting_point']['measurements']['0']['dataid']) 
        + ' - %.3f'%data_dict['starting_point']['measurements']['0']['loss'])
fig.legend()
ax.set_yticks(np.arange(0,26,2))
ax.grid(axis='y')

#%%

def plot_voltages(voltages=np.arange(9),sweep_gates=None):
    fig,ax=plt.subplots()
    h=ax.imshow(voltages.reshape((3,3)))
    plt.colorbar(h,label='V')
    
    ax.set_title('resulting voltages - sweep gates are white, ID: 7941')
    ax.set_yticks([0.5,1.5])
    ax.set_xticks([0.5,1.5])
    ax.grid(color='black',linewidth=2)
    
    #this removes the ticks but keep the gridlines
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    
    #set names
    ax.text(0-0.5,0-0.4,'BNC12:%.3f'%voltages[0],color='r')
    ax.text(1-0.5,0-0.4,'BNC15:%.3f'%voltages[1],color='r')
    ax.text(2-0.5,0-0.4,'BNC19:%.3f'%voltages[2],color='r')
    ax.text(0-0.5,1-0.4,'BNC5:%.3f'%voltages[3],color='r')
    ax.text(1-0.5,1-0.4,'BNC18:%.3f'%voltages[4],color='r')
    ax.text(2-0.5,1-0.4,'BNC48:%.3f'%voltages[5],color='r')
    ax.text(0-0.5,2-0.4,'BNC3:%.3f'%voltages[6],color='r')
    ax.text(1-0.5,2-0.4,'BNC2:%.3f'%voltages[7],color='r')
    ax.text(2-0.5,2-0.4,'BNC50:%.3f'%voltages[8],color='r')

# voltages_plot is a list with indexes [12,15,19,5,18,48,3,2,50]
voltages_plot=np.empty(9)

voltages_plot[0]=voltages[np.argmin(losses)][0]
voltages_plot[1]=np.nan
voltages_plot[2]=voltages[np.argmin(losses)][1]
voltages_plot[3]=voltages[np.argmin(losses)][2]
voltages_plot[4]=voltages[np.argmin(losses)][3]
voltages_plot[5]=voltages[np.argmin(losses)][4]
voltages_plot[6]=voltages[np.argmin(losses)][5]
voltages_plot[7]=np.nan
voltages_plot[8]=voltages[np.argmin(losses)][6]


plot_voltages(voltages_plot)

#%%
iterations=[0,10,20,30]
mpl.rcParams['figure.dpi']=300
for iteration in iterations:
    fig,ax=plt.subplots()
    for num in np.arange(iteration*9,iteration*9+5):
        ax.plot(vals,staircases[num],label='#' + str(dataids_list[num]) + ' : %.3f'%losses[num])
    fig.legend(bbox_to_anchor=(0.4,1))
    ax.set_title('iteration:%i'%iteration)
    ax.set_yticks(np.arange(0,26,2))
    ax.grid(axis='y')
    plt.tight_layout()
    # plt.tight_layout()
#%%
nums=[0,2,300,500,728]
fig,ax=plt.subplots()
for j,i in enumerate(nums):
    label="{:.3f} : {:.3f} ".format(stairs.deriv_metric_cube_addsmall_zeros(np.flip(staircases[i])),stairs.deriv_metric_cube_addsmall(np.flip(staircases[i])))
    ax.plot(vals,j+np.array(staircases[i]),label=label)
plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()

#%%
def custom_plot_by_id(dataid,ax=None):
    data=load_by_id(dataid)
    Conductance=data.get_parameter_data('g')['g']['g']
    
    loss=stairs.deriv_metric_zeros1(Conductance)
    # gates=[data.get_parameter_data('g')['g'][gate.full_name] for gate in pixel_gates]
    # avg=data.get_parameter_data('g')['g']['outer_offset']
    # avg=np.mean(np.array(gates),axis=0)

    if ax==None:
        fig,ax=plt.subplots()
        
    ax.plot(vals,Conductance,label=str(dataid)+":{:.2f}".format(loss))
    # plt.savefig("F:/qcodes_data/BBQPC_2021/figures/optimization_test2.pdf",format='pdf')
    # fig.show()
    
import json
def load_dataids(outcmaes_run):
    with open("F:/qcodes_data/BBQPC2_2021/saved_data/outcmaes/{}/dataids.txt".format(outcmaes_run)) as file:
        test=json.load(file)
    return test
dataiddict=load_dataids(3)
loss=[dataiddict[key]['loss'] for key in dataiddict.keys()]



fig,ax=plt.subplots()
ax.plot(loss)
ax.set_xlabel("#Function call")
ax.set_ylabel("Loss")

fig,ax=plt.subplots()
custom_plot_by_id(int(list(dataiddict.keys())[np.argmin(loss)]),ax=ax)
custom_plot_by_id(int(list(dataiddict.keys())[np.argmax(loss)]),ax=ax)

def calc_loss(dataid,loss_function):
    data=load_by_id(dataid)
    Conductance=data.get_parameter_data('g')['g']['g']
    
    return loss_function(Conductance)

loss2=[calc_loss(int(dataid),stairs.histogram) for dataid in list(dataiddict.keys())]
loss3=[calc_loss(int(dataid),stairs.deriv_metric) for dataid in list(dataiddict.keys())]


def plot_pixel_values(outcmaes_run,dataid):
    with open("F:/qcodes_data/BBQPC2_2021/saved_data/outcmaes/{}/dataids.txt".format(outcmaes_run)) as file:
        test=json.load(file)
        vals=np.array(test[str(dataid)])
    plt.imshow(vals.reshape((3,3)))
    plt.colorbar()

fig,ax=plt.subplots()
for dataid in np.arange(114,146):
    custom_plot_by_id(dataid,ax)
        
ax.set_title("run 3",fontsize=15)
ax.set_xlabel('pixel gate voltage [V]',fontsize=15)
ax.set_ylabel(r'Conductance [$e^2/h$]',fontsize=15)
ax.legend()
ax.set_ylim(0,12)
ax.grid('on')