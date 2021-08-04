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

pixel_gates_list=[qdac.BNC3,
                  qdac.BNC2,
                  qdac.BNC50,
                  qdac.BNC5,
                  qdac.BNC18,
                  qdac.BNC48,
                  qdac.BNC12,
                  qdac.BNC15,
                  qdac.BNC19]

bounds=(-0.09 ,0.09)
pfactor = 0.001

start = -1.1
stop = -1.22
points = 200
wait = 0.5


vals = np.linspace(start, stop, points)


stairs = staircasiness(delta=0.05, last_step=20)

dat = datahandler('BBQPC3')


def Conductance_get():
    voltage=lockin2.X()/100
    current=lockin3.X()*1e-7
    if current==0:
        return 0
    return 1/((voltage/current)/25.8125e3)

Conductance = qc.Parameter(name='g',label='Conductance',unit=r'$e^2/h$', get_cmd=Conductance_get)



def func_to_minimize(x,table): #x len 8
    x0=[0]
    x0.extend(x)
    _,voltages=fourier_to_potential(x0)
    voltages,penalty=new_point(voltages.ravel(),bounds)
    
    voltages_send=(voltages+vals[:,np.newaxis]).tolist()
    
    
    result,dataid=sweep_gates(pixel_gates_list,voltages_send,0.1,Conductance)
    
    
    val=stairs.deriv_metric_zeros1(result)+stairs.L_1_regularization(x0, 0.001)+stairs.L_2_regularization(x0, 0.001)
    
    key=table['next_key']
    table['next_key']+=1
    
    table['measurements'][key]={'val':val+penalty*pfactor,'staircase':result,'x':x0,'voltages':voltages.ravel().tolist(),'dataid':dataid}
    
    return val+penalty*pfactor




# set bias on 1 ohmic with qdac
qdac.BNC8(0.001)
# lockin2.amplitude(0.1) #100uV 
outer_gates(-1.23)

xbest,es,run_id=optimize_cma(func_to_minimize,dat,start_point=np.zeros(8),stop_time=18*3600,options={'tolx':1e-3})
  
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