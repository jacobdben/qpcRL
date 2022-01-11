# from qcodes.tests.instrument_mocks import DummyInstrument

# from functools import partial
import numpy as np

# from simulations.pixel_array_sim_2 import pixelarrayQPC
# from optimization.newpoint import new_point
from optimization.cma import optimize_cma
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler

import matplotlib.pyplot as plt
import time

# from qcodes.dataset.experiment_container import new_experiment
# from qcodes.dataset.database import initialise_database
# from qcodes.dataset.measurements import Measurement

from triton7.pixel_sweep import sweep_gates
from optimization.newpoint import new_point, simple_new_point


def func_to_minimize(x, dataidsdict, vals):
    """
    Parameters
    ----------
    x : Optimization parameters, 
    foldername : data saving folder, for saving dataids of individual runs

    Returns
    -------
    loss : loss of this particular set of parameters.

    """
    
    x_projected,penalty=new_point(x,bounds=bounds,offset=-0.05)
    # x_projected, penalty = new_point(x, bounds=bounds)
    # for i,val in enumerate(x_projected):
    #     if val>0:
    #         x_projected[i]=0
    # if x_projected<=-0.2:
    #     raise
    for i,gate in enumerate(pixel_gates):
        if x_projected[i]>0:
            x_projected[i]=0
        gate(x_projected[i])

    # vals_x=np.vstack((vals1,vals2)).T
    # vals_x = (x_projected[:, np.newaxis]+vals).T
    # vals_x[np.where(vals_x > 0)] = 0
    # vals_x=np.vstack((vals,vals)).T
    vals_x=vals


    result, dataid = sweep_gates(param_sets=[outer_offset], param_set_vals=vals_x, delay=wait, param_meas=Conductance)

    loss = pfactor*penalty+stairs.deriv_metric(result)

    dataidsdict[str(dataid)] = {'x_projected':x_projected.tolist(),'loss':loss}

    return loss


bounds=(-0.1 ,0)
# bounds.append((-0.0005,0.0005))
# bounds = (-0.1, 0.1)
pfactor = 0.001

start = -0.5
stop = -0.25
points = 100
wait = 0.1
# start1=-0.56
# stop1=-0.5
# start2=-0.32
# stop2=-0.26
# points=100
# wait=0.1

vals = np.linspace(start, stop, points)
# vals1=np.linspace(start1,stop1,points)
# vals2=np.linspace(start2,stop2,points)


stairs = staircasiness(delta=0.05, last_step=20)
# QPC=pixelarrayQPC()
dat = datahandler('QCODESCHECK')



def Conductance_get():
    voltage=lockin2.X()/100
    current=lockin3.X()*1e-7
    if current==0:
        return 0
    return 1/((voltage/current)/25.8125e3)

Conductance = qc.Parameter(name='g',label='Conductance',unit=r'$e^2/h$', get_cmd=Conductance_get)

# def Outer_set(value):
#     #top outer gates
#     qdac2.BNC11(value)
#     qdac2.BNC13(value)
#     qdac2.BNC15(value)
#     qdac2.BNC17(value)
    
#     #botttom outer gates
#     qdac2.BNC3(value)
#     qdac2.BNC1(value)
#     qdac2.BNC48(value)
#     qdac2.BNC47(value)
        
# outer_gates_pixel = qc.Parameter(name='outer_gates',label='outer gate voltage',unit='V', set_cmd=Outer_set)

def outer_offset_set(value):
    #top outer gates
    qdac2.BNC11(value+0.1)
    qdac2.BNC13(value+0.1)
    qdac2.BNC15(value+0.1)
    qdac2.BNC17(value+0.1)
    
    #botttom outer gates
    qdac2.BNC3(value)
    qdac2.BNC1(value)
    qdac2.BNC48(value)
    qdac2.BNC47(value)
        
outer_offset = qc.Parameter(name='outer_offset',label='(bot) & (top+0.1) gate voltage',unit='V', set_cmd=outer_offset_set)


# def gate_pair_set(value):
#     qdac2.BNC11(value)
#     qdac2.BNC3(value)
    
# gate_pair = qc.Parameter(name='gate_pair',label='BNC11&BNC3 [V]',unit='V', set_cmd=gate_pair_set)
# outer_gate_pair=[qdac2.BNC3,
#                  qdac2.BNC11]

pixel_gates=[qdac2.BNC10,
             qdac2.BNC12,
             qdac2.BNC14,
             qdac2.BNC4,
             qdac2.BNC16,
             qdac2.BNC49,
             qdac2.BNC2,
             qdac2.BNC50,
             qdac2.BNC46]

# pixel_gates_and_bias=[qdac2.BNC10,
#                       qdac2.BNC12,
#                       qdac2.BNC14,
#                       qdac2.BNC4,
#                       qdac2.BNC16,
#                       qdac2.BNC49,
#                       qdac2.BNC2,
#                       qdac2.BNC50,
#                       qdac2.BNC46,
#                       qdac2.BNC7]


# param_sets=[bottom_gates_pixel,top_gates_pixel]
outer_offset(-0.5)


# set bias on 1 ohmic with qdac
qdac2.BNC27(0.001)
lockin2.amplitude(0.1) #100uV 
# lockin2.amplitude_true(5e-5)

xbest,es=optimize_cma(func_to_minimize,dat,maxfevals=9999,sigma=0.5,start_point=-0.05*np.ones(9),time_stop=3*3600,args=[vals])
    
# all_gates_pixel(0)

def custom_plot_by_id(dataid,ax=None):
    data=load_by_id(dataid)
    Conductance=data.get_parameter_data('g')['g']['g']
    
    loss=stairs.deriv_metric(Conductance)
    # gates=[data.get_parameter_data('g')['g'][gate.full_name] for gate in pixel_gates]
    avg=data.get_parameter_data('g')['g']['outer_offset']
    # avg=np.mean(np.array(gates),axis=0)

    if ax==None:
        fig,ax=plt.subplots()
        
    ax.plot(avg,Conductance)

    ax.set_title("run#:{} loss:{:.4f}".format(dataid,loss),fontsize=15)
    ax.set_xlabel('outer offset voltage [V]',fontsize=15)
    ax.set_ylabel(r'Conductance [$e^2/h$]',fontsize=15)
    ax.grid('on')
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

plot=False
if plot:
    for dataid in np.arange(1176,1186):
        custom_plot_by_id(dataid)
        
