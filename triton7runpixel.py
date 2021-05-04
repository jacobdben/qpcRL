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

    # x_projected,penalty=simple_new_point(x,bounds=bounds)
    x_projected, penalty = new_point(x, bounds=bounds)
    # for i,val in enumerate(x_projected):
    #     if val>0:
    #         x_projected[i]=0
    # if x_projected<=-0.2:
    #     raise
    # for i,gate in enumerate(pixel_gates):
    #     gate(x_projected[i])

    # vals_x=np.vstack((vals1,vals2)).T
    vals_x = (x_projected[:, np.newaxis]+vals).T
    vals_x[np.where(vals_x > 0)] = 0

    # outer gate1, outer gate2, opt params for registering custom parameters, start,stop,number,delay and measurement param
    result, dataid = sweep_gates(param_sets=pixel_gates, param_set_vals=vals_x, delay=wait, param_meas=Conductance)

    loss = pfactor*penalty+stairs.deriv_metric(result)
    # with open(foldername+'dataids.txt','a+') as file:
    # file.write('dataid={}, loss={} , x= \n'.format(dataid,loss))
    # np.savetxt(file,x_projected.T)
    # file.write()
    dataidsdict[str(dataid)] = {'x_projected':x_projected.tolist(),'loss':loss}
    # print('!@#$%^!@QEWSDCXQ#RWEFSD')
    # print(dataidsdict)
    return loss


# bounds=[(-0.1,0)]*9
bounds = (-0.1, 0.1)
pfactor = 0.001

start = 0
stop = -0.15
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

def Outer_set(value):
    #top outer gates
    qdac2.BNC11(value)
    qdac2.BNC13(value)
    qdac2.BNC15(value)
    qdac2.BNC17(value)
    
    #botttom outer gates
    qdac2.BNC3(value)
    qdac2.BNC1(value)
    qdac2.BNC48(value)
    qdac2.BNC47(value)
        
outer_gates_pixel = qc.Parameter(name='outer_gates',label='outer gate voltage',unit='V', set_cmd=Outer_set)

pixel_gates=[qdac2.BNC10,
             qdac2.BNC12,
             qdac2.BNC14,
             qdac2.BNC4,
             qdac2.BNC16,
             qdac2.BNC49,
             qdac2.BNC2,
             qdac2.BNC50,
             qdac2.BNC46]


# param_sets=[bottom_gates_pixel,top_gates_pixel]
outer_gates_pixel(-0.2)
qdac2.BNC11(-0.4)
qdac2.BNC3(-0.4)
# set bias on 1 ohmic with qdac
qdac2.BNC24(0.002)
lockin2.amplitude(0.004) #0.2mV 

xbest,es=optimize_cma(func_to_minimize,dat,maxfevals=9999,sigma=0.5,start_point=np.zeros(9),time_stop=12*3600,args=[vals])
    
# all_gates_pixel(0)

def custom_plot_by_id(dataid,ax=None):
    data=load_by_id(dataid)
    Conductance=data.get_parameter_data('g')['g']['g']
    
    loss=stairs.deriv_metric(Conductance)
    gates=[data.get_parameter_data('g')['g'][gate.full_name] for gate in pixel_gates]
    avg=np.mean(np.array(gates),axis=0)

    if ax==None:
        fig,ax=plt.subplots()
        
    ax.plot(avg,Conductance)

    # ax.set_title("run#:{} loss:{:.4f}".format(dataid,loss),fontsize=15)
    ax.set_xlabel('avg pixel voltage [V]',fontsize=15)
    ax.set_ylabel(r'Conductance [$e^2/h$]',fontsize=15)
    ax.grid('on')
    # plt.savefig("F:/qcodes_data/BBQPC_2021/figures/optimization_test2.pdf",format='pdf')
    # fig.show()
    
import json
def load_dataids(outcmaes_run):
    with open("F:/qcodes_data/BBQPC_2021/saved_data/outcmaes/{}/dataids.txt".format(outcmaes_run)) as file:
        test=json.load(file)
    return test
dataiddict=load_dataids(47)
loss=[dataiddict[key]['loss'] for key in dataiddict.keys()]


custom_plot_by_id(int(list(dataiddict.keys())[np.argmin(loss)]))
def plot_pixel_values(outcmaes_run,dataid):
    with open("F:/qcodes_data/BBQPC_2021/saved_data/outcmaes/{}/dataids.txt".format(outcmaes_run)) as file:
        test=json.load(file)
        vals=np.array(test[str(dataid)])
    plt.imshow(vals.reshape((3,3)))
    plt.colorbar()

plot=False
if plot:
    for dataid in np.arange(1176,1186):
        custom_plot_by_id(dataid)
        
