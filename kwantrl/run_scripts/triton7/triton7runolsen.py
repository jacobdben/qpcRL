# from qcodes.tests.instrument_mocks import DummyInstrument

# from functools import partial
import numpy as np

# from simulations.pixel_array_sim_2 import pixelarrayQPC
# from optimization.newpoint import new_point
from optimization.cma import optimize_cma
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler

import matplotlib.pyplot as plt


# from qcodes.dataset.experiment_container import new_experiment
# from qcodes.dataset.database import initialise_database
# from qcodes.dataset.measurements import Measurement

from triton7.olsen_sweep import sweep_gates


bounds=((-0.5,0.5),(-0.1,0),(-0.05,0.05)) #(tilt, middle gate)
pfactor=0.001

start=-0.2
stop=-0.1
points=150
wait=0.1

# set bias on 1 ohmic with qdac
# qdac2.BNC28(0.001)

vals=np.linspace(start,stop,points)


stairs=staircasiness(delta=0.05,last_step=20)
# QPC=pixelarrayQPC()
dat=datahandler('QCODESCHECK')

opt_params={}
opt_params['tilt']={'label': 'tilt BNC38=(1+tilt)*V, BNC43=(1-tilt)*V', 'unit':'num'}
opt_params['middle_gate']={'label' : 'middle gate voltage' , 'unit' : 'V'}
opt_params['offset']={'label' : 'BNC38 offset' , 'unit' : 'V'}

def check_bound(bound,x):
    penalty=0
    if not bound[0]<=x:
        penalty+=(bound[0]-x)**2
        x=bound[0]
    if not x<=bound[1]:
        penalty+=(bound[1]-x)**2
        x=bound[1]
    return x,penalty


def olsen_new_point(x,bounds):
    penalty=0
    for i in range(len(x)):
        x[i],p_temp=check_bound(bounds[i],x[i])
        penalty+=p_temp
        
    return x, penalty

def Conductance_get():
    voltage=lockin2.X()/100
    current=lockin3.X()*1e-7
    if current==0:
        current=np.nan
    return 1/((voltage/current)/25.8125e3)

Conductance = qc.Parameter(name='g',label='Conductance',unit=r'$e^2/h$', get_cmd=Conductance_get)

def func_to_minimize(x,foldername):
# new_point ensures points are valid within bounds and constraints.
    
    x_projected,penalty=olsen_new_point(x,bounds=bounds) #simpler new_point suited for tilt,middle gate optimization
    opt_params['tilt']['value']=x_projected[0]
    opt_params['middle_gate']['value']=x_projected[1]
    opt_params['offset']['value']=x_projected[2]
    
    qdac2.BNC40.set(x_projected[1])

    vals1=(1+x_projected[0])*vals+x_projected[2]
    vals2=(1-x_projected[0])*vals
    
    
    result,dataid=sweep_gates(qdac2.BNC38,qdac2.BNC43,opt_params,vals1,vals2,wait,Conductance) #outer gate1, outer gate2, opt params for registering custom parameters, start,stop,number,delay and measurement param
    
    loss=pfactor*penalty+stairs.deriv_metric(result)
    with open(foldername+'dataids.txt','a+') as file:
        file.write('dataid={}, loss={}, tilt={}, middle gate={}, offset={}\n'.format(dataid,loss,x_projected[0],x_projected[1],x_projected[2]))
    
    return loss





xbest=optimize_cma(func_to_minimize,dat,maxfevals=150,start_point=[0,0,0])
    
    

def custom_plot_by_id(dataid):
    data=load_by_id(dataid)
    Conductance=data.get_parameter_data('g')['g']['g']
    # gates=all_data['Ithaco']['qdac2_BNC1']
    # V43=all_data['Ithaco']['qdac2_BNC2']
    # tilt=all_data['Ithaco']['tilt'][0]
    # middle_gate=all_data['Ithaco']['middle_gate'][0]
    
    # Ithaco=all_data['Ithaco']['Ithaco']
    
    # avg=(V38+V43)/2
    
    fig,ax=plt.subplots()
    ax.plot(Conductance)
    # ax.plot(avg,Ithaco)
    # ax.set_title("#:{} tilt:{:.4f}, middle gate: {:.4f}".format(dataid,tilt,middle_gate),fontsize=15)
    # ax.set_xlabel('outer gate voltage 38&43[V]',fontsize=15)
    ax.set_ylabel('Conductance',fontsize=15)
    ax.grid('on')
    # plt.savefig("F:/qcodes_data/BBQPC_2021/figures/optimization_test2.pdf",format='pdf')
    # fig.show()

#562-576

    
def plot_trace(dataid1,dataid2):
    fig,(ax1,ax2)=plt.subplots(1,2)
    data=load_by_id(dataid1)
    current=data.get_parameter_data('Ithaco')['Ithaco']['Ithaco']
    gate38=data.get_parameter_data('Ithaco')['Ithaco']['qdac2_BNC38']
    gate43=data.get_parameter_data('Ithaco')['Ithaco']['qdac2_BNC43']
    
    current[np.where(current<=0)]=np.nan
    logcurrent=np.log(current)
    size=int(np.sqrt(len(logcurrent)))
    ax1.imshow(logcurrent.reshape((size,size)).T,extent=(np.min(gate38),np.max(gate38),np.min(gate43),np.max(gate43)),origin='lower')
    ax1.set_title('run#{}, logscale'.format(dataid1))
    ax1.set_xlabel('BNC38 [V]')
    ax1.set_ylabel('BNC43 [V]')
    
    data2=load_by_id(dataid2)
    conductance=data2.get_parameter_data('g')['g']['g']
    qdac2_BNC38=data2.get_parameter_data('g')['g']['qdac2_BNC38']
    qdac2_BNC43=data2.get_parameter_data('g')['g']['qdac2_BNC43']
    ax2.plot(conductance)
    # ax2.set_xlabel(' V43, V38=-0.185 [V]')
    ax2.set_ylabel(r'conductance [$e^2/h$]')
    ax2.set_title('run#{}'.format(dataid2))
    ax2.grid('on')
    
    ax1.plot(qdac2_BNC38,qdac2_BNC43,'r')
    plt.tight_layout()
    # plt.savefig("F:/qcodes_data/BBQPC_2021/figures/trace_{}.pdf".format(dataid2),format='pdf')
    
for dataid in np.arange(562,577):
   plot_trace(507,dataid)

# plotting=False
# if plotting:
    
#     dataid=238
#     ohm=30
#     ax=plot_by_id(dataid)[0][0]
#     ax.grid('on')
#     ax.set_xlabel('Voltage on 43&38 [V]',fontsize=15)
#     ax.set_ylabel('Current on #{} [nA]'.format(ohm),fontsize=15)
#     ax.set_title('Run#{}, Pinch off on Olsen Device, outer gates'.format(dataid),fontsize=15)
#     plt.savefig("F:/qcodes_data/BBQPC_2021/figures/pinch_off_outer.pdf",format='pdf')
    