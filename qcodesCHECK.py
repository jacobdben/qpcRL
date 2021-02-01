# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:31:58 2020

@author: Torbjørn
"""
from qcodes.tests.instrument_mocks import DummyInstrument

from functools import partial
import numpy as np

from simulations.pixel_array_sim_2 import pixelarrayQPC
from optimization.newpoint import new_point
from optimization.cma import optimize_cma
from lossfunctions.staircasiness import staircasiness
from datahandling.datahandling import datahandler

bounds=(-1,1)
pfactor=0.001


stairs=staircasiness(delta=0.05,last_step=20)
QPC=pixelarrayQPC()
dat=datahandler('QCODESCHECK',QPC)

def transmission_func(qpc=QPC):
    while True:
        (V1,V2,V3)=yield
        QPC.V1(V1)
        QPC.V2(V2)
        QPC.V3(V3)
        yield QPC.transmission()
    
    


instrument = DummyInstrument('gate', gates=['v1', 'v2','v3'])
instrument2 = DummyInstrument('inst', gates=['transmission'])

name=transmission_func(QPC)
next(name)     



def transGet(ins):
    val1 = name.send((instrument.v1.get(),instrument.v2.get(),instrument.v3.get()))
    next(name)
    return val1

instrument2.transmission.get=partial(transGet,instrument)



from qcodes.dataset.experiment_container import new_experiment
from qcodes.dataset.database import initialise_database
from qcodes.dataset.measurements import Measurement




def qcmeas(V2,tilt,common_voltages):
    initialise_database()
    new_experiment(name='QPCtest',
                              sample_name="no sample")
    
    meas = Measurement()
    meas.register_parameter(instrument.v1)  # 
    # meas.register_parameter(instrument.v2)
    meas.register_parameter(instrument.v3)# 
    meas.register_parameter(instrument2.transmission, setpoints=(instrument.v1,instrument.v3))  # now register the dependent oone

    meas.write_period = 2
    
    with meas.run() as datasaver:
        def func_to_minimize(x):
        # new_point ensures points are valid within bounds and constraints.
            x_projected,penalty=new_point(x,bounds=bounds)
            
            result=[]
            for avg_gates in common_voltages:
                QPC.set_all_pixels(x_projected+avg_gates)
                result.append(QPC.transmission())
                datasaver.add_result((instrument.v1, 1),
                                     (instrument.v3, 2),
                                     (instrument2.transmission, QPC.transmission()))
            return pfactor*penalty+stairs.histogram(result)
        
        xbest=optimize_cma(func_to_minimize,dat,maxfevals=10)
    
    

if __name__=="__main__":
    from qcodes.dataset.plotting import plot_by_id
    from qcodes.dataset.data_set import load_by_id
    from qcodes.dataset.data_export import get_shaped_data_by_runid
    # import sys
    # sys.path.append('../')
    # from staircasiness import staircasiness
    # stair=staircasiness(delta=0.05,last_step=5)
    qcmeas(1,2,np.linspace(-1,2,5))
    
    # common_voltages=np.linspace(-5,-1,30)
    # for i in np.linspace(0.1,0.3,5):
    #     for j in np.linspace(-0.1,0.1,3):
    #         result, dataid = qcmeas(i,j,common_voltages)
    #         plt.plot(result,label="{:.2f} : {:.2f} : {}".format(i,j,stair.histogram(result)))
    #         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
