from typing import Callable, Sequence, Union, Tuple, List, Optional
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qcodes.dataset.measurements import Measurement
from qcodes.instrument.base import _BaseParameter
from qcodes.dataset.plotting import plot_by_id
from qcodes.dataset.data_set import load_by_id
from qcodes import config
import sys
sys.path.append("..")
from export_functions import export_by_id, export_by_id_pd, export_snapshot_by_id
import datetime



AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
AxesTupleListWithRunId = Tuple[int, List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
number = Union[float, int]

def folder_path(datasaver):
    dataid = datasaver.run_id
    start = time.time()
    stop = time.time()

    mainfolder = config.user.mainfolder
    experiment_name = datasaver._dataset.exp_name
    sample_name = datasaver._dataset.sample_name

    storage_dir = os.path.join(mainfolder, experiment_name, sample_name,str(dataid))
    os.makedirs(storage_dir, exist_ok=True)
    return storage_dir







def sweep_gates(param_set1: _BaseParameter,param_set2: _BaseParameter, opt_params, 
                vals1: number, vals2: number,
                delay: number,
                param_meas: _BaseParameter) \
                -> AxesTupleListWithRunId:
    """
    Perform a 1D scan of ``param_set1`` and 2 from ``start`` to ``stop`` in
    ``num_points`` measuring param_meas at each step. 

    Args:
        param_set1: The QCoDeS parameter to sweep over
        param_set2: The QCoDeS parameter to sweep over
        sweeping values for param_set1
        sweeping values for param_set2
        delay: Delay after setting paramter before measurement is performed
        param_meas: Parameter(s) to measure at each step 
       

    Returns:
        resulting sweep and The run_id of the DataSet created
    """

    meas = Measurement()

    meas.register_parameter(param_set1)
    meas.register_parameter(param_set2) # register the first independent parameter
    
    #register the optimization parameters, (tilt,middle gate for example)
    for key in opt_params.keys():
        meas.register_custom_parameter(name=key,
                                       label=opt_params[key]['label'],
                                       unit=opt_params[key]['unit'],
                                       paramtype='numeric')
    
    
    meas.register_parameter(param_meas,setpoints=(param_set1,param_set2,*list(opt_params.keys())))
    
    output, mname, mlabel = ([] for i in range(3))
    param_set1.post_delay = delay
    param_set2.post_delay = delay
    interrupted = False


    inst=list(meas.parameters.values())

    opt_output=[(key,opt_params[key]['value']) for key in opt_params.keys()]
    results=[]
    try:
        
        with meas.run() as datasaver:
            # os.makedirs(datapath+'{}'.format(datasaver.run_id))
            npath=folder_path(datasaver)+'/{}_set.dat'.format(inst[0].name)
            npathh=folder_path(datasaver)+'/{}_setHEADER.dat'.format(inst[0].name)
            with open(npathh, "a") as new:
                for parameter in inst:
                    mname.append(parameter.name)
                    mlabel.append(parameter.label)
                new.write('#'+"\t".join(mname)+'\n')
                new.write('#'+"\t".join(mlabel)+'\n')
                # new.write(f'#{num_points}'+'\n')
                start_time = time.perf_counter()
                for i in range(len(vals1)):
                    param_set1.set(vals1[i])
                    param_set2.set(vals2[i])
#                    time.sleep(20)
                    result=param_meas.get()
                    results.append(result)
                    datasaver.add_result((param_set1, vals1[i]),(param_set2, vals2[i]), (param_meas,result),*opt_output)
                stop_time = time.perf_counter()


    except KeyboardInterrupt:
        interrupted = True
        

    dataid = datasaver.run_id  # convenient to have for plotting

   
    

    if interrupted:
        raise KeyboardInterrupt
        
    export_by_id_pd(dataid,npath)
    export_snapshot_by_id(dataid,folder_path(datasaver)+'/snapshot.dat')
    
    # print("Acquisition took:  %s seconds " % (stop_time - start_time))

    return results,dataid