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
sys.path.append("../..")
from triton7.export_functions import export_by_id, export_by_id_pd, export_snapshot_by_id
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







def sweep_gates(param_sets: _BaseParameter,
                param_set_vals,
                delay,
                param_meas: _BaseParameter) -> AxesTupleListWithRunId:#,
                # do_plot: bool=True) \
                
    """
    Perform a 1D scan of all parameters in param_sets, over the values listed in param_set_vals, at each step measure param_meas.
    
    Args:
        param_sets : list of parameters to be set (1 BNC for each pixel)
        param_set_vals : array of values to be set to, should be sized [numbar_of_points,len(param_sets)]
        delay: Delay after setting paramter before measurement is performed, this is currently being set to every parameter, we could do a manual wait with time.sleep instead
        param_meas: Parameter to measure at each step 
       

    Returns:
        resulting sweep and The run_id of the DataSet created
    """

    meas = Measurement()

    for param in param_sets:
        meas.register_parameter(param)
        # param.post_delay = delay #delay possible just once instead? implemented below just before result=param_meas.get()
           
    
    meas.register_parameter(param_meas,setpoints=(param_sets))
    
    output, mname, mlabel = ([] for i in range(3))
    interrupted = False


    inst=list(meas.parameters.values())

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
                # start_time = time.perf_counter()
                for i in range(len(param_set_vals)):
                    values=param_set_vals[i]
                    param_set_output=[]
                    if isinstance(values,np.float64):
                        param_sets[0].set(values)
                        param_set_output.append((param_sets[0],values))
                    else: 
                        for j in range(len(values)):
                            param_sets[j].set(values[j])
                            param_set_output.append((param_sets[j],values[j]))
                        
                    time.sleep(delay)
                    result=param_meas.get()
                    results.append(result)
                    datasaver.add_result(*param_set_output, (param_meas,result))
                # stop_time = time.perf_counter()


    except KeyboardInterrupt:
        interrupted = True
        

    dataid = datasaver.run_id  # convenient to have for plotting

   
    # if do_plot is True:
    ax, cbs = _save_image(datasaver)

    if interrupted:
        raise KeyboardInterrupt
        
    export_by_id_pd(dataid,npath)
    export_snapshot_by_id(dataid,folder_path(datasaver)+'/snapshot.dat')
    
    # print("Acquisition took:  %s seconds " % (stop_time - start_time))

    return results,dataid

def _save_image(datasaver) -> AxesTupleList:
    """
    Save the plots created by datasaver as pdf and png

    Args:
        datasaver: a measurement datasaver that contains a dataset to be saved
            as plot.

    """
    plt.ioff()
    dataid = datasaver.run_id
    # start = time.time()
    axes, cbs = plot_by_id(dataid)
    # stop = time.time()
    # print(f"plot by id took {stop-start}")

    mainfolder = config.user.mainfolder
    experiment_name = datasaver._dataset.exp_name
    sample_name = datasaver._dataset.sample_name

    storage_dir = os.path.join(mainfolder, experiment_name, sample_name)
    os.makedirs(storage_dir, exist_ok=True)

    png_dir = os.path.join(storage_dir, 'png')
    pdf_dif = os.path.join(storage_dir, 'pdf')

    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dif, exist_ok=True)

    save_pdf = True
    save_png = True

    for i, ax in enumerate(axes):
        if save_pdf:
            full_path = os.path.join(pdf_dif, f'{dataid}_{i}.pdf')
            ax.figure.savefig(full_path, dpi=500)
        if save_png:
            full_path = os.path.join(png_dir, f'{dataid}_{i}.png')
            ax.figure.savefig(full_path, dpi=500)
    plt.ion()
    return axes, cbs