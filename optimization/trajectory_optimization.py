import numpy as np
from .fourier.fourier_modes_hardcoded import fourier_to_potential 
from .newpoint import new_point, new_point_array
import time

def generate_polynomials(x,params):
    """Generate polynomials of arbitrary x-axis and order

    Args:
        x (np.array): array of arbitrary length
        params (np.array): params array with shape (order+1,number of polynomials), order+1 is to account for the constant term, so a 
        polynomial of order 1: a+bx, of order 2: a+bx+cx^2 etc.

    Returns:
        np.array: polynomials evaluated at x-values with shape (len(x),number of polynomials)
    """
    return np.sum([p*(x**i)[:,np.newaxis] for i,p in enumerate(params)],axis=0)


def fourier_polynomials_to_voltages(vals,polynomials):
    if len(vals)!=polynomials.shape[0]:
        raise Exception('len(vals) and polynomials.shape[0] do no match')

    #converting polynomials of the evolution of fourier modes into evolution of voltages
    #could probably be made into an array operation but requires changes in fourier_modes module
    voltages=[]
    for avg_voltage,modes in zip(vals,polynomials):
        fourier_modes=[avg_voltage] 
        fourier_modes.extend(modes)
        voltages.append(fourier_to_potential(fourier_modes)[1].ravel())
    return voltages

def trajectory_func_to_optimize(X,table,common_mode,QPC_instance,order,loss_function,bounds,pfactor):
    #remove timers
    X=X.reshape(order,-1)
    time1=time.perf_counter()
    polynomials=generate_polynomials(np.linspace(0,1,len(common_mode)),X) #previously just commom mode as first argument
    time2=time.perf_counter()
    voltages=fourier_polynomials_to_voltages(common_mode,polynomials)
    time3=time.perf_counter()

    voltages_send,penalty=new_point_array(np.array(voltages),bounds,offset=np.array(common_mode))

    # voltages_send=[]
    # penalty=0
    # for voltages_i in voltages:
    #     voltages_ref,penalty_i=new_point(voltages_i,bounds)
    #     voltages_send.append(voltages_ref.tolist())
    #     penalty+=penalty_i

    time4=time.perf_counter()
    conductance_trace=QPC_instance.parallel_transmission(voltages_send)
    time5=time.perf_counter()

    loss=loss_function(np.array(conductance_trace))

    key=table['next_key']
    table['next_key']+=1
    
    times=[time2-time1,time3-time2,time4-time3,time5-time4]
    table['measurements'][key]={'loss':loss+penalty*pfactor,'staircase':conductance_trace,'x':X.ravel().tolist(),'voltages':voltages_send.tolist(),'deriv_metric':loss,'xaxis':common_mode.tolist(),'time_for_voltages':times}
    
    return loss+penalty*pfactor




