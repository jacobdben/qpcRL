import numpy as np
from .fourier.fourier_modes_hardcoded import fourier_to_potential 
from .newpoint import new_point_array
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
    #and doesnt seem to take very long time.
    voltages=[]
    for avg_voltage,modes in zip(vals,polynomials):
        fourier_modes=[avg_voltage] 
        fourier_modes.extend(modes)
        voltages.append(fourier_to_potential(fourier_modes)[1].ravel())
    return voltages

def trajectory_func_to_optimize(X,table,common_mode,QPC_instance,order,loss_function,bounds,pfactor,num_cpus):
    #first generate full array of polynomials with shape(len(common_mode),num of polynomials (with fourier pixels: 8))
    X=X.reshape(order,-1)
    polynomials=generate_polynomials(np.linspace(0,1,len(common_mode)),X) #previously just commom mode as first argument

    #convert all the fourier modes to voltages and ensure they are true within the constraints
    voltages=fourier_polynomials_to_voltages(common_mode,polynomials)
    voltages_send,penalty=new_point_array(np.array(voltages),bounds,offset=np.array(common_mode))

    #perform the measurement parallelized
    conductance_trace=QPC_instance.parallel_transmission(voltages_send,num_cpus=num_cpus)

    #evaluate 
    loss=loss_function(np.array(conductance_trace))

    key=table['next_key']
    table['next_key']+=1
    
    table['measurements'][key]={'loss':loss+penalty*pfactor,'staircase':conductance_trace,'x':X.ravel().tolist(),'voltages':voltages_send.tolist(),'deriv_metric':loss,'xaxis':common_mode.tolist()}
    
    return loss+penalty*pfactor




