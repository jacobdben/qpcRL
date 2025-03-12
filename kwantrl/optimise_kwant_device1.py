#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:35:11 2023

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt
from os import cpu_count
import sys
from lossfunctions.staircasiness import staircasiness
 
from optimization.cma import parallel_cma
from helper_functions import generate_polynomials, fourier_polynomials_to_voltages, initialise_device






def func_to_optimize(X, common_mode, qpca, order, bounds, loss_function):
    """
    X: An array of shape (8*n, ) containing the parameters to tune using cmaes. 
       This corresponds to 8 Fourier coefficients as n-order polynomials of the
       ninth Fourier coefficient a0 (pixel average).
    
    common_mode: array of Fourier coefficient to sweep
    
    qpca: KwantChip object
    
    order: polynomial order
    
    bounds: constraints on the pixel voltages
    
    loss_func: loss function 
    
    """
    
    
    X = X.reshape(order, -1)
    
    polynomials = generate_polynomials(np.linspace(0,1,len(common_mode)), X) 

    voltages = np.array(fourier_polynomials_to_voltages(polynomials,vals=common_mode))

    if voltages.max() > bounds[1] or voltages.min() < bounds[0]:
	# If voltage out of bounds: return maximum loss and a list of NaNs
    	return loss_function(np.zeros(voltages.shape[0])) , [np.nan for i in range(voltages.shape[0])]
    Gs = []
    
    

    for i in range(len(common_mode)):
        
        qpca.set_gate('Vp1', voltages[i, 0])
        qpca.set_gate('Vp2', voltages[i, 1])
        qpca.set_gate('Vp3', voltages[i, 2])
        qpca.set_gate('Vp4', voltages[i, 3])
        qpca.set_gate('Vp5', voltages[i, 4])
        qpca.set_gate('Vp6', voltages[i, 5])
        qpca.set_gate('Vp7', voltages[i, 6])
        qpca.set_gate('Vp8', voltages[i, 7])
        qpca.set_gate('Vp9', voltages[i, 8])
        Gs.append(qpca.transmission())
    

    
    return loss_function(np.array(Gs)), Gs
    


disorder = None

if len(sys.argv)>1:
    disorder=(0.0005, int(sys.argv[1])) # Disorder size 50 is reference value.
    #print("Disorder size:", int(sys.argv[1]))


print("CPUs available:", cpu_count(), flush=True)

qpca = initialise_device(L=500, W=300, dis=disorder)
print("Disorder size:", qpca.dis_ls)

stairs=staircasiness(cond_window=(1e-1, 9.5))

common_voltages = np.linspace(-1.75, 0, 100)
order=1
start_point=np.zeros(shape=(order,8)).ravel()

kwargs={'common_mode':common_voltages,'qpca':qpca,'order':order,
        'loss_function':stairs.multiple_windows_histogram,'bounds':(-3,1)}

sigma0 = np.hstack([1/(i+1)*np.ones(8) for i in range(order)])

cma_options={'timeout':2*24*60*60,'popsize':cpu_count(), 'maxiter':300}


parallel_cma(func_to_optimize,function_args=kwargs, starting_point=start_point, sigma=0.5, options=cma_options)

