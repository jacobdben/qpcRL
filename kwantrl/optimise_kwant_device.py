#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:35:11 2023

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt
from os import cpu_count
from lossfunctions.staircasiness import staircasiness
 
from optimization.cma import parallel_cma
from optimization.newpoint import new_point_array
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

    voltages = fourier_polynomials_to_voltages(polynomials,vals=common_mode)
    voltages, penalty = new_point_array(np.array(voltages), bounds, 
                                          offset=np.array(common_mode))
    
    
    Gs = []
    
    

    for i in range(len(common_mode)):
        
        if not i%10:
            print(i)
        
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
    


print("CPUs available:", cpu_count())

qpca = initialise_device(L=200, W=80)


stairs=staircasiness(cond_window=(1e-1, 11))

common_voltages = np.linspace(-1.75, 0, 100)
order=2
start_point=np.zeros(shape=(order,8)).ravel()

kwargs={'common_mode':common_voltages,'qpca':qpca,'order':order,
        'loss_function':stairs.multiple_windows_histogram,'bounds':(-3,3)}

cma_options={'timeout':2,'popsize':cpu_count(), 'maxiter':10}


parallel_cma(func_to_optimize,function_args=kwargs, starting_point=start_point, options=cma_options)

