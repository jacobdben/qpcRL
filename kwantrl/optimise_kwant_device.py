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
from optimization.newpoint import simple_new_point 
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
    penalty = 0
    pfac = 100

    # Convert fourier modes to corresponding voltages to be swept
    polynomials = generate_polynomials(np.linspace(0,1,len(common_mode)), X) 
    voltages = np.array(fourier_polynomials_to_voltages(polynomials,vals=common_mode))

    v_p = [simple_new_point(voltages[:,i], bounds) for i in range(9)]
    voltages = np.array([v_p[i][0] for i in range(9)]).T
    penalty = sum([v_p[i][1] for i in range(9)])
    
    Gs = [] # Define list of conductances
    
    
    # Sweep through voltages in device
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
        
        # Get conductance for this voltage config
        Gs.append(qpca.transmission())
    

    # Return loss for the conductance curve, in addition to returning the conductances themselves
    return loss_function(np.array(Gs))+pfac*penalty, Gs
    

# By default, no disorder added
disorder = None
runid = sys.argv[1]
print("runid:", runid, flush=True)

# To add disorder, call program with the disorder size as an input
if len(sys.argv)>2:
    disorder=(0.0005, int(sys.argv[2])) # Disorder size 50 is reference value.


print("CPUs available:", cpu_count(), flush=True)
print("Disorder:", disorder, flush=True)

qpca = initialise_device(L=500, W=300, dis=disorder) # Initialises our 3x3 pixel qpc device simulated in KWANT
print("Disorder size:", qpca.dis_ls)

stairs=staircasiness() # Initialize staircase loss object

common_voltages = np.linspace(-1.75, 0, 100) # Define average pixel voltage values to sweep

order=1
start_point=np.zeros(shape=(order,8)).ravel()

voltage_bounds = (-3,1) # Range of allowed voltages to use

# Arguments to be passed to CMA loop
kwargs={'common_mode':common_voltages,'qpca':qpca,'order':order,
        'loss_function':stairs.stair_loss_simple,'bounds':voltage_bounds}


cma_options={'timeout':1*24*60*60,'popsize':cpu_count(), 'maxiter':150}

# Run parallelised CMA
parallel_cma(func_to_optimize,function_args=kwargs, starting_point=start_point, runid=runid, sigma=0.5, options=cma_options, savefolder='outcmaes_paper_results')

