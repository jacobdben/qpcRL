#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:22:33 2025

@author: jacob
"""

from scipy.optimize import minimize, OptimizeResult
from os import listdir, mkdir
from functools import partial

# general
import numpy as np
import json
import pickle
import time


################## DATA HANDLING ##################

def save_qpca(qpca, folder):
    with open(folder+'saved_qpca.pkl','wb') as file:  # Overwrites any existing file.
        pickle.dump(qpca, file, pickle.HIGHEST_PROTOCOL)

def load_qpca(folder):
    qpca = None
    with open(folder+'saved_qpca.pkl','rb') as file:
        qpca = pickle.load(file)
    return qpca
            
class OptimizeData():
    def __init__(self):
        self.data = []
    
    def add(self, iteration, coordinate, loss, nfev):
        
        if type(coordinate) is np.ndarray:
            coordinate = list(coordinate.flatten())
        
        self.data.append({'iteration': iteration,  'coordinate': coordinate, 'loss': loss, 'nfev': nfev})
    
    def save(self, folder):
        with open(folder+"datadict.txt",mode='w') as file_object:
            file_object.write(json.dumps(self.data))
    
    def load(self, folder):
        datadict = None
        with open(folder+'datadict.txt','rb') as file:
            datadict=json.load(file)
        return datadict




################## NELDER-MEAD ALGORITHM ##################

def regular_simplex(n, radius=1.0):
    """
    Generate a regular n-dimensional simplex with:
    - n+1 vertices
    - All vertices equidistant from each other
    - All vertices at distance 'radius' from the origin (fixed across dimensions)

    Returns:
        numpy.ndarray of shape (n+1, n): each row is a vertex
    """
    # Create centered matrix
    A = np.eye(n + 1) - 1 / (n + 1)
    
    # Orthonormalize the transpose
    Q, _ = np.linalg.qr(A.T)
    
    # Take the first n components (embedded in R^n)
    simplex = Q[:, :n]
    
    # Normalize to unit distance from origin
    norms = np.linalg.norm(simplex, axis=1, keepdims=True)
    unit_simplex = simplex / norms

    # Scale to fixed radius
    return unit_simplex * radius

current_iter_count = [0]

def make_tracked_objective(f):
    def wrapped(x):
        current_iter_count[0] += 1
        return f(x)
    return wrapped

def nelder_mead(func_to_minimize,function_args, starting_point, runid, sigma=0.5,options=None, savefolder = 'NM_paper_results'):
    
    
    # Location to save files
    if savefolder in listdir():
        savefolder += '/'+runid+'/'
        mkdir(savefolder)
    else:
        mkdir(savefolder)
        savefolder += '/'+runid+'/'
        mkdir(savefolder)
        
    t0 = time.time()
    n = starting_point.shape[0]
    init_vertices = regular_simplex(n, radius=sigma) + np.tile(starting_point, (n+1, 1))
    
    objective=partial(func_to_minimize,**function_args)
    
    nmdata = OptimizeData()
    nmdata.add(0, starting_point, objective(starting_point), 1)
    # Store intermediate function values
    function_values = []
    nfevals = []

    # Callback function to store function value at each step
    def callback(xk):
        fx = objective(xk)
        function_values.append(fx)
        nfevals.append(current_iter_count[0])
        print('time:', time.time()-t0, 'fx:', fx, 'nfev:', current_iter_count[0], 'xk:', xk)
        current_iter_count[0] = 0

  
    # Run optimization with Nelder-Mead and tracking
    result = minimize(
        make_tracked_objective(objective),
        starting_point,
        method='Nelder-Mead',
        callback=callback,
        options={'initial_simplex': init_vertices,'return_all': True, 'disp': True, 'maxiter':options['maxiter']}
    )
    
    for i, val in enumerate(function_values):
        nmdata.add(i+1, result['allvecs'][i], val, nfevals[i])
    nmdata.save(savefolder)
    np.save(savefolder+"fevals_NM.npy", np.array(nfevals))
    
    print("\nFinal result:")
    print("x =", result.x)
    print("fun =", result.fun)
    print("nfev =", result.nfev)
    print("success =", result.success)
    print("message =", result.message)

