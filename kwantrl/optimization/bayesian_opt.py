#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 13:27:36 2025

@author: jacob
"""

from skopt import gp_minimize
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




################## BAYESIAN OPTIMIZATTION ALGORITHM ##################


current_iter_count = [0]

def make_tracked_objective(f):
    def wrapped(x):
        current_iter_count[0] += 1
        return f(x)
    return wrapped

def bayesian_opt(func_to_minimize,function_args, starting_point, runid, sigma=0.5,options=None, savefolder = 'gp_paper_results'):
    
    
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
    
    objective=partial(func_to_minimize,**function_args)
    
    gpdata = OptimizeData()

    # Store intermediate function values

    nfevals = []

    # Callback function to store function value at each step
    def callback(res):
        nfevals.append(current_iter_count[0])
        print('time:', time.time()-t0, 'fx:', res.fun, 'nfev:', current_iter_count[0], 'xk:', res.x)
        current_iter_count[0] = 0

    print('Start of optimalisation', flush=True)
    # Run optimization with Nelder-Mead and tracking
    result = gp_minimize(
        make_tracked_objective(objective),
        [(-3.0*sigma,3.0*sigma) for j in range(n)],
        n_calls=options['maxiter'],
        acq_func='EI',
        callback=callback,
        x0 = [sp for sp in starting_point],
        n_initial_points=10*n,
        noise = 1e-10,
        verbose=True
    )
    
    for i in range(result.func_vals.shape[0]):
        gpdata.add(i, result.x_iters[i], result.func_vals[i], nfevals[i])
    gpdata.save(savefolder)
    np.save(savefolder+"fevals_gp.npy", np.array(nfevals))
    
    print("\nFinal result:")
    print("x =", result.x)
    print("fun =", result.fun)

