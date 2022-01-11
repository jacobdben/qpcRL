# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:48:32 2021

@author: Torbjørn
"""


from scipy.optimize import minimize, LinearConstraint, fmin_l_bfgs_b, Bounds
import numpy as np

def optimize_gradient(func_to_minimize,datahandler,bounds,maxiter):
    start_point=np.zeros(9)


    bounds=Bounds([bounds[0] for i in range(9)],[bounds[1] for i in range(9)])
    
    result=minimize(fun=func_to_minimize,x0=start_point,method="SLSQP", constraints=[{'type':'eq','fun':lambda x: np.sum(x)/9}],bounds=bounds,options={'eps':0.1,'maxiter':maxiter})
    return result