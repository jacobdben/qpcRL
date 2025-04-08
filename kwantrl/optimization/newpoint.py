import numpy as np
from scipy.optimize import minimize


def check_bound(bound,x):
    penalty=0
    if not bound[0]<=x:
        penalty+=(bound[0]-x)**2
        x=bound[0]
    if not x<=bound[1]:
        penalty+=(bound[1]-x)**2
        x=bound[1]
    return x,penalty


def simple_new_point(x,bounds):
    penalty=0
    for i in range(len(x)):
        x[i],p_temp=check_bound(bounds,x[i])
        penalty+=p_temp
        
    return x, penalty

