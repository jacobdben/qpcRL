#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:30:05 2023

@author: jacob
"""

import numpy as np
from simulations.kwant_sim import KwantChip
from lossfunctions.staircasiness import *
from optimization.fourier.fourier_modes_hardcoded import fourier_to_potential 
from os import cpu_count
from time import time_ns



def initialise_device(L=200, W=150, dis=None):
    
    a = 3000//W

    pixel_size = 2*W//(3*7)
    gap_size = pixel_size // 4
    channel_halfwidth = 3*pixel_size//2+2*gap_size
    channel_halflength = channel_halfwidth-gap_size

    dL = 2*gap_size+3*pixel_size-2*channel_halflength
    dW = 4*gap_size+3*pixel_size-2*channel_halfwidth


    qpca = KwantChip(length=L, width=W, a=a, continuum=False)



    Vg = -3
    Vp = -0
    qpca.add_rect_gate([L//2-channel_halflength, L//2+channel_halflength+dL, 
                        -10*W, W//2-channel_halfwidth-dW], dist=200, V=Vg, gate_name='Vg1')
    qpca.add_rect_gate([L//2-channel_halflength, L//2+channel_halflength+dL, 
                        W//2+channel_halfwidth, 11*W], dist=200, V=Vg, gate_name='Vg2')


    pixel1 = [L//2-channel_halflength, 
              L//2-channel_halflength+pixel_size, 
              W//2+channel_halfwidth-gap_size-pixel_size, 
              W//2+channel_halfwidth-gap_size]

    pixel2 = [L//2-channel_halflength+gap_size+pixel_size, 
              L//2-channel_halflength+gap_size+2*pixel_size, 
              W//2+channel_halfwidth-gap_size-pixel_size, 
              W//2+channel_halfwidth-gap_size]

    pixel3 = [L//2-channel_halflength+2*gap_size+2*pixel_size, 
              L//2-channel_halflength+2*gap_size+3*pixel_size, 
              W//2+channel_halfwidth-gap_size-pixel_size, 
              W//2+channel_halfwidth-gap_size]

    pixel4 = [L//2-channel_halflength, 
              L//2-channel_halflength+pixel_size, 
              W//2+channel_halfwidth-2*gap_size-2*pixel_size, 
              W//2+channel_halfwidth-2*gap_size-pixel_size]

    pixel5 = [L//2-channel_halflength+gap_size+pixel_size, 
              L//2-channel_halflength+gap_size+2*pixel_size, 
              W//2+channel_halfwidth-2*gap_size-2*pixel_size, 
              W//2+channel_halfwidth-2*gap_size-pixel_size]

    pixel6 = [L//2-channel_halflength+2*gap_size+2*pixel_size, 
              L//2-channel_halflength+2*gap_size+3*pixel_size, 
              W//2+channel_halfwidth-2*gap_size-2*pixel_size, 
              W//2+channel_halfwidth-2*gap_size-pixel_size]

    pixel7 = [L//2-channel_halflength, 
              L//2-channel_halflength+pixel_size, 
              W//2+channel_halfwidth-3*gap_size-3*pixel_size, 
              W//2+channel_halfwidth-3*gap_size-2*pixel_size]

    pixel8 = [L//2-channel_halflength+gap_size+pixel_size, 
              L//2-channel_halflength+gap_size+2*pixel_size, 
              W//2+channel_halfwidth-3*gap_size-3*pixel_size, 
              W//2+channel_halfwidth-3*gap_size-2*pixel_size]

    pixel9 = [L//2-channel_halflength+2*gap_size+2*pixel_size, 
              L//2-channel_halflength+2*gap_size+3*pixel_size, 
              W//2+channel_halfwidth-3*gap_size-3*pixel_size, 
              W//2+channel_halfwidth-3*gap_size-2*pixel_size]

    qpca.add_rect_gate(pixel1, dist=200, V=Vp, gate_name='Vp1')
    qpca.add_rect_gate(pixel2, dist=200, V=Vp, gate_name='Vp2')
    qpca.add_rect_gate(pixel3, dist=200, V=Vp, gate_name='Vp3')
    qpca.add_rect_gate(pixel4, dist=200, V=Vp, gate_name='Vp4')
    qpca.add_rect_gate(pixel5, dist=200, V=Vp, gate_name='Vp5')
    qpca.add_rect_gate(pixel6, dist=200, V=Vp, gate_name='Vp6')
    qpca.add_rect_gate(pixel7, dist=200, V=Vp, gate_name='Vp7')
    qpca.add_rect_gate(pixel8, dist=200, V=Vp, gate_name='Vp8')
    qpca.add_rect_gate(pixel9, dist=200, V=Vp, gate_name='Vp9')


    if dis != None:
        Ud, ls = dis
        qpca.make_fourier_disorder(Ud, ls, random_seed=int(str(time_ns())[-9:])) # Debug: random_seed=142

    qpca.build()
        
        
    return qpca
        

def plot_pixel_disorder(qpca):
    
    L, W = qpca.L, qpca.W
    
    a = 3000//W

    pixel_size = 2*W//(3*7)
    gap_size = pixel_size // 4
    channel_halfwidth = 3*pixel_size//2+2*gap_size
    channel_halflength = channel_halfwidth-gap_size

    dL = 2*gap_size+3*pixel_size-2*channel_halflength
    dW = 4*gap_size+3*pixel_size-2*channel_halfwidth
    
    fig,ax=plt.subplots()
    h = ax.imshow(qpca.disorder[L//2-channel_halflength:L//2+channel_halflength,
                             W//2-channel_halfwidth+gap_size:W//2+channel_halfwidth-gap_size].T
               , origin='lower')
    plt.colorbar(h, label='potential [eV]', shrink=0.45)
    plt.show()
    
    
    pixel1 = qpca.disorder[L//2-channel_halflength: 
              L//2-channel_halflength+pixel_size, 
              W//2+channel_halfwidth-gap_size-pixel_size: 
              W//2+channel_halfwidth-gap_size].flatten().mean()

    pixel2 = qpca.disorder[L//2-channel_halflength+gap_size+pixel_size: 
              L//2-channel_halflength+gap_size+2*pixel_size, 
              W//2+channel_halfwidth-gap_size-pixel_size: 
              W//2+channel_halfwidth-gap_size].flatten().mean()

    pixel3 = qpca.disorder[L//2-channel_halflength+2*gap_size+2*pixel_size: 
              L//2-channel_halflength+2*gap_size+3*pixel_size, 
              W//2+channel_halfwidth-gap_size-pixel_size: 
              W//2+channel_halfwidth-gap_size].flatten().mean()

    pixel4 = qpca.disorder[L//2-channel_halflength: 
              L//2-channel_halflength+pixel_size, 
              W//2+channel_halfwidth-2*gap_size-2*pixel_size: 
              W//2+channel_halfwidth-2*gap_size-pixel_size].flatten().mean()

    pixel5 = qpca.disorder[L//2-channel_halflength+gap_size+pixel_size: 
              L//2-channel_halflength+gap_size+2*pixel_size, 
              W//2+channel_halfwidth-2*gap_size-2*pixel_size: 
              W//2+channel_halfwidth-2*gap_size-pixel_size].flatten().mean()

    pixel6 = qpca.disorder[L//2-channel_halflength+2*gap_size+2*pixel_size: 
              L//2-channel_halflength+2*gap_size+3*pixel_size, 
              W//2+channel_halfwidth-2*gap_size-2*pixel_size: 
              W//2+channel_halfwidth-2*gap_size-pixel_size].flatten().mean()

    pixel7 = qpca.disorder[L//2-channel_halflength: 
              L//2-channel_halflength+pixel_size, 
              W//2+channel_halfwidth-3*gap_size-3*pixel_size: 
              W//2+channel_halfwidth-3*gap_size-2*pixel_size].flatten().mean()

    pixel8 = qpca.disorder[L//2-channel_halflength+gap_size+pixel_size: 
              L//2-channel_halflength+gap_size+2*pixel_size, 
              W//2+channel_halfwidth-3*gap_size-3*pixel_size: 
              W//2+channel_halfwidth-3*gap_size-2*pixel_size].flatten().mean()

    pixel9 = qpca.disorder[L//2-channel_halflength+2*gap_size+2*pixel_size: 
              L//2-channel_halflength+2*gap_size+3*pixel_size, 
              W//2+channel_halfwidth-3*gap_size-3*pixel_size: 
              W//2+channel_halfwidth-3*gap_size-2*pixel_size].flatten().mean()
    
    plt.figure()
    plt.imshow([[pixel1, pixel2, pixel3],[pixel4, pixel5, pixel6],[pixel7, pixel8, pixel9]])
    plt.show()


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


def fourier_polynomials_to_voltages(polynomials,vals=None):
    """Converts polynomials of fourier-coefficients into their corresponding voltages

    Args:
        polynomials (list): list of lists of polynomials to convert, should be either 9 parameters or 8 if vals is given 
        vals (list, optional): another way to give the fourier common mode. Defaults to None.

    Raises:
        Exception: if vals is provided and the length of the polynomials and vals do not match

    Returns:
        list of lists: Voltages at each time step, outer list is over time, inner lists is over the different gates.
    """
    vals_given=False
    if isinstance(vals,list) or isinstance(vals,np.ndarray):
        vals_given=True
        if len(vals)!=polynomials.shape[0]:
            raise Exception('len(vals) and polynomials.shape[0] do not match')


    voltages=[]
    for i,modes in enumerate(polynomials):
        if vals_given:
            fourier_modes=[vals[i]] 
            fourier_modes.extend(modes)
            voltages.append(fourier_to_potential(fourier_modes)[1].ravel())
        else:
            voltages.append(fourier_to_potential(modes)[1].ravel())
    return voltages
