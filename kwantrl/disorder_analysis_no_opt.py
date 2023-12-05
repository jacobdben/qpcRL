#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:36:20 2023

@author: jacob
"""


import numpy as np
import matplotlib.pyplot as plt
from os import cpu_count
from helper_functions import initialise_device, plot_pixel_disorder
from concurrent.futures import ProcessPoolExecutor


qpca = initialise_device(L=500, W=300, dis=(0.0005, 30))

qpca.plot_potential()
qpca.plot_disorder()


plot_pixel_disorder(qpca)

vqpcs = np.linspace(-1.75, 0, 100)


def solve_qpc(vqpc):
    qpca.set_gate('Vp1', vqpc)
    qpca.set_gate('Vp2', vqpc)
    qpca.set_gate('Vp3', vqpc)
    qpca.set_gate('Vp4', vqpc)
    qpca.set_gate('Vp5', vqpc)
    qpca.set_gate('Vp6', vqpc)
    qpca.set_gate('Vp7', vqpc)
    qpca.set_gate('Vp8', vqpc)
    qpca.set_gate('Vp9', vqpc)
    return qpca.transmission()

Gs = None
with ProcessPoolExecutor(cpu_count()) as executor:
    Gs = np.array(list(executor.map(solve_qpc, vqpcs)))
    
    


plt.figure()
plt.plot(vqpcs, Gs)
plt.show()
print(qpca.transmission())