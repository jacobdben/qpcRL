#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:15:43 2023

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt
from kwant_sim import *
from os import cpu_count
from concurrent.futures import ProcessPoolExecutor


L = 500
W = 300
a = 10

cl = 7*7
cw = 8*7


qpca = KwantChip(length=L, width=W, a=a, continuum=False)


Vg = -3
Vp = -1.75
qpca.add_rect_gate([L//2-cl, L//2+cl, -10*W, W//2-cw], dist=200, V=Vg, gate_name='Vg1')
qpca.add_rect_gate([L//2-cl, L//2+cl, W//2+cw, 11*W], dist=200, V=Vg, gate_name='Vg2')


pixel1 = [L//2-cl, L//2-cl+4*7, W//2+cw-5*7, W//2+cw-7]
pixel2 = [L//2-cl+5*7, L//2-cl+9*7, W//2+cw-5*7, W//2+cw-7]
pixel3 = [L//2-cl+10*7, L//2-cl+14*7, W//2+cw-5*7, W//2+cw-7]
pixel4 = [L//2-cl, L//2-cl+4*7, W//2+cw-10*7, W//2+cw-6*7]
pixel5 = [L//2-cl+5*7, L//2-cl+9*7, W//2+cw-10*7, W//2+cw-6*7]
pixel6 = [L//2-cl+10*7, L//2-cl+14*7, W//2+cw-10*7, W//2+cw-6*7]
pixel7 = [L//2-cl, L//2-cl+4*7, W//2+cw-15*7, W//2+cw-11*7]
pixel8 = [L//2-cl+5*7, L//2-cl+9*7, W//2+cw-15*7, W//2+cw-11*7]
pixel9 = [L//2-cl+10*7, L//2-cl+14*7, W//2+cw-15*7, W//2+cw-11*7]

qpca.add_rect_gate(pixel1, dist=200, V=Vp, gate_name='Vp1')
qpca.add_rect_gate(pixel2, dist=200, V=Vp, gate_name='Vp2')
qpca.add_rect_gate(pixel3, dist=200, V=Vp, gate_name='Vp3')
qpca.add_rect_gate(pixel4, dist=200, V=Vp, gate_name='Vp4')
qpca.add_rect_gate(pixel5, dist=200, V=Vp, gate_name='Vp5')
qpca.add_rect_gate(pixel6, dist=200, V=Vp, gate_name='Vp6')
qpca.add_rect_gate(pixel7, dist=200, V=Vp, gate_name='Vp7')
qpca.add_rect_gate(pixel8, dist=200, V=Vp, gate_name='Vp8')
qpca.add_rect_gate(pixel9, dist=200, V=Vp, gate_name='Vp9')


#qpca.make_disorder(magnitude=.0001, length_scale=100)
qpca.build()

qpca.plot_potential()
qpca.plot_gates()

qpca.plot_potential_cross_sect()

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

#print(qpca.disorder)






den = qpca.density(0)
kwant.plotter.map(qpca.sysf, den, cmap='inferno')