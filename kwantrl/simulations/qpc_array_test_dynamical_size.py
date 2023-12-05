#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:10:56 2023

@author: jacob
"""

import numpy as np
import matplotlib.pyplot as plt
from kwant_sim import *


L = 200
W = 150
a = 20
E_F = 0.005

pixel_size = 2*W//(3*7)
gap_size = pixel_size // 4
channel_halfwidth = 3*pixel_size//2+2*gap_size
channel_halflength = channel_halfwidth-gap_size

dL = 2*gap_size+3*pixel_size-2*channel_halflength
dW = 4*gap_size+3*pixel_size-2*channel_halfwidth


qpca = KwantChip(length=L, width=W, a=a, energy=E_F, continuum=False)



Vg = -1
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


qpca.build()


qpca.plot_gates()

qpca.plot_potential_cross_sect()

Gs = []
vqpcs = np.linspace(-0.45, 0, 100)

for vqpc in vqpcs:
    
    qpca.set_gate('Vp1', vqpc)
    qpca.set_gate('Vp2', vqpc)
    qpca.set_gate('Vp3', vqpc)
    qpca.set_gate('Vp4', vqpc)
    qpca.set_gate('Vp5', vqpc)
    qpca.set_gate('Vp6', vqpc)
    qpca.set_gate('Vp7', vqpc)
    qpca.set_gate('Vp8', vqpc)
    qpca.set_gate('Vp9', vqpc)
    Gs.append(qpca.transmission())


plt.figure()
plt.plot(vqpcs, Gs)
plt.show()
print(qpca.transmission())



den = qpca.density(0)
kwant.plotter.map(qpca.sysf, den, cmap='inferno')