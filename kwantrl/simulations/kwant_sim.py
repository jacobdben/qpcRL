#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:35:32 2023

@author: jacob


Creates QPC array in kwant
"""

import kwant
import kwant.continuum

import numpy as np
from math import atan2, pi, sqrt
from cmath import exp
import scipy.linalg as la
from scipy.constants import physical_constants
from scipy.fftpack import fft2, ifft2, fftfreq

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#from shapely.plotting import plot_polygon, plot_points
from scipy.interpolate import griddata
import cProfile
import time
import warnings




# Define physical constants
hbar = physical_constants['Planck constant over 2 pi'][0] 
m_e = physical_constants['electron mass'][0]
eV = physical_constants['electron volt'][0] # electronovolt expressed in Joules








def rectangular_gate_pot(rectangle, dist, a=1):
    """Compute the potential of a rectangular gate.
    
    The gate hovers at the given distance over the plane where the
    potential is evaluated.
    
    Based on J. Appl. Phys. 77, 4504 (1995)
    http://dx.doi.org/10.1063/1.359446
    """
    left, right, bottom, top = rectangle

    def g(u, v):
        return np.arctan2(u * v * a**2, dist * np.sqrt((u*a)**2 + (v*a)**2 + dist**2)) / (2 * np.pi)

    def func(x, y):
        return (g(x-left, y-bottom) + g(x-left, top-y) 
                + g(right-x, y-bottom) + g(right-x, top-y))

    return func



        


class KwantChip():
    
    def __init__(self, length, width, a=1, t=None, 
                 m_eff=0.067, energy=None, continuum=False):
        """
        Builds an instance of a simulated chip.
        
        length (int): System length in number of sites
        width (int): System width in number of sites
        a (float): Lattice spacing in nanometers?
        t (float): Hopping parameter in eV
        m_eff (float): Effective mass in units of the electron mass
        e_diel (float): Dielectric constant
        energy (float): Fermi energy in eV
        continuum (bool): Use continuum model
        """
        
        self.L = length
        self.W = width
        self.a = a
        self.m_eff = m_eff*m_e
        
        self.t = t
        if t == None or continuum:
            self.t = hbar**2/(2*self.m_eff*(self.a*10**-9)**2) / eV
            
        

        self.energy = energy
        if energy == None:
            self.energy = self.t / 4
        self.continuum = continuum
        
        self.voltages = {}
        self.gate_potentials = {}
        self.disorder = np.zeros((self.L, self.W))
        self.dis_ls = None
        self.dis_magn = None
        
        self.gate_shapes = {"Polygons": [], "Rectangles": []}
        
        # Start with an empty tight-binding system and a single square lattice.
        # `a` is the lattice constant (by default set to 1 for simplicity).
        self.lat = kwant.lattice.square(a)
        self.sys = kwant.Builder()
        self.sysf = None
        

    def add_rect_gate(self, rectangle, dist, V=0.0, gate_name=None):
        """
        Adds a rectangular gate.
        
        rectangle (list): rectangle geometry
        dist (float): distance the gate lies above the 2DEG
        V (float): voltage on gate
        gate_name ("string"): gate label
        """
        
        
        if gate_name==None:
            gate_name = 'V' + str(len(self.voltages))
        
        potential = np.zeros((self.L, self.W))
        
        for x in range(self.L):
            for y in range(self.W):
                potential[x,y] = rectangular_gate_pot(rectangle, dist, self.a)(x,y)
                
        self.gate_potentials[gate_name]=potential
        self.voltages[gate_name] = V
        self.gate_shapes["Rectangles"].append(rectangle)
          
    
    def make_fourier_disorder(self, magnitude, length_scale, smoother='exponential', random_seed=42):
        """
        Makes the disorder potential
        
        magnitude (float): disorder magnitude
        length_scale (int): disorder length scale in number of lattice sites
        smoother (string): smoothing kernel for noise correlation
        random_seed (float): random seed
        """
        
        self.dis_ls = length_scale
        self.dis_magn = magnitude

        rng=np.random.RandomState(random_seed)
        disorder=rng.uniform(-1,1,size=(self.L,self.W))
        
        
        qx, qy = fftfreq(self.L), fftfreq(self.W)
        Qx, Qy = np.meshgrid(qx, qy, indexing='ij')
        
        ft_dis = fft2(disorder)
        
        if smoother=='exponential':
            disorder = np.real(ifft2(ft_dis*np.exp(-2*np.pi*np.sqrt(Qx**2+Qy**2)*length_scale)))
        elif smoother=='gaussian':
            disorder = np.real(ifft2(ft_dis*np.exp(-2*np.pi*(Qx**2+Qy**2)*length_scale**2)))
        elif smoother=='hard':
            disorder = np.real(ifft2(np.where(Qx**2+Qy**2 < (1/(2*np.pi*length_scale))**2, ft_dis, 0)))
        
        self.disorder = magnitude*disorder/np.max(np.abs(disorder))
        
        

    
    def get_potential(self, x, y):
        """
        Get the electrostatic potential with disorder at location (x,y)
        """
        if self.continuum:
            x, y = int(x/self.a), int(y/self.a)

        return sum([self.voltages[gate_name]*self.energy*potential[x,y] for gate_name, potential in self.gate_potentials.items()]) \
            - self.disorder[x,y]
    
    def set_gate(self, name, voltage):
        """
        Set voltage of gate for the given label
        """
        assert(name in self.voltages.keys())
        self.voltages[name] = voltage
    
    def onsite(self, site):
        return 4 * self.t - self.get_potential(site.tag[0], site.tag[1])
    
    def hopping(self, site_i, site_j):
        return -self.t
    
    def build_tb(self):
        """
        Builds tightbinding model
        """
        
        # On site effects
        self.sys[(self.lat(x, y) for x in range(self.L) for y in range(self.W))] = self.onsite
        
        # Hopping terms
        self.sys[self.lat.neighbors()] = self.hopping
        
        # Add leads
        lead = kwant.Builder(kwant.TranslationalSymmetry((-self.a, 0)))
        lead[(self.lat(0, j) for j in range(self.W))] = 4 * self.t
        lead[self.lat.neighbors()] = -self.t
        self.sys.attach_lead(lead)
        self.sys.attach_lead(lead.reversed())
        
        self.sysf = self.sys.finalized()
        self.sysf = self.sysf.precalculate(self.energy)
     
    
    def shape(self, site):      #function defining the shape of the system (rectangular shape)
        x,y = site.tag
        return (0<=x<self.L and 0<=y<self.W)  
        
    def shape_lead(self, site):  #function defining the shape of the lead
        x,y = site.tag
        return (0 <= y < self.W)

    def build_ema(self):
        """
        Builds tightbinding model from continuum model using KWANT's continuum functionality
        """
        
        if self.energy > self.t:
            print(self.t)
            warnings.warn(f"Hopping energy {self.t}, Fermi energy {self.energy}. Effective Mass Approximation may not be valid.", UserWarning)
        
        hamiltonian = """+ h**2/(2*m)*( k_x**2 + k_y**2 ) - U(x, y)"""
        
        template = kwant.continuum.discretize(hamiltonian, grid=self.a)  #carrying out the discretization of the Schrodinger equation
        self.sys.fill(template, self.shape, (0,0))     #filling the system with hopping and oniste energies from the discretization

        lead = kwant.Builder(kwant.TranslationalSymmetry([-self.a, 0]))     #building the lead
        lead.fill(template, self.shape_lead, (0, 0))    #filling the lead with hopping and oniste energies from the discretization

        self.sys.attach_lead(lead)               #attaching the left lead
        self.sys.attach_lead(lead.reversed())    #attaching the right lead
        
        self.sysf = self.sys.finalized()               #finalizing the system
        
        
        
    def build(self):
        if self.continuum:
            self.build_ema()
        else:
            self.build_tb()
    
    def transmission(self):
        """
        Calculate transmission for the system
        """
        params={k:self.__dict__[k] for k in ['energy', 'voltages']}
        params['U'] = self.get_potential
        params['h'] = physical_constants['Planck constant over 2 pi'][0]
        params['m'] = self.m_eff*physical_constants['electron volt'][0]*(10**-9)**2

        smatrix = kwant.smatrix(self.sysf, self.energy, params=params)
        return smatrix.transmission(1,0)
    
    def density(self, lead_nr):
        """
        Get state density
        """
    
        params={k:self.__dict__[k] for k in ['energy', 'voltages']}
        params['U'] = self.get_potential
        params['h'] = physical_constants['Planck constant over 2 pi'][0]
        params['m'] = self.m_eff*physical_constants['electron volt'][0]*(10**-9)**2
        wf = kwant.wave_function(self.sysf, self.energy, params=params)
        return(abs(wf(lead_nr))**2).sum(axis=0)
    
    def plot_potential(self, bounds=None):
        
        if bounds==None:
            bounds=((0,self.L),(0,self.W))
        
        pot = np.zeros((self.L, self.W))
        
        for x in range(self.L):
            for y in range(self.W):
                pot[x,y] = -sum([self.voltages[gate_name]*self.energy*potential[x,y] for gate_name, potential in self.gate_potentials.items()]) + self.disorder[x,y]
        
        
        fig,ax=plt.subplots()
        h = ax.imshow(pot.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.colorbar(h, label='potential [eV]', shrink=0.45)
        plt.show()
        
    def plot_disorder(self, bounds=None):
        
        if bounds==None:
            bounds=((0,self.L),(0,self.W))
        
        fig,ax=plt.subplots()
        h = ax.imshow(self.disorder.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.colorbar(h, label='potential [eV]', shrink=0.45)
        plt.show()
    
    def plot_potential_cross_sect(self, x=None):
        
        if x == None:
            x = self.L//2
        
        pot = np.zeros(self.W)
        
        for y in range(self.W):
            pot[y] = -sum([self.voltages[gate_name]*self.energy*potential[x,y] for gate_name, potential in self.gate_potentials.items()])
    
        
        fig,ax=plt.subplots()
        ax.plot(pot)
        ax.axhline(y=self.energy, color='k', ls='--')
        plt.show()
    
    def plot_gates(self):
        
        fig,ax=plt.subplots()
        kwant.plot(self.sys, ax=ax)
        for rect in self.gate_shapes["Rectangles"]:
            ax.fill_between(self.a*np.array([rect[0], rect[0], rect[1], rect[1]]), 
                            self.a*np.array([rect[2], rect[2], rect[2], rect[2]]),
                            self.a*np.array([rect[3], rect[3], rect[3], rect[3]]),
                            alpha=0.3, color='g')
        for pol in self.gate_shapes["Polygons"]:
            ax.fill_between(*pol.exterior.xy, alpha=0.3, color='g')
        ax.set_ylim(-1, self.W*self.a)
        plt.show()


