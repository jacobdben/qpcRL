# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:39:53 2020

@author: Torbjørn
"""




import kwant
from kwant.digest import uniform
import numpy as np
from math import atan2, pi, sqrt
from cmath import exp
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle



hbar=1
E_z=1
results=[]

omega_y=5#ratio omega_y/omega_x increased the "well-defined-ness" of the plateaus
omega_x=1
m_star=1e-6
V_z=0

def potential(xs,ys): 
    return -0.5*m_star* omega_x**2 *(xs-40e-9)**2 \
           +0.5*m_star* omega_y**2 *(ys-25e-9)**2 +V_z

xs,ys=np.meshgrid(np.linspace(0,80e-9,80),np.linspace(0,50e-9,50))
pot=potential(xs,ys)
plt.figure()
plt.imshow(pot,origin='lower')
plt.colorbar()

def epsilon(E,m):
    return (E-hbar*omega_y*(m+0.5)-E_z)/(hbar*omega_x)

def total_transmission(E):
    T=0
    for m in range(5):
        T+=1/(1+np.exp(-2*np.pi*epsilon(E,m)))
    return T

result=[]
energies=np.linspace(-3,15,30)
for E in energies:
    result.append(total_transmission(E))
results.append(result)


plt.figure()
plt.plot(energies,result)
plt.xticks([])
# plt.yticks([])
plt.grid(True,axis='y')


def qpc_potential(site, params):
    x, y = site.pos
    return potential(x*1e-9, y*1e-9)*6.24150913e18
    
    

def disorder(site, params):
    return  params.U0 * (uniform(repr(site), repr(params.salt)) - 0.5)

def hopping(site_i, site_j, params):
    xi, yi = site_i.pos
    xj, yj = site_j.pos
    return -exp(-0.5j * params.phi * (xi - xj) * (yi + yj))


class IhnQPC():
    def __init__(self,plot=True):
        #------------------------------------------------------------------------------
        # Set up KWANT basics
        # Parameters are:
        # phi, flux through unit cell of the latice phi=Ba^2
        # V1-V3 voltage on the three gates,
        # salt is a parameter controlling random nr generation in kwant.digest.uniform, used in disorder
        # U0 parameter controlling the amount of disorder in the system
        # energy is the fermi level
        # t is a hopping parameter
        # 
        #------------------------------------------------------------------------------
        self.lat = kwant.lattice.square()
        self.params=SimpleNamespace(phi=0,V1=-1,V2=-1,V3=-1,salt=13,U0=0,energy=0.2,t=1)
        
        def make_lead_x(start,stop, t=1):
            syst = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
            syst[(self.lat(0, y) for y in np.arange(start,stop))] = 4 * t #no disorder in lead
            syst[self.lat.neighbors()] = hopping
            return syst
        
        # W=40 gives just the lower region, W=80 gives full
        def make_barrier(pot,dis, W=50, L=80, t=1):
            def onsite(s, params):
                # params=SimpleNamespace(**params)
                return 4 * t - pot(s,params) + dis(s,params)

            # Construct the scattering region.
            sr = kwant.Builder()
            sr[(self.lat(x, y) for x in range(L) for y in range(W))] = onsite 
            sr[self.lat.neighbors()] = hopping

            # Build and attach 2 leads from both sides.
            
            lead = make_lead_x(start=0,stop=W, t=1)
            sr.attach_lead(lead)
            sr.attach_lead(lead.reversed())
            
            # lead2 = make_lead_x(start=43,stop=80, t=1)
            # sr.attach_lead(lead2)
            # sr.attach_lead(lead2.reversed())

            return sr
        
        self.qpc = make_barrier(qpc_potential,disorder)
        
        # Plotting the gates and sites/leads
        if plot:
           
            # Plotting the potential from the gates
            # fig2,ax2=plt.subplots()
            kwant.plotter.map(self.qpc, lambda s: qpc_potential(s, self.params));
            # ax2.set_xlim(xlims)
            # ax2.set_ylim(ylims)
        
        self.fqpc = self.qpc.finalized()
        
    
    def transmission(self):
        Params={'params':self.params}
        # print(self.params)
        # print(Params)
        smatrix = kwant.smatrix(self.fqpc, self.params.energy, params=Params)
        return smatrix.transmission(1,0)

    
    # Below is just to make it act like qcodes instruments (callable as OBJECTNAME.phi())
    # This might be using too much extra time. 
    # The individual values can be changed through OBJECTNAME.params.phi=x
    def phi(self,val=None,verbose=False):
        if val !=None:
            self.params.phi=val
        else:
            if verbose: print(self.params.phi)
            return(self.params.phi)
            
    def V1(self,val=None,verbose=False):
        if val !=None:
            self.params.V1=val
        else:
            if verbose: print(self.params.V1)
            return(self.params.V1)
        
    def V2(self,val=None,verbose=False):
        if val !=None:
            self.params.V2=val
        else:
            if verbose: print(self.params.V2)
            return(self.params.V2)
        
    def V3(self,val=None,verbose=False):
        if val !=None:
            self.params.V3=val
        else:
            if verbose: print(self.params.V3)
            return(self.params.V3)
                    
    def salt(self,val=None,verbose=False):
        if val !=None:
            self.params.salt=val
        else:
            if verbose: print(self.params.salt)
            return(self.params.salt)
    
    def U0(self,val=None,verbose=False):
        if val !=None:
            self.params.U0=val
        else:
            if verbose: print(self.params.U0)
            return(self.params.U0)
            
    def energy(self,val=None,verbose=False):
        if val !=None:
            self.params.energy=val
        else:
            if verbose: print(self.params.energy)
            return(self.params.energy)
            
    def t(self,val=None,verbose=False):
        if val !=None:
            self.params.t=val
        else:
            if verbose: print(self.params.t)
            return(self.params.t)
            
    
if( __name__ == "__main__" ):
    test=IhnQPC(plot=True)

    # THIS is just for plotting

    def meas(start,stop,numpoints):
        plt.figure()
        result=[]

        test.phi(0)
        test.U0(0)
        test.V2(0.2)

        sweep=np.linspace(start,stop,numpoints)

        # plt.title("phi : %.2f" %(test.phi()) + " U0 : %.2f" %(test.U0()) + " energy : %.2f" %(test.energy()) + " V2 : %.2f"%test.V2())
        plt.ylabel('transmission',fontsize=15)
        # plt.xlabel('Energy',fontsize=15)
        for i in sweep:
            test.energy(i)
            result.append(test.transmission())
        plt.plot(sweep,result)
        plt.xticks([])
        plt.grid(True,axis='y')
    meas(0,0.1,30)
