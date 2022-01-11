import kwant
from kwant.digest import uniform
import numpy as np
from math import atan2, pi, sqrt
from cmath import exp
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def rectangular_gate_pot(dims):
    """Compute the potential of a rectangular gate.
    
    The gate hovers at the given distance over the plane where the
    potential is evaluated.
    
    Based on J. Appl. Phys. 77, 4504 (1995)
    http://dx.doi.org/10.1063/1.359446
    """
    distance, left, right, bottom, top = dims[0], dims[1], dims[2], dims[3], dims[4]
    d, l, r, b, t = distance, left, right, bottom, top

    def g(u, v):
        return atan2(u * v, d * sqrt(u**2 + v**2 + d**2)) / (2 * pi)

    def func(x, y, voltage):
        return voltage * (g(x-l, y-b) + g(x-l, t-y) +
                          g(r-x, y-b) + g(r-x, t-y))

    return func

# def onsite(site, params):
#     return  params.U0 * (uniform(repr(site), repr(params.salt)) - 0.5)
gate1dims=[5, 20, 30, -10, 25]
gate2dims=[5, 20, 30, 30, 50]
gate3dims=[5, 20, 30, 55, 90]
# gate4dims=[5, -10, 10, 37, 43]

allgatedims=[gate1dims,gate2dims,gate3dims]#,gate4dims]

_gate1 = rectangular_gate_pot(gate1dims)
_gate2 = rectangular_gate_pot(gate2dims)

_gate3 = rectangular_gate_pot(gate3dims) #
# _gate4 = rectangular_gate_pot(gate4dims)

def qpc_potential(site, params):
    x, y = site.pos
    return _gate1(x, y, params.V1) + _gate2(x, y, params.V2) + _gate3(x, y, params.V3)# + _gate4(x,y,params.V3) 

def disorder(site, params):
    return  params.U0 * (uniform(repr(site), repr(params.salt)) - 0.5)

def hopping(site_i, site_j, params):
    xi, yi = site_i.pos
    xj, yj = site_j.pos
    return -exp(-0.5j * params.phi * (xi - xj) * (yi + yj))


class TripleGateQPC():
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
        def make_barrier(pot,dis, W=80, L=50, t=1):
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
            fig,ax=plt.subplots()
            rects=[]
            for dims in allgatedims:
                rect=Rectangle((dims[1],dims[3]),dims[2]-dims[1],dims[4]-dims[3])
                rects.append(rect)
    
            pc=PatchCollection(rects, facecolor='green',alpha=1)
            ax.add_collection(pc)
            
            kwant.plot(self.qpc,ax=ax)
            xlims=ax.get_xlim()
            ylims=ax.get_ylim()
            
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
    test=TripleGateQPC(plot=True)

    # THIS is just for plotting

    def meas(start,stop,numpoints):
        plt.figure()
        result=[]

        test.phi(0)
        test.U0(0)
        test.V2(0.2)
        test.energy(0.2)
        sweep=np.linspace(start,stop,numpoints)

        plt.title("phi : %.2f" %(test.phi()) + " U0 : %.2f" %(test.U0()) + " energy : %.2f" %(test.energy()) + " V2 : %.2f"%test.V2())
        plt.ylabel('transmission',fontsize=15)
        plt.xlabel('V1&V3',fontsize=15)
        for i in sweep:
            test.V3(i)
            test.V1(i)
            result.append(test.transmission())
        plt.plot(sweep,result)

    meas(-8,-0.5,30)
