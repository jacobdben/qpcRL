import kwant
from kwant.digest import uniform
import numpy as np
from math import atan2, pi, sqrt
from cmath import exp
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


gate1dims=[5, 50, 70, -5, 27.5]

gate2dims=[5, 50, 55, 30, 35]
gate3dims=[5, 57.5, 62.5, 30, 35]
gate4dims=[5, 65, 70, 30, 35]

gate5dims=[5, 50, 55, 37.5, 42.5]
gate6dims=[5, 57.5, 62.5, 37.5, 42.5]
gate7dims=[5, 65, 70, 37.5, 42.5]

gate8dims=[5, 50, 55, 45, 50]
gate9dims=[5, 57.5, 62.5, 45, 50]
gate10dims=[5, 65, 70, 45, 50]

gate11dims=[5, 50, 70, 52.5, 85]






allgatedims=[gate1dims,gate2dims,gate3dims,gate4dims,gate5dims,gate6dims,gate7dims,gate8dims,gate9dims,gate10dims,gate11dims]

#gate 1 and 11 are the outer gates, 2-10 are the pixel array
_gate1 = rectangular_gate_pot(gate1dims) 

_gate2 = rectangular_gate_pot(gate2dims)
_gate3 = rectangular_gate_pot(gate3dims) 
_gate4 = rectangular_gate_pot(gate4dims)
_gate5 = rectangular_gate_pot(gate5dims)
_gate6 = rectangular_gate_pot(gate6dims) 
_gate7 = rectangular_gate_pot(gate7dims)
_gate8 = rectangular_gate_pot(gate8dims)
_gate9 = rectangular_gate_pot(gate9dims) 
_gate10 = rectangular_gate_pot(gate10dims)

_gate11 = rectangular_gate_pot(gate11dims)




def qpc_potential(site, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11):
    x, y = site.pos
    return _gate1(x, y, V1) + _gate2(x, y, V2) + _gate3(x, y, V3) + _gate4(x,y,V4) + \
           _gate5(x, y, V5) + _gate6(x, y, V6) + _gate7(x, y, V7) + _gate8(x,y,V8) + \
           _gate9(x, y, V9) + _gate10(x, y, V10) + _gate11(x, y, V11) 

def disorder(site, U0, salt):
    return  U0 * (uniform(repr(site), repr(salt)) - 0.5)

def hopping(site_i, site_j, phi):
    xi, yi = site_i.pos
    xj, yj = site_j.pos
    return -exp(-0.5j * phi * (xi - xj) * (yi + yj))


class pixelarrayQPC():
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
        lat = kwant.lattice.square()

        self.phi=0
        self.salt=13
        self.U0=0
        self.energy=0.5
        self.t=1
        
        
        self.V1=-1
        
        self.V2=0
        self.V3=0
        self.V4=0
        self.V5=0
        self.V6=0
        self.V7=0
        self.V8=0
        self.V9=0
        self.V10=0
        
        self.V11=-1


        
        def make_lead_x(start,stop, t=1):
            syst = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
            syst[(lat(0, y) for y in np.arange(start,stop))] = 4 * t #no disorder in lead
            syst[lat.neighbors()] = hopping
            return syst
        
        def make_barrier(pot,dis, W=80, L=120, t=1):
            def onsite(s, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, U0, salt, t):

                return 4 * t - pot(s,V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11) + dis(s, U0, salt)
            # Construct the scattering region.
            sr = kwant.Builder()
            sr[(lat(x, y) for x in range(L) for y in range(W))] = onsite 
            sr[lat.neighbors()] = hopping

            
            lead = make_lead_x(start=0,stop=W, t=self.t)
            sr.attach_lead(lead)
            sr.attach_lead(lead.reversed())
            

            return sr
        
        self.qpc = make_barrier(qpc_potential,disorder)
        # Plotting the gates and sites/leads and potential
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
            
            kwant.plotter.map(self.qpc, lambda s: qpc_potential(s, self.V1, self.V2, self.V3, self.V4, self.V5, self.V6, self.V7, self.V8, self.V9, self.V10, self.V11));

        self.fqpc = self.qpc.finalized()
    
    def transmission(self):
        Params=self.__dict__
        smatrix = kwant.smatrix(self.fqpc, self.energy, params=Params)
        return smatrix.transmission(1,0)  
    
    def set_all_pixels(self,val):
        self.V2=val
        self.V3=val
        self.V4=val
        self.V5=val
        self.V6=val
        self.V7=val
        self.V8=val
        self.V9=val
        self.V10=val
    

if( __name__ == "__main__" ):
    test=pixelarrayQPC(plot=True)
    test.U0=0.5
    test.set_all_pixels(0)
    import time
    def measure(start,stop,numpoints):
        plt.figure()
        result=[]
        sweep=np.linspace(start,stop,numpoints)
        for i in sweep:
            test.V11=i
            test.V1=i
            result.append(test.transmission())
        plt.plot(sweep,result)

    
    start_time=time.perf_counter()  
    measure(-8,-2,30)
    stop_time=time.perf_counter()
    print("time spent minimizing: {:.2f}".format(stop_time-start_time))