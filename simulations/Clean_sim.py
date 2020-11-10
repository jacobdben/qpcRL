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

gate1dims=[5, 35, 45,-10, 12]
gate2dims=[5, 35, 45, 15, 35]

gate3dims=[5, 35, 45, 38, 60]
# gate4dims=[5, -10, 10, 37, 43]

allgatedims=[gate1dims,gate2dims,gate3dims]#,gate4dims]

_gate1 = rectangular_gate_pot(gate1dims)
_gate2 = rectangular_gate_pot(gate2dims)

_gate3 = rectangular_gate_pot(gate3dims) #
# _gate4 = rectangular_gate_pot(gate4dims)

def qpc_potential(site, V1, V2, V3):
    x, y = site.pos
    return _gate1(x, y, V1) + _gate2(x, y, V2) + _gate3(x, y, V3)# + _gate4(x,y,params.V3) 

def disorder(site, U0, salt):
    return  U0 * (uniform(repr(site), repr(salt)) - 0.5)

def hopping(site_i, site_j, phi):
    xi, yi = site_i.pos
    xj, yj = site_j.pos
    return -exp(-0.5j * phi * (xi - xj) * (yi + yj))


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
        lat = kwant.lattice.square()

        self.phi=0
        self.V1=-1
        self.V2=-1
        self.V3=-1
        self.salt=13
        self.U0=0
        self.energy=0.2
        self.t=1
        def make_lead_x(start,stop, t=1):
            syst = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
            syst[(lat(0, y) for y in np.arange(start,stop))] = 4 * t #no disorder in lead
            syst[lat.neighbors()] = hopping
            return syst
        
        def make_barrier(pot,dis, W=50, L=80, t=1):
            def onsite(s, V1, V2, V3, U0, salt, t):

                return 4 * t - pot(s,V1, V2, V3) + dis(s, U0, salt)
            # Construct the scattering region.
            sr = kwant.Builder()
            sr[(lat(x, y) for x in range(L) for y in range(W))] = onsite 
            sr[lat.neighbors()] = hopping

            
            lead = make_lead_x(start=0,stop=W, t=self.t)
            sr.attach_lead(lead)
            sr.attach_lead(lead.reversed())
            

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
            
            kwant.plotter.map(self.qpc, lambda s: qpc_potential(s, self.V1, self.V2, self.V3));

        self.fqpc = self.qpc.finalized()
    
    def transmission(self):
        Params=self.__dict__
        smatrix = kwant.smatrix(self.fqpc, self.energy, params=Params)
        return smatrix.transmission(1,0)      
    
if( __name__ == "__main__" ):
    test=TripleGateQPC(plot=False)


    def measure(start,stop,numpoints):
        plt.figure()
        result=[]

        test.phi=0
        test.U0=0
        test.V2=0
        test.energy=0.2
        sweep=np.linspace(start,stop,numpoints)

        # plt.title("phi : %.2f" %(test.phi()) + " U0 : %.2f" %(test.U0()) + " energy : %.2f" %(test.energy()) + " V2 : %.2f"%test.V2())
        plt.ylabel('transmission',fontsize=15)
        plt.xlabel('V1&V3',fontsize=15)
        for i in sweep:
            test.V3=i
            test.V1=i
            result.append(test.transmission())
        plt.plot(sweep,result)

    measure(-6,-0.5,20)
