import kwant
from kwant.digest import uniform
import numpy as np
from math import atan2, pi, sqrt
from cmath import exp
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from simulations.new_disorder import make_disorder, make_pixel_disorder
from types import SimpleNamespace

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


def make_gates(distance_to_gate=5,left=30,right=50,spacing=2.5,W=80,L=80):
    pixel_size=(right-left-2*spacing)/3
    array_size=3*pixel_size+2*spacing
    
    center=(W/2,L/2)
    
    gate1dims=[distance_to_gate,left,right,0,center[0]-array_size/2-spacing]
    gate11dims=[distance_to_gate,left,right,center[0]+array_size/2+spacing,W]
    
    bottom_of_array=center[0]-array_size/2
    gates=[gate1dims]
    for i in range(3):
        for j in range(3):
            gates.append([distance_to_gate,
                           left+j*(pixel_size+spacing),
                           left+j*(pixel_size+spacing)+pixel_size,
                           bottom_of_array+i*(pixel_size+spacing),
                           bottom_of_array+i*(pixel_size+spacing)+pixel_size])
    gates.append(gate11dims)
    return gates
    
W=70
L=60
allgatedims=make_gates(left=20,right=40,W=W,L=L,spacing=2)


# allgatedims=[gate1dims,gate2dims,gate3dims,gate4dims,gate5dims,gate6dims,gate7dims,gate8dims,gate9dims,gate10dims,gate11dims]

#gate 1 and 11 are the outer gates, 2-10 are the pixel array
_gate1 = rectangular_gate_pot(allgatedims[0]) 

_gate2 = rectangular_gate_pot(allgatedims[1])
_gate3 = rectangular_gate_pot(allgatedims[2]) 
_gate4 = rectangular_gate_pot(allgatedims[3])
_gate5 = rectangular_gate_pot(allgatedims[4])
_gate6 = rectangular_gate_pot(allgatedims[5]) 
_gate7 = rectangular_gate_pot(allgatedims[6])
_gate8 = rectangular_gate_pot(allgatedims[7])
_gate9 = rectangular_gate_pot(allgatedims[8]) 
_gate10 = rectangular_gate_pot(allgatedims[9])

_gate11 = rectangular_gate_pot(allgatedims[10])




def qpc_potential(site, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11):
    x, y = site.pos
    return _gate1(x, y, V1) + _gate2(x, y, V2) + _gate3(x, y, V3) + _gate4(x,y,V4) + \
           _gate5(x, y, V5) + _gate6(x, y, V6) + _gate7(x, y, V7) + _gate8(x,y,V8) + \
           _gate9(x, y, V9) + _gate10(x, y, V10) + _gate11(x, y, V11) 

disorder_values=make_disorder(L, W, length_scale=5).T
# print(disorder_values)
pixel_disorder_values=make_pixel_disorder(L,W,allgatedims[1:10])

def disorder(site,U0):
    x,y=site.tag
    # print(x,y)
    return disorder_values[x,y]*U0

def pixel_disorder(site,U0):
    x,y=site.tag
    # print(x,y)
    return pixel_disorder_values[x,y]*U0
    
def disorder_old(site, U0, salt=13):
    return  U0 * (uniform(repr(site), repr(salt)) - 0.5)

def hopping(site_i, site_j, phi):
    xi, yi = site_i.pos
    xj, yj = site_j.pos
    return -exp(-0.5j * phi * (xi - xj) * (yi + yj))


class pixelarrayQPC():
    def __init__(self,W=W,L=L,plot=False,disorder_type='regular'):
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
        self.energy=1
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
        
        def make_barrier(pot,dis, W=W, L=L, t=1):
            def onsite(s, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, U0, salt, t):

                return 4 * t - pot(s,V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11) + dis(s, U0)
            # Construct the scattering region.
            sr = kwant.Builder()
            sr[(lat(x, y) for x in range(L) for y in range(W))] = onsite 
            sr[lat.neighbors()] = hopping

            
            lead = make_lead_x(start=0,stop=W, t=self.t)
            sr.attach_lead(lead)
            sr.attach_lead(lead.reversed())
            

            return sr
        
        if disorder_type=='pixel':
            self.disorder_func=pixel_disorder
        else:
            self.disorder_func=disorder
        self.qpc = make_barrier(qpc_potential,self.disorder_func,t=self.t)
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
            
            kwant.plotter.map(self.qpc, lambda s: 4*self.t-qpc_potential(s, self.V1, self.V2, self.V3, self.V4, self.V5, self.V6, self.V7, self.V8, self.V9, self.V10, self.V11) \
                          +self.disorder_func(s,self.U0))
        self.fqpc = self.qpc.finalized()
    
    def transmission(self):
        Params=self.__dict__
        smatrix = kwant.smatrix(self.fqpc, self.energy, params=Params)
        return smatrix.transmission(1,0)  
   
    def plot_potential(self,ax=None):
        kwant.plotter.map(self.qpc, lambda s: 4*self.t-qpc_potential(s, self.V1, self.V2, self.V3, self.V4, self.V5, self.V6, self.V7, self.V8, self.V9, self.V10, self.V11)+self.disorder_func(s,self.U0), oversampling=1,ax=ax)
    
    def plot_potential_section(self,bounds=((0,L),(0,W)),ax=None):
        if ax is None:
            fig,ax=plt.subplots()
        vals=np.zeros([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0]])
        i=0
        for x in np.arange(bounds[0][0],bounds[0][1]):
            j=0
            for y in np.arange(bounds[1][0],bounds[1][1]):
                site=SimpleNamespace(tag=(x,y),pos=(x,y))
                vals[i,j]=4*self.t-qpc_potential(site, self.V1, self.V2, self.V3, self.V4, self.V5, self.V6, self.V7, self.V8, self.V9, self.V10, self.V11)+self.disorder_func(site,self.U0)
                j+=1
            i+=1
        h=ax.imshow(vals.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        # plt.colorbar()
        return vals,h
    
    def set_all_pixels(self,val):
        if isinstance(val,float):
            self.V2=val
            self.V3=val
            self.V4=val
            self.V5=val
            self.V6=val
            self.V7=val
            self.V8=val
            self.V9=val
            self.V10=val
        elif isinstance(val,np.ndarray):
            self.V2=val[0]
            self.V3=val[1]
            self.V4=val[2]
            self.V5=val[3]
            self.V6=val[4]
            self.V7=val[5]
            self.V8=val[6]
            self.V9=val[7]
            self.V10=val[8]
    

if( __name__ == "__main__" ):
    test=pixelarrayQPC(W=W,L=L,plot=False,disorder_type='regular')
    test.U0=0.1
    test.phi=0.1
    test.energy=1
    test.V1=-1
    test.V11=-1
    test.set_all_pixels(0)
    test.plot_potential()
    
    

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
        
    def measure2(start,stop,numpoints):
        # test.V1=-10
        # test.V11=-4
        # plt.figure()
        result=[]
        sweep=np.linspace(start,stop,numpoints)
        for i in sweep:
            test.set_all_pixels(i)
            result.append(test.transmission())
        # plt.plot(sweep,result)
        return result,sweep

    
    start_time=time.perf_counter()  
    # measure(-8,-2,30)

    testresult,sweep=measure2(-2,0,30)
    # plt.plot(sweep,testresult,'*')
    plt.plot(sweep,testresult)
    plt.grid('on')
    stop_time=time.perf_counter()
    print("time spent: {:.2f}".format(stop_time-start_time))
