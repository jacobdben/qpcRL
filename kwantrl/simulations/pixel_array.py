import kwant
from kwant.digest import uniform
import numpy as np
from math import atan2, pi, sqrt
from cmath import exp
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from .new_disorder import make_disorder, make_pixel_disorder
from types import SimpleNamespace
import scipy.sparse.linalg as sla
from multiprocessing import Pool, cpu_count

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
        return np.arctan2(u * v, d * np.sqrt(u**2 + v**2 + d**2)) / (2 * np.pi)

    def func(x, y, voltage):
        return voltage * (g(x-l, y-b) + g(x-l, t-y) +
                          g(r-x, y-b) + g(r-x, t-y))

    return func


def make_gates(distance_to_gate=5,left=30,right=50,spacing=2.5,W=80,L=80,gates_outside=0):
    pixel_size=(right-left-2*spacing)/3
    array_size=3*pixel_size+2*spacing
    
    center=(W/2,L/2)
    
    gate1dims=[distance_to_gate,left,right,-gates_outside,center[0]-array_size/2-spacing]
    gate11dims=[distance_to_gate,left,right,center[0]+array_size/2+spacing,W+gates_outside]
    
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
 
#standard size   
if False:
    W=70
    L=60
    allgatedims=make_gates(left=20,right=40,W=W,L=L,spacing=2,gates_outside=10)
else:
    #longer
    W=70
    L=120
    allgatedims=make_gates(left=int(L/2-10),right=int(L/2+10),W=W,L=L,spacing=2,gates_outside=10)

# possibly move these sections into the class
disorder_values=make_disorder(L, W, length_scale=5,random_seed=2)
pixel_disorder_values=make_pixel_disorder(L,W,allgatedims[1:10])

def disorder(site,U0):
    x,y=site.tag
    return disorder_values[x,y]*U0

def pixel_disorder(site,U0):
    x,y=site.tag
    return pixel_disorder_values[x,y]*U0
    
def disorder_old(site, U0, salt=13):
    return  U0 * (uniform(repr(site), repr(salt)) - 0.5)



class pixelarrayQPC():
    def __init__(self,W=70,L=120,plot=False,disorder_type='regular'):
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
        self.lat = kwant.lattice.square(norbs=1)

        self.phi=0
        self.salt=13
        self.U0=0
        self.energy=1
        self.t=1
        
        
        self.V1=-2
        
        self.V2=0
        self.V3=0
        self.V4=0
        self.V5=0
        self.V6=0
        self.V7=0
        self.V8=0
        self.V9=0
        self.V10=0
        
        self.V11=-2

        self.lattice=np.meshgrid(np.arange(L),np.arange(W),indexing='ij')

        self.allgatedims=make_gates(left=int(L/2-10),right=int(L/2+10),W=W,L=L,spacing=2,gates_outside=10)

        self.all_gate_calcs=[rectangular_gate_pot(dims)(self.lattice[0],self.lattice[1],1) for dims in self.allgatedims]

        if disorder_type=='pixel':
            self.disorder_func=pixel_disorder
        else:
            self.disorder_func=disorder
        self.qpc = self.make_barrier(self.qpc_potential,self.disorder_func)

        # Plotting the gates and sites/leads and potential
        if plot:
            self.plot_system()
            self.plot_potential()

        # self.precalc_leads_qpc=self.qpc.precalculate(self.energy)
        self.fqpc = self.qpc.finalized()
        self.fqpc=self.fqpc.precalculate(self.energy)

    def make_lead_x(self,start,stop, t=1):
            syst = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
            syst[(self.lat(0, y) for y in np.arange(start,stop))] = 4 * t #no disorder in lead
            syst[self.lat.neighbors()] = self.hopping
            return syst
        
    def make_barrier(self,pot,dis, W=W, L=L):
        
        # Construct the scattering region.
        sr = kwant.Builder()
        sr[(self.lat(x, y) for x in range(L) for y in range(W))] = self.onsite 
        sr[self.lat.neighbors()] = self.hopping

        
        lead = self.make_lead_x(start=0,stop=W, t=self.t)
        sr.attach_lead(lead)
        sr.attach_lead(lead.reversed())
        

        return sr
    
    def onsite(self,s, U0, t):
            return 4 * t - self.qpc_potential(s) + self.disorder_func(s, U0)

    def hopping(self,site_i, site_j):
        xi, yi = site_i.tag
        xj, yj = site_j.tag
        return -exp(-0.5j * self.phi * (xi - xj) * (yi + yj))
    
    #just gets an array with the voltages
    def get_voltages(self):
        return np.array([self.V1,self.V2,self.V3,self.V4,self.V5,self.V6,self.V7,self.V8,self.V9,self.V10,self.V11])

    #calculates the total potential from a vector with voltages multiplied with the precomputed gate potentials with V=1, then sums it together
    #this should be called in every function that wants to calculate anything
    def calc_potential(self):
        self.calculated_potential=np.einsum('i,ikj->kj',self.get_voltages(),np.array(self.all_gate_calcs))
        # return np.einsum('i,ikj->kj',self.get_voltages(),np.array(self.all_gate_calcs))

    
    def qpc_potential(self,site):
        x, y = site.tag
        return self.calculated_potential[x,y]


    def transmission(self):
        self.calc_potential()
        Params=self.__dict__
        smatrix = kwant.smatrix(self.fqpc, self.energy, params=Params)
        return smatrix.transmission(1,0)  
    
    def plot_system(self):
        fig,ax=plt.subplots()
        kwant.plot(self.qpc,ax=ax)
        rects=[]
        for gate,dims in enumerate(allgatedims):
            rect=Rectangle((dims[1],dims[3]),dims[2]-dims[1],dims[4]-dims[3],zorder=999)
            rects.append(rect)
            ax.text(x=dims[1],y=dims[3],s=str(gate))
            
            
        pc=PatchCollection(rects, facecolor='green',alpha=10)
        ax.add_collection(pc)
   
    def plot_disorder(self,bounds=((0,L),(0,W)),ax=None):
        if ax is None:
            fig,ax=plt.subplots()
        vals=np.zeros([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0]])
        i=0
        for x in np.arange(bounds[0][0],bounds[0][1]):
            j=0
            for y in np.arange(bounds[1][0],bounds[1][1]):
                site=SimpleNamespace(tag=(x,y),pos=(x,y))
                vals[i,j]=self.disorder_func(site,self.U0)
                j+=1
            i+=1
        h=ax.imshow(vals.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.colorbar(h)
        return fig,ax,vals
        
    def plot_potential(self,bounds=((0,L),(0,W)),ax=None):
        self.calc_potential()
        if ax is None:
            fig,ax=plt.subplots()
        vals=np.zeros([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0]])
        i=0
        for x in np.arange(bounds[0][0],bounds[0][1]):
            j=0
            for y in np.arange(bounds[1][0],bounds[1][1]):
                site=SimpleNamespace(tag=(x,y),pos=(x,y))
                vals[i,j]=4*self.t-self.qpc_potential(site)+self.disorder_func(site,self.U0)
                j+=1
            i+=1
        h=ax.imshow(vals.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        plt.colorbar(h)
        return fig,ax,vals.T
    
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
        elif isinstance(val,np.ndarray) or isinstance(val,list):
            self.V2=val[0]
            self.V3=val[1]
            self.V4=val[2]
            self.V5=val[3]
            self.V6=val[4]
            self.V7=val[5]
            self.V8=val[6]
            self.V9=val[7]
            self.V10=val[8]
    
    def plot_current(self, eig_num):
        self.calc_potential()
        # Calculate the wave functions in the system.
        fig,ax=plt.subplots()
        Params=self.__dict__
        ham_mat = self.fqpc.hamiltonian_submatrix(sparse=True, params=Params)
        evals, evecs = sla.eigsh(ham_mat.tocsc(), k=eig_num, sigma=0)
    
        # Calculate and plot the local current of the 10th eigenmode.
        J = kwant.operator.Current(self.fqpc)
        
        current=0
        for i in range(eig_num):
            current += J(evecs[:, i], params=Params)
        
        kwant.plotter.current(self.fqpc, current, colorbar=True,ax=ax)
        return fig,ax
        
    def wave_func(self,lead=0,ax=None,self_plot=True,plot_both=False):
        self.calc_potential()
        if ax==None:
            fig,ax=plt.subplots()
        
        Params=self.__dict__
        wfs = kwant.wave_function(self.fqpc, energy=self.energy, params=Params)
        if plot_both:
            scattering_wf1=wfs(0)
            scattering_wf2=wfs(1)
            
            total=np.sum(abs(scattering_wf1)**2, axis=0)+np.sum(abs(scattering_wf2)**2, axis=0)
            
            h=ax.imshow(total.reshape((L,W)).T,origin='lower',cmap='inferno')
            return fig,ax,h
        scattering_wf = wfs(lead)  # all scattering wave functions from lead "lead"
        if self_plot:
            h=ax.imshow(np.sum(abs(scattering_wf)**2, axis=0).reshape((L,W)).T,origin='lower',cmap='inferno')
            
        
            
        # kwant.plotter.map(self.fqpc, np.sum(abs(scattering_wf)**2, axis=0),ax=ax)
        return fig,ax,h

    def pool_func(self,vals):
        result1=[]
        for val in vals:
            self.set_all_pixels(val)
            result1.append(self.transmission())
        return result1

    def split_vals(self,vals,n):
        newlist=[]
        for i in range(0, len(vals), int(len(vals)/n)):
            newlist.append(vals[i:i + int(len(vals)/n) ])
        return newlist

    def parallel_transmission(self,vals,num_cpus=None):
        if num_cpus==None:
            num_cpus=cpu_count()
        elif num_cpus>cpu_count():
            num_cpus=cpu_count()
        
        split_vals=self.split_vals(vals,num_cpus)

        with Pool(num_cpus) as p:
            final_result=p.map(self.pool_func,split_vals)
        return [val for sublist in final_result for val in sublist] #returns the entire trace as one list
        
        
if __name__=="__main__":
    QPC=pixelarrayQPC(plot=False)
    vals=np.linspace(-2,0,100)
    result=[]
    for val in vals:
        QPC.set_all_pixels(val)
        result.append(QPC.transmission())
    print(result)