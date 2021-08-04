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
import scipy.sparse.linalg as sla

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
        lat = kwant.lattice.square(norbs=1)
        self.W=W
        self.L=L
        # parameters
        self.phi=0
        self.salt=13
        self.U0=0
        self.energy=1
        self.t=1
        
        self.voltages=np.zeros(11)
        self.voltages[0]=-1
        self.voltages[10]=-1
        
        # self.V1=-4
        
        # self.V2=0
        # self.V3=0
        # self.V4=0
        # self.V5=0
        # self.V6=0
        # self.V7=0
        # self.V8=0
        # self.V9=0
        # self.V10=0
        
        # self.V11=-4


        allgatedims=make_gates(left=int(L/2-10),right=int(L/2+10),W=W,L=L,spacing=2)
        self.gates=[rectangular_gate_pot(allgatedims[i]) for i in range(len(allgatedims))]
        
        # _gate1 = rectangular_gate_pot(allgatedims[0]) 

        # _gate2 = rectangular_gate_pot(allgatedims[1])
        # _gate3 = rectangular_gate_pot(allgatedims[2]) 
        # _gate4 = rectangular_gate_pot(allgatedims[3])
        # _gate5 = rectangular_gate_pot(allgatedims[4])
        # _gate6 = rectangular_gate_pot(allgatedims[5]) 
        # _gate7 = rectangular_gate_pot(allgatedims[6])
        # _gate8 = rectangular_gate_pot(allgatedims[7])
        # _gate9 = rectangular_gate_pot(allgatedims[8]) 
        # _gate10 = rectangular_gate_pot(allgatedims[9])
        
        # _gate11 = rectangular_gate_pot(allgatedims[10])
        
        self.disorder_type=disorder_type
        if disorder_type=='pixel':
            pixel_disorder_values=make_pixel_disorder(L,W,allgatedims[1:10])
            def pixel_disorder(site,U0):
                x,y=site.tag
                # print(x,y)
                return pixel_disorder_values[x,y]*U0
            self.disorder_func=pixel_disorder
        else:
            disorder_values=make_disorder(L, W, length_scale=5,random_seed=2).T
            def disorder(site,U0):
                x,y=site.tag
                # print(x,y)
                return disorder_values[x,y]*U0
            self.disorder_func=disorder


        def disorder_old(site, U0, salt=13):
            return  U0 * (uniform(repr(site), repr(salt)) - 0.5)
        
        def hopping(site_i, site_j, phi):
            xi, yi = site_i.pos
            xj, yj = site_j.pos
            return -exp(-0.5j * phi * (xi - xj) * (yi + yj))
        

        def make_lead_x(start,stop, t=1):
            syst = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
            syst[(lat(0, y) for y in np.arange(start,stop))] = 4 * t #no disorder in lead
            syst[lat.neighbors()] = hopping
            return syst
        
        def make_barrier(pot,dis, W=W, L=L, t=1):
            def onsite(s, voltages, U0, salt, t):
                return 4 * t - pot(s,voltages) + dis(s, U0)
            # Construct the scattering region.
            sr = kwant.Builder()
            sr[(lat(x, y) for x in range(L) for y in range(W))] = onsite 
            sr[lat.neighbors()] = hopping

            
            lead = make_lead_x(start=0,stop=W, t=self.t)
            sr.attach_lead(lead)
            sr.attach_lead(lead.reversed())
            

            return sr
        

        self.qpc = make_barrier(self.qpc_potential,self.disorder_func,t=self.t)
        # Plotting the gates and sites/leads and potential
        if plot:
            fig,ax=plt.subplots()
            rects=[]
            for gate,dims in enumerate(allgatedims):
                rect=Rectangle((dims[1],dims[3]),dims[2]-dims[1],dims[4]-dims[3])
                rects.append(rect)
                
                ax.text(x=dims[1],y=dims[3],s=str(gate))
                
                
            pc=PatchCollection(rects, facecolor='green',alpha=1)
            ax.add_collection(pc)
            
            kwant.plot(self.qpc,ax=ax)
            xlims=ax.get_xlim()
            ylims=ax.get_ylim()
            
            #copy paste of plot potential section
            bounds=((0,L),(0,W))
            fig,ax=plt.subplots()
            vals=np.zeros([bounds[0][1]-bounds[0][0],bounds[1][1]-bounds[1][0]])
            i=0
            for x in np.arange(bounds[0][0],bounds[0][1]):
                j=0
                for y in np.arange(bounds[1][0],bounds[1][1]):
                    site=SimpleNamespace(tag=(x,y),pos=(x,y))
                    vals[i,j]=4*self.t-self.qpc_potential(site, self.voltages)+self.disorder_func(site,self.U0)
                    j+=1
                i+=1
            h=ax.imshow(vals.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
            plt.colorbar(h)
            # kwant.plotter.map(self.qpc, lambda s: 4*self.t-qpc_potential(s, self.V1, self.V2, self.V3, self.V4, self.V5, self.V6, self.V7, self.V8, self.V9, self.V10, self.V11) \
            #               +self.disorder_func(s,self.U0))
            
        self.fqpc = self.qpc.finalized()
        

    def qpc_potential(self, site, voltages):
        x, y = site.pos
        return np.sum(np.array([self.gates[i](x,y,voltages[i]) for i in range(len(voltages))]))

    def transmission(self):
        Params=self.__dict__
        smatrix = kwant.smatrix(self.fqpc, self.energy, params=Params)
        return smatrix.transmission(1,0)  
   
    def plot_disorder(self,bounds=None,ax=None):
        if bounds==None:
            bounds=((0,self.L),(0,self.W))
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
        return h
        
    def plot_potential(self,bounds=None,ax=None):
        if bounds==None:
            bounds=((0,self.L),(0,self.W))
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
            
    def V1(self,val=None):
        if val==None:
            return self.voltages[0]
        else:
            self.voltages[0]=val
    
    def V2(self,val=None):
        if val==None:
            return self.voltages[1]
        else:
            self.voltages[1]=val
    
    def plot_current(self, eig_num):
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
        
    def plot_current2(self,lead=0,ax=None):
        if ax==None:
            fig,ax=plt.subplots()
        Params=self.__dict__
        wfs = kwant.wave_function(self.fqpc, energy=self.energy, params=Params)
        scattering_wf=wfs(lead)[0]
        
        J = kwant.operator.Current(self.fqpc)
        current = J(scattering_wf,params=Params)
        kwant.plotter.current(self.fqpc, current, colorbar=True,ax=ax)
        # return fig,ax
        
    def wave_func(self,lead=0,ax=None,self_plot=True,plot_both=False):
        if ax==None:
            fig,ax=plt.subplots()
        
        Params=self.__dict__
        wfs = kwant.wave_function(self.fqpc, energy=self.energy, params=Params)
        if plot_both:
            scattering_wf1=wfs(0)
            scattering_wf2=wfs(1)
            
            total=np.sum(abs(scattering_wf1)**2, axis=0)+np.sum(abs(scattering_wf2)**2, axis=0)
            
            h=ax.imshow(total.reshape((L,W)).T,origin='lower',cmap='inferno')
            return ax,h
        scattering_wf = wfs(lead)  # all scattering wave functions from lead "lead"
        if self_plot:
            h=ax.imshow(np.sum(abs(scattering_wf)**2, axis=0).reshape((L,W)).T,origin='lower',cmap='inferno')
            
        
            
        # kwant.plotter.map(self.fqpc, np.sum(abs(scattering_wf)**2, axis=0),ax=ax)
        return ax,h
        
if( __name__ == "__main__" ):
    QPC=pixelarrayQPC(W=70,L=120,plot=True,disorder_type='regular')
    QPC.U0=0.1
    # test.phi=0.1
    QPC.energy=1
    QPC.V1=-4
    QPC.V11=-4
    QPC.set_all_pixels(0)
    # test.plot_potential()
    
    VS=np.array([0.39106099253230425,0.5987833795442608,1.0,-0.7457734376614225,-0.4452707995339354,1.0,-1.0,-0.5279270935795706,-0.27087304130163664])
    result=[]
    for V in np.linspace(-3,0,100):
        QPC.set_all_pixels(V+VS)
        result.append(QPC.transmission())
    plt.figure()
    plt.plot(result)
    plt.grid()
    # ax,h=QPC.wave_func(0,plot_both=True)
    # plt.colorbar(h)
    
    from lossfunctions.staircasiness import staircasiness
    stairs=staircasiness()
    test=stairs.gaussian_fit_single_plateau(result, 3)
    test2=stairs.simple_plateau(result, 3)
    print("gauss:%f"%test)
    print("simple:%f"%test2)