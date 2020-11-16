import kwant
from kwant.digest import uniform
import numpy as np
import time
# from math import atan2, pi, sqrt
from cmath import exp
from types import SimpleNamespace
import matplotlib.pyplot as plt
#from matplotlib.collections import PatchCollection
#from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

from QPCimport import QPC

Simfunc=QPC()

def disorder(site, params):
    return  params.U0 * (uniform(repr(site), repr(params.salt)) - 0.5)

def hopping(site_i, site_j, params):
    xi, yi = site_i.pos
    xj, yj = site_j.pos
    return -exp(-0.5j * params.phi * (xi - xj) * (yi + yj))


class PoissonQPC():
    """
    Set up KWANT basics
    Parameters are:
    phi, flux through unit cell of the latice phi=Ba^2
    V1-V3 voltage on the three gates,
    salt is a parameter controlling random nr generation in kwant.digest.uniform, used in disorder
    U0 parameter controlling the amount of disorder in the system
    energy is the fermi level?
    t is given by hbar^2/(2*m*a^2)
    """
    def __init__(self):
        
        self.pot=None
#        self.a=50e-9 #scaling, so lattice unit vector is 50nm
        self.a=1
        self.L=80 # length
        self.W=60 # width
        self.lat = kwant.lattice.square(a=self.a)
        
#        hbar=1.0545718e-34 # m**2*kg/s
#        m0=9.10938356e-31 #kg
#        mstar=0.145*m0 #effective mass https://aip.scitation.org/doi/10.1063/1.2197306
#        t=hbar**2/(2*mstar*50e-9**2)
#        t*=6.2415093433e18
        t=1
        self.params=SimpleNamespace(phi=0,V1=0,V2=0,V3=0,salt=13,U0=0,energy=0.1,t=t)
        
        def make_lead_x(start,stop, t=self.params.t):
            syst = kwant.Builder(kwant.TranslationalSymmetry([-self.a, 0]))
            syst[(self.lat(0, y) for y in np.arange(start,stop))] = 4 * t #no disorder in lead
            syst[self.lat.neighbors()] = hopping
            return syst
        

        def make_barrier(dis, W=self.W, L=self.L, t=self.params.t):
            def onsite(s, params):
                
                if s.tag[0]==0 and s.tag[1]==0:
#                    print("newrun")
                    # Calls the poisson solver function with given voltages
                    self.pot=Simfunc(xp=L,yp=W,VL=params.V1,VM=params.V2,VR=params.V3)
                    self.pot*=1.60217662e-19 #multiply by charge of e- to get energy
                    self.pot*=6.2415093433e18 #get energy in electron volts.

                    
                    
#                    plt.figure()
#                    plt.imshow(4*t-vals,origin='lower')
#                    plt.colorbar()
                    
                # unsure if we can set all of the lattice sites energy at the same time instead of this query one point at a time method, see line with sr[(self.lat(x,y) for x in range.....)]
                # which is copied from kwant docs.
#                print(s.tag)
                y,x=s.tag
#                x=int(x)
#                y=int(y)
                
                return 4 * t - self.pot[x,y] + dis(s,params)  #pot(s,XS,YS,comsol_pot_data)

            # Construct the scattering region.
            sr = kwant.Builder()
            sr[(self.lat(x, y) for x in range(L) for y in range(W))] = onsite 
            sr[self.lat.neighbors()] = hopping

            # Build and attach leads from both sides.
            
            lead = make_lead_x(start=0,stop=W, t=self.params.t)
            sr.attach_lead(lead)
            sr.attach_lead(lead.reversed())

            
            
            return sr
        
        self.qpc = make_barrier(disorder)
        
        
        self.fqpc = self.qpc.finalized()
        
    
    def transmission(self):
        Params={'params':self.params}

        smatrix = kwant.smatrix(self.fqpc, self.params.energy, params=Params)
        return smatrix.transmission(1,0)
    
    def wave_func(self):
        Params={'params':self.params}
        wfs = kwant.wave_function(self.fqpc, energy=self.params.energy, params=Params)
        # return kwant.wave_function(self.fqpc, energy=self.params.energy, params=Params)
        scattering_wf = wfs(0)  # all scattering wave functions from lead 0
        kwant.plotter.map(self.fqpc, np.sum(abs(scattering_wf)**2, axis=0));
        
    def plot_pot(self,new=False):
        if not isinstance(self.pot,np.ndarray):
            self.pot=Simfunc(xp=self.L,yp=self.W,VL=self.params.V1,VM=self.params.V2,VR=self.params.V3)
            self.pot*=1.60217662e-19 #multiply by charge of e- to get energy
            self.pot*=6.2415093433e18 #get energy in electron volts.
        if new:
            self.pot=Simfunc(xp=self.L,yp=self.W,VL=self.params.V1,VM=self.params.V2,VR=self.params.V3)
            self.pot*=1.60217662e-19 #multiply by charge of e- to get energy
            self.pot*=6.2415093433e18
            plt.figure()
            
        result=4*self.params.t-self.pot
        x=(-1000,1000)
        y=(-1500,1500)
        plt.imshow(result,origin='lower',extent=[y[0],y[1],x[0],x[1]])
        plt.plot(y,[0,0],'red',label='Horizontal Cut')
        plt.plot([0,0],x,'black',label='Vertical Cut')
        plt.legend()
        plt.xlabel('y (nm)')
        plt.ylabel('x (nm)')
        cbar=plt.colorbar()
        cbar.set_label('Voltage at 2DEG [V]')
        plt.show()
        
        ax2=plt.subplot(121)
        ax2.set_title('Horizontal Cut')
        ax2.plot(np.linspace(y[0],y[1],80),result[30,:],'red')
        ax2.set_ylabel('V')
        ax2.set_xlabel('y (nm)')
        ax3=plt.subplot(122)
        ax3.set_title('Vertical Cut')
        ax3.set_ylabel('V')
        ax3.set_xlabel('x (nm)')
        ax3.plot(np.linspace(x[0],x[1],60),result[:,40],'black')
        plt.show()
#            
        

    
    # Below is just to make it act like qcodes instruments (callable as OBJECTNAME.phi())
    # This might be using too much extra time. 
    # The individual values can also be changed through OBJECTNAME.params.phi=x
    def phi(self,val=None):
        if val !=None:
            self.params.phi=val
        else:
            print(self.params.phi)
            return(self.params.phi)
            
    def V1(self,val=None):
        if val !=None:
            self.params.V1=val
        else:
            print(self.params.V1)
            return(self.params.V1)
        
    def V2(self,val=None):
        if val !=None:
            self.params.V2=val
        else:
            print(self.params.V2)
            return(self.params.V2)
        
    def V3(self,val=None):
        if val !=None:
            self.params.V3=val
        else:
            print(self.params.V3)
            return(self.params.V3)
                    
    def salt(self,val=None):
        if val !=None:
            self.params.salt=val
        else:
            print(self.params.salt)
            return(self.params.salt)
    
    def U0(self,val=None):
        if val !=None:
            self.params.U0=val
        else:
            print(self.params.U0)
            return(self.params.U0)
            
    def energy(self,val=None):
        if val !=None:
            self.params.energy=val
        else:
            print(self.params.energy)
            return(self.params.energy)
            
    def t(self,val=None):
        if val !=None:
            self.params.t=val
        else:
            print(self.params.t)
            return(self.params.t)
            

    
    
if __name__=="__main__":
    test=comsolQPC()
    test.V1(-2.5)
    test.V3(-2.5)
    test.energy(1)
    test.V2(-0.5)
    test.plot_pot(new=True)
#    test.t(0.25)
    # t=test.transmission()
    

    start_time=time.perf_counter()
    results=[]    

#    common_mode_voltages=np.linspace(0,1e-8,10)
    common_mode_voltages=np.linspace(-3.5,-1,30)
    for v in common_mode_voltages:
        test.V1(v)
        test.V3(v)
#        test.plot_pot()
#         test.energy(v)
        results.append(test.transmission())
        
    stop_time=time.perf_counter()
    print("total time %.3f" %(stop_time-start_time))
    
    plt.figure()
    # plt.plot(np.linspace(0,0.5,25),results)
    plt.plot(common_mode_voltages,results)
    plt.xlabel('Energy',fontsize=18)
    plt.xlabel('V1&V3',fontsize=18)
#    plt.ylabel('Conductance',fontsize=18)
