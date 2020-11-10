import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from poisson import plot as p_plt
from poisson.continuous import shapes
from poisson.tools import post_process
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)
from poisson.tools.post_process import value_from_coordinates as VFC

def tmakeqpc():
    ### gates
#    gatez = 5+0.5  # lower position of the gate in z (on top of gas)
#    gateh = 1  # heigth of the gates
##    gatedepth = 6   # x extension of the gate
##    gatexpos = 3   # x position of the interior of the gate
#    gatelength = 5 #gate extension along the channel
#    gatedepth = 6 #gate extension transverse to the channel
#    middlespacing = 1
#
#    ### gas
#    gasx = 22   # gas extension along x
#    gasy = 32  # gas extension along y
#
#    ### refinement 
##    stepsize when creating the mesh i think
#    alphab = 1
#    alphag = 0.25
#    alphagas = 0.25
    
    
    gatez = 150  # lower position of the gate in z (on top of gas)
    gateh = 50  # heigth of the gates
#    gatedepth = 6   # x extension of the gate
#    gatexpos = 3   # x position of the interior of the gate
    gatelength = 600 #gate extension along the channel
    gatedepth = 1100 #gate extension transverse to the channel
    middlegatedepth = 600
#    middlespacing = 30

    ### gas
    gasx = 3000   # gas extension along x
    gasy = 4000  # gas extension along y

    ### refinement 
#    stepsize when creating the mesh i think
    alphab = 100
    alphag = 25
    alphagas = 25
    alphadopant=25
    
    ### external box
    # used as [xmin,xmax,ymin,ymax,zmin,zmax]
    extbox = np.array([-gasx/2-1,gasx/2+1,-gasy/2-1,gasy/2+1,
                       -10 * alphag, 2 * (gateh + gatez)])
#    extbox = np.array(
#            [-(2 * gatexpos + gatedepth), (2 * gatexpos + gatedepth),
#             -.75 * gasy, .75 * gasy, -10 * alphag, 2 * (gateh + gatez)])
    
    #create the external box
    extbbox = shapes.Delaunay(
            [extbox[[0, 2, 4]], extbox[[0, 2, 5]],
             extbox[[0, 3, 4]], extbox[[0, 3, 5]],
             extbox[[1, 2, 4]], extbox[[1, 2, 5]],
             extbox[[1, 3, 4]], extbox[[1, 3, 5]]])
    ext_mesh = [extbox, alphab, extbbox, 0]
    print(extbox)
    #12*alhphagas, why? could we set the gas layer height to just be 1nm? instad of 3nm like above
    gas=shapes.Rectangle([gasx,gasy,2*alphagas+1],center=[0,0,0])
    gas_mesh = [extbox, alphagas, gas, 1]
    
    dopant=shapes.Rectangle([gasx,gasy,2*alphadopant+1],center=[0,0,0.5*gatez])
    dopant_mesh = [extbox, alphadopant, dopant, 5]
    #make the left gate
    gatel=shapes.Rectangle([gatedepth,gatelength,gateh],
                           center=[extbox[0]+gatedepth/2,0,gatez])
    gatel_mesh=[extbox, alphag, gatel, 2]
    
    #make the middle gate
#    gatedepthmiddle = (extbox[1]-gatedepth)-(extbox[0]+gatedepth)-2*middlespacing
#    gatem=shapes.Rectangle([gatedepthmiddle,gatelength,gateh],
#                           center=[0,0,gatez])
    gatem=shapes.Rectangle([middlegatedepth,gatelength,gateh],
                       center=[0,0,gatez])
    gatem_mesh=[extbox, alphag, gatem, 3]
    
    #make the right gate
    gater=shapes.Rectangle([gatedepth,gatelength,gateh],
                           center=[extbox[1]-gatedepth/2,0,gatez])
    gater_mesh=[extbox, alphag, gater, 4]
    
    return (GridBuilder(meshs=[ext_mesh,gatel_mesh,gatem_mesh,gater_mesh,gas_mesh,dopant_mesh]),
            gatel,gatem,gater,gas,dopant,extbbox)

def QPC():

    grid, gatel, gatem, gater, gas, dopant, extbbox = tmakeqpc() #All geometry

    dielectric = (extbbox - (dopant + gatel + gatem + gater + gas)) #these are all functions that return true when inside their respective object.
    geo_inst = ContinuousGeometry(
            space=(gatel + gatem + gater + gas + dielectric+ dopant),
            voltage = {'gatel':gatel,
                       'gatem':gatem,
                       'gater':gater})
#                        ,dielectric=[(dielectric,1)]) #this sets the relative permivitivity of the dielectric.

    geo_inst.plot(direction=2, bbox=extbbox.get_bbox()[:4], plot_type='2D')(150)
    geo_inst.plot(points=grid.points, plot_type='3D', scale_factor=0.1)

    sys_inst = DiscretePoisson(geo_inst, grid=grid,
                             selection={'Neuman-Dirichlet':[['voltage', '*']]})

    print(len(sys_inst.points_system))
    print(len(sys_inst.mesh.points))

    def sim(x=(-1400,1400),y=(-1900,1900),xp=60,yp=80,VL=0,VM=0,VR=0,extraplot=False):
#        sys_eq_inst = LinearProblem(sys_inst,
#                                voltage_val=[(gatel, VL),(gatem, VM), (gater, VR)],
#                                charge_val=[(gas, -0.9e-4),(dopant, 1.04e-4)])#,(dielectric, 0) #charge val?
        sys_eq_inst = LinearProblem(sys_inst,
                                voltage_val=[(gatel, VL),(gatem, VM), (gater, VR)],
                                charge_val=[(gas, -1e-7),(dopant, 5e-7)])#,(dielectric, 0) #charge val?
        
        
        x=np.linspace(x[0],x[1],yp)
        y=np.linspace(y[0],y[1],xp)
        
        XS,YS=np.meshgrid(x,y)
        grid=np.vstack([XS.ravel(),YS.ravel(),np.zeros(xp*yp)]).T
        
        getvals=VFC(sys_eq_inst.points_voltage,sys_inst,deg_interpolation=1)
        
        if extraplot:
            plot_voltage, plot_charge = sys_eq_inst.plot_cut_1d(
            directions=(1, 2), bbox='default', npoints=1000)

            fig2 = plt.figure()
            ax_voltage = fig2.add_subplot(111)
        
            t, data_volt = plot_voltage(
                    (0, 0), ax = ax_voltage, marker='.', color='k',
                    label='Voltage simulation', linestyle='None')
        
            ax_voltage.set_xlabel('{0}(nm)'.format('y'))
            ax_voltage.set_ylabel('Voltage(V)')
            ax_voltage.legend(loc='upper center')
        
            ax_charge = ax_voltage.twinx()
        
            tt, data_charge = plot_charge(
                    (0, 0), ax = ax_charge, marker='.', color='b',
                    label='Charge simulation', linestyle='None')
        
            ax_charge.set_xlabel('{0}(nm)'.format('y'))
            ax_charge.set_ylabel(r'Charge density $(\#.nm^{{{-2}}})$')
            ax_charge.legend(loc='lower center')
        
            plt.show()
        
            plot_voltage_2d, plot_charge_2d = sys_eq_inst.plot_cut_2d(direction=2)
        
            plot_voltage_2d(0, colorbar_label=r'Voltage (V)')
            plot_charge_2d(0, colorbar_label=r'Charge density($\#.nm^{{{-2}}}$)')
        
            plot_voltage_2d(160, colorbar_label=r'Voltage (V)')
            plot_charge_2d(160, colorbar_label=r'Charge density($\#.nm^{{{-2}}}$)')
        
            plt.show()
        return getvals(list(grid))[1].reshape(xp,yp).T
    
        
    return sim

if __name__==('__main__'):
    import time
    sim=QPC()
    x=(-1000,1000)
    y=(-1500,1500)
    start_time=time.perf_counter()
    result=sim(x=x,y=y,VL=-2.5,VM=-0.5,VR=-2.5,extraplot=False)
    stop_time=time.perf_counter()
    print("calculation time: {:.2f}".format(stop_time-start_time))
    plt.figure()
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
    ax2.plot(np.linspace(y[0],y[1],60),result[40,:],'red')
    ax2.set_ylabel('V')
    ax2.set_xlabel('y (nm)')
    ax3=plt.subplot(122)
    ax3.set_title('Vertical Cut')
    ax3.set_ylabel('V')
    ax3.set_xlabel('x (nm)')
    ax3.plot(np.linspace(x[0],x[1],80),result[:,30],'black')
    plt.show()

"""
    from mpl_toolkits.mplot3d import Axes3D
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    
    XS,YS=np.meshgrid(np.linspace(-15,15,50),np.linspace(-10,10,80))
    surf=ax.plot_surface(XS,YS,-result)
"""