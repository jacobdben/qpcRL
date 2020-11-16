'''
    Precision of points coordinates - > 10^-13
    Precision of voronoi diagram with qhull is around 10^-13, 10^-14.
'''
import time
import os

from scipy.spatial import voronoi_plot_2d, Voronoi

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from poisson_toolbox.mesher import Mesher
from poisson_toolbox import meshes, shapes
from poisson import geometry, solver, system, reader, repeated_values
from poisson import surface_calc as sc
from poisson import plot_toolbox as p_plt

def make():
    ### define geometry
    gz = 140 # gate z position
    gs = 1000 / 2 # half of the gate spacer heigth
    gw = 300 # gate width
    gh = 120 # gate heigth
    #    kwantl = 2 * gw + gs
    kwantl = 2200
    bbox = np.array([-2000, 2000, -2000, 2000])
    Nkwant = 501
    gas_wdth = 10
    dz = 43 # dopant z position
    dw = 89 # dopant width
    dl = kwantl # dopant length
    ### mesh refinement
    # ! should match with the geometry
    alphab = 5
    alphag = 5
    alphak = np.around(kwantl / Nkwant, decimals=3)

    dz = -gz-gh
    dw = gh

    # adapt kwantl
    kwantl = (Nkwant + 1) * alphak
    print('alpha k', alphak)


    ### build geometry
    # box
    box = shapes.PointsShape(
            [bbox[[0, 2]], bbox[[0, 3]], bbox[[1, 2]], bbox[[1, 3]]])

    box_reg = [bbox, alphab, box, 0]

    # gates
    gater = shapes.PointsShape([(-gs - gw, gz), (-gs, gz),
                                (-gs - gw, gz + gh), (-gs, gz + gh)])
    gater.translate([gs*2 + gw, 0])

    gater_reg = [gater.get_bbox(), alphag, gater, 2]

    gatel = shapes.PointsShape([(-gs - gw, gz), (-gs, gz),
                                (-gs - gw, gz + gh), (-gs, gz + gh)])

    gatel_reg = [gatel.get_bbox(), alphag, gatel, 1]

    # Dopant
    dopant_box = shapes.PointsShape([(-dl , (dw + dz) * 1.1), (dl , dz * 0.5),
                                     (-dl , dz * 0.5),
                                     (dl , (dw + dz) * 1.1)])

    dopant_reg_box = [dopant_box.get_bbox(), [5, 5], dopant_box, 5]

    dopant = shapes.PointsShape([(-dl / 2 , (dw + dz) ), (dl / 2 , dz ),
                                 (-dl / 2 , dz ), (dl / 2 , (dw + dz))])

    # kwant
    kwant = shapes.PointsShape([
            (-kwantl / 2 - 1, -gas_wdth ), (kwantl / 2 + 1,  -gas_wdth ),
            (-kwantl / 2 - 1, 0 ), (kwantl / 2 + 1,  0 )])

    # Kwant_die_interface
    kwant_box = shapes.PointsShape([(-kwantl - (alphak/2), -4.5 * gas_wdth),
                            (kwantl + (alphak/2), -4.5 * gas_wdth),
                            (-kwantl - (alphak/2), 4.5 * gas_wdth),
                            (kwantl + (alphak/2), 4.5 * gas_wdth)])

    kwant_interface_reg = [kwant_box.get_bbox(), [alphak, gas_wdth],
                           kwant_box, 4]
#   TODO: Change mesher to point..
#    mesh = Mesher(meshs=[
#            box_reg, gatel_reg, gater_reg, kwant_interface_reg, dopant_reg_box])
    mesh = Mesher(meshs=[box_reg])

    func_ele = [box, gatel, gater, kwant, dopant]
    func_key = ['box', 'gatel', 'gater', 'kwant', 'dopant']
    funcs = {key:func for key, func in zip(func_key, func_ele)}

    return mesh, funcs, kwantl, gas_wdth, dw

def voltage_charge_variation(gatel, gater, discretized_obj,
                             kwantl, gas_wdth, dw, dopant,
                             elec_pot=0):

    kwantl = 2200

    gz = 140 # gate z position
    gs = 1000 / 2 # half of the gate spacer heigth
    gw = 300 # gate width
    gh = 120 # gate heigth
    dl = kwantl # dopant length

    gate_capa = shapes.PointsShape([(-dl / 2 , (gw + gz) ), (dl / 2 , gz ),
                                    (-dl / 2 , gz ), (dl / 2 , (gw + gz))])

#    dirichlet_val = [(gatel, elec_pot),
#                     (gater, elec_pot)]

    dirichlet_val = [(gate_capa, elec_pot)]

    rho_0 = 2.11e15
    rho_0 = rho_0 * (discretized_obj.distance_normalization**2)
    print(discretized_obj.points_system_mixed)

    val_charge = ((((4 * rho_0) / kwantl**2)
                   * discretized_obj.mesh.points[
                           discretized_obj.points_system_mixed][:, 0]**2)
                  - np.ones(len(discretized_obj.points_system_mixed)) * rho_0)
    charge = val_charge * 0
    charge = charge / (gas_wdth)


    mixed_val = [[discretized_obj.points_system_mixed, charge]]
    charge_val = [(dopant, rho_0/dw)]

    return dirichlet_val, charge_val, mixed_val


def print_times(times, discretized_obj):
    '''
    '''

    print('Number of points: {0}'.format(
            discretized_obj.mesh.npoints))
    print('Poisson obj build: {0}'.format(
            times[1] - times[0]))
    print('Discretize_obj build:{0}'.format(
            times[2] - times[1]))
    print('Build system of equations obj: {0}'.format(times[4] - times[3]))
    print('Solve using mumps obj: {0}'.format(times[5] - times[4]))
    print('Update system of equations obj: {0}'.format(times[6] - times[5]))
    print('Solve using mumps without factorization obj: {0}'.format(
            times[7] - times[6]))

def example_2DEG():

    kwantl = 2200

    gz = 140 # gate z position
    gs = 1000 / 2 # half of the gate spacer heigth
    gw = 300 # gate width
    gh = 120 # gate heigth
    dl = kwantl # dopant length

    gate_capa = shapes.PointsShape([(-dl / 2 , (gw + gz) ), (dl / 2 , gz ),
                                    (-dl / 2 , gz ), (dl / 2 , (gw + gz))])

####        TODO: Finish removing mutable attributes

    # Make the grid of points
    mesh, funcs, kwantl, gas_wdth, dw = make()
    func_keys = ['box', 'gatel', 'gater', 'kwant', 'dopant']
    box, gatel, gater, kwant, dopant = [funcs[key] for key in func_keys]
    points = mesh.mesh_points


####
    plot_geometry = p_plt.sliced_geometry_2d(
            figsize=(10, 10), bbox=np.array([-2000, 2000, -2000, 2000]),
            npt=[100, 1400],
            fctnames=['gatel', 'gater', 'kwant_reg', 'dopants'],
            functions=[gatel, gater, kwant, dopant],
            xlabel='x(nm)', ylabel='y(nm)', direction=2, vmin=-2, vmax=2,
            cmap=ListedColormap(['w', 'orange', 'blue', 'red', 'green']),
            interpolation='none', aspect='equal')

#    plot_geometry(0)

#### Define the poisson problem (continuous version)
    #
    times = [time.time()]
    poisson_obj = geometry.Geometry(space=box,
                                 voltage=[gate_capa], #voltage=[gatel, gater],
                                 charge=[dopant],
                                 mixed=[kwant],
                                 default_relative_permittivity=12)
    times.append(time.time())
#### Discretize the continous poisson problem
    discretized_obj = system.System(poisson_obj, meshes.Voronoi(grid=points))

    times.append(time.time())
#### Solving the discretized poisson problem

    # Defining the voltage variation and charge variation.
    voltage_val, charge_val, mixed_val = voltage_charge_variation(
            gatel, gater, discretized_obj,
            kwantl, gas_wdth, dw, dopant)

    # Creating the system of equations obj
    times.append(time.time())
    system_eq_obj = solver.SysEquations(
            discretized_obj, voltage_val=voltage_val, charge_val=charge_val,
            mixed_val=mixed_val, is_charge_density=True, build_equations=True,
            solve_system=False)

    times.append(time.time())
    # Solving the system of equations
    points_charge, points_voltage = system_eq_obj.solve()

    # Save file in vtk format
    system_eq_obj.save_to_vtk(
            filename='/'.join((os.path.dirname(os.path.abspath(__file__)),
                               '2DEG')),
            encoding='ascii')

    reader.regions_summary_charge(discretized_obj, points_charge,
                                  verbose=True)

    times.append(time.time())
    # Updating the potential at the electrode
    voltage_val_upd, t, tt = voltage_charge_variation(
            gatel, gater, discretized_obj,
            kwantl, gas_wdth, dw, dopant, elec_pot=-1)

    # Updating the system of equations
    times.append(time.time())
    system_eq_obj.update(voltage_val=voltage_val_upd)

    # Solving the system of equations without making a lu_factorization
    times.append(time.time())
    points_charge_upd, points_voltage_upd = system_eq_obj.solve(
            factorize=False)

    times.append(time.time())
    print_times(times, discretized_obj)

#    func_vol = p_plt.sliced_values_1d(
#            figsize=(10,10), xlabel='x(nm)', ylabel='Voltage',
#            directions=[0], discretized_obj=discretized_obj,
#            points_value=points_voltage, bbox=[-140, 150, 2000])
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    func_vol([0], ax=ax,
#         label='2DEG', marker='o', color='k', linestyle='None')
#    plt.show()
#
#    points_voltage_0 = discretized_obj.points_system_voltage[
#            points_charge[discretized_obj.points_system_voltage] != 0.0]
#    points_charge_0 = discretized_obj.points_system_charge
#    point_surface = reader.surface_between_points(points_r=points_voltage_0,
#                                                  points_l=points_charge_0,
#                                                  discretized_obj=discretized_obj)
#    print(points_charge[points_voltage_0])
#    print(sum(points_charge[points_voltage_0]))
#    print(sum(point_surface[point_surface != 0.0]))
#    print(sum(discretized_obj.mesh.points_hypervolume[points_voltage_0[
#            point_surface != 0.0]]))
#    func_vol([145], ax=ax,
#         label='Electrodes', marker='o', color='b', linestyle='None')
#    func_vol([80], ax=ax,
##         label='Dopants', marker='o', color='g', linestyle='None')
#    ax.legend()
#    ax.set_xlim(-2000, 2000)
#    plt.title('Voltage = 0')
    # Plot 1D cut of voltage
#    func_vol = p_plt.sliced_values_1d(
#            figsize=(10,10), xlabel='x(nm)', ylabel='Voltage',
#            directions=[1], discretized_obj=discretized_obj,
#            points_value=points_voltage, bbox=[-2500, 2500, 2000])
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    func_vol([-5], ax=ax,
#         label='2DEG', marker='o', color='k', linestyle='None')
##    func_vol([145], ax=ax,
##         label='Electrodes', marker='o', color='b', linestyle='None')
##    func_vol([80], ax=ax,
##         label='Dopants', marker='o', color='g', linestyle='None')
#    ax.legend()
#    ax.set_xlim(-2000, 2000)
#    plt.title('Voltage = 0')
#    # Plot 1D cut of votlage for updated grid voltage
#    func_vol_upd = p_plt.sliced_values_1d(
#            figsize=(10,10), xlabel='x(nm)', ylabel='Voltage',
#            directions=[1], discretized_obj=discretized_obj,
#            points_value=(points_voltage_upd),
#            bbox=[-3000, 3000, 200])
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    func_vol_upd([-5], ax=ax,
#         label='2DEG', marker='o', color='k', linestyle='None')
##    func_vol_upd([145], ax=ax,
##         label='Electrodes', marker='o', color='b', linestyle='None')
##    func_vol_upd([80], ax=ax,
##         label='Dopants', marker='o', color='g', linestyle='None')
#    ax.legend()
#    ax.set_xlim(-2000, 2000)
#    plt.title('Voltage  = -1')

    plot_voltage = p_plt.sliced_values_2d(
            figsize=(11, 11),
            colorbar_label=r'Voltage (V)',
            bbox=[-1500, 1500, -500, 500], xlabel='{0}(nm)'.format('X'),
            ylabel='{0}(nm)'.format('Y'), points_value=points_charge,
            discretized_obj=discretized_obj, npt=[1500, 1500],
            direction=2, cmap='seismic',
            interpolation=None, aspect='equal')

    plot_voltage([0])

#### Printing the charge at each region
    points_charge = points_charge * discretized_obj.mesh.points_hypervolume

example_2DEG()
