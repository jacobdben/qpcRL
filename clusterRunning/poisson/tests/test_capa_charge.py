'''
'''

from scipy.spatial import voronoi_plot_2d, cKDTree, Voronoi
from scipy import linalg
from scipy import constants

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm

from mayavi import mlab

from poisson.discrete import surface_calc as sc
from poisson.tools import plot as p_plt
from poisson.tools import post_process
from poisson.continuous import shapes
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)

def test_capa_charge():
    '''
    '''

    l = 2
    L = 2
    W = 30
    # Define your grid of points.( not a mesh)

    rect_int = shapes.Rectangle(length=(l, l), corner=(-l/2, -l/2))
    rect_out = shapes.Rectangle(length=(W, L + W/2), corner=(-W/2, -L))
    rect_pot = shapes.Rectangle(length=(W, 1.01), corner=(-W/2, -L - 1))

    bbox =  [-W/2, W/2, -L - 5, W/2]

#    rect1 = shapes.Rectangle(length=(10, 1))
#    rect2 = shapes.Rectangle(length=(5, 0.2), corner=(0, 1.1))
#    rect3 = shapes.Rectangle(length=(10, 10), corner=(0, 1.01))
#    rect4 = shapes.Rectangle(length=(2, 2), corner=(4, 4))
#    rect2 = shapes.Rectangle(length=(5, 3), corner=(0, 1.1))
#    rect3 = shapes.Rectangle(length=(5, 1), corner=(0, 4.1))

#    bbox = [0, 10, 0, 10]
    bstep=0.05
    bstep_3 = 0.1
    bstep2=0.01

#    grid = GridBuilder(meshs=[(
#            bbox, bstep,
#            shapes.Rectangle(length=(W, W + L), corner=( - W/2, -L - W/2)), 0)])
##
    grid = GridBuilder(meshs=[(bbox, bstep_3, rect_out, 0),
                         (bbox, bstep2, rect_int, 1),
                         (bbox, bstep, rect_pot, 2)])

    poissonpro_obj = ContinuousGeometry(
            space=rect_out,
            voltage=[rect_pot], mixed=[rect_int])

    poissonpro_obj.plot(direction=2, bbox=bbox,
                    plot_type='2D')(0)

    discretized_obj = DiscretePoisson(
            poissonpro_obj, grid=grid,
            selection={'Voronoi':[['voltage', '*']]})

#    rho_0 = 2.11e15
#    rho_0 = rho_0 * (discretized_obj.distance_normalization**2)
    tree =  cKDTree(discretized_obj.mesh.points)
    part_pos_y = 0.0
    part_pos_x = 0.0
    t, tt = tree.query([part_pos_x,  part_pos_y])
#    mixed = [([tt], [rho_0] )]
    mixed = [([tt], [1] )]
    voltage = 0.0
    system_eq_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            voltage_val = [(rect_pot, voltage)],
            mixed_val=mixed)

# Analytical

    A = -constants.elementary_charge / (2 * np.pi * constants.epsilon_0 * 1e-9)

    y = np.linspace(-W, W, 10000)
    r_0 = -2*L
    V = -A * (np.log(np.abs(y)) - np.log((np.abs(y - r_0))))


####    #Plotting
    plot_voltage_2d, plot_charge_2d = system_eq_inst.plot_cut_2d(direction=2)
    plot_voltage_2d(0, colorbar_label=r'Voltage (V)')
    plot_charge_2d(0, colorbar_label=r'Charge density($\#.nm^{{{-2}}}$)')
    plt.show()

    #bbox=[(-W/2, W/2, 3000)]
    plot_voltage, plot_charge = system_eq_inst.plot_cut_1d(
            directions=(0,), bbox='default', npoints=2000)

    fig2 = plt.figure()
    ax_voltage = fig2.add_subplot(111)

    t, data_volt = plot_voltage(
            (0, ), ax = ax_voltage, marker='.', color='k',
            label='Voltage simulation', linestyle='None')

    ax_voltage.set_xlabel('{0}(nm)'.format('y'))
    ax_voltage.set_ylabel('Voltage(V)')
    ax_voltage.legend(loc='upper center')
    ax_voltage.plot(y, V, 'g-')

#    ax_charge = ax_voltage.twinx()
    fig2 = plt.figure()
    ax_charge = fig2.add_subplot(111)

    tt, data_charge = plot_charge(
            (0, ), ax = ax_charge, marker='.', color='b',
            label='Charge simulation', linestyle='None')

    ax_charge.set_xlabel('{0}(nm)'.format('y'))
    ax_charge.set_ylabel(r'Charge density $(\#.nm^{{{-2}}})$')
    ax_charge.legend(loc='lower center')

    plt.show()

    post_process.regions_summary_charge(lin_prob_inst=system_eq_inst,
                                        verbose=True)

def test_dipole():
    '''
        For regular mesh with bstep = 0.5 -> cond number 8.374027126691689e+17
        TODO: test irregular mesh
    '''

    l = 2
    W = 10
    dist = 5
    # Define your grid of points.( not a mesh)

    rect_out = shapes.Rectangle(length=(W, W, W), corner=(-W/2, -W/2, -W/2))
    bbox =  [-W/2, W/2, -W/2, W/2, -W/2, W/2]

    bstep=0.5

    grid = GridBuilder(meshs=[(bbox, bstep, rect_out, 0)])

    poissonpro_obj = ContinuousGeometry(space=rect_out)

    discretized_obj = DiscretePoisson(poissonpro_obj, grid=grid)

    tree =  cKDTree(discretized_obj.mesh.points)
    part_pos_y = dist/2 + l/2
#    part_pos_y = 0.0
    part_pos_x = 0.0
    part_pos_x_2 = 0.0
    part_pos_y_2 = -part_pos_y
    t, tt = tree.query([part_pos_x,  part_pos_y, 0.0])
    t, tt_2 = tree.query([part_pos_x_2,  part_pos_y_2, 0.0])
    mixed = [([tt, tt_2], [1, -1] )]
#    mixed = [([tt_2], [+1])]

    charge_val_nul = (rect_out, 0)

    system_eq_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            charge_val=[charge_val_nul, mixed[0]])

# Analytical

    A = -constants.elementary_charge / (4 * np.pi * constants.epsilon_0 * 1e-9)

    y = np.linspace(-W/2, W/2, 10000)
    r_0 = dist/2 + l/2
    V = A * (1 / (np.abs(y - r_0)) - 1 / (np.abs(y + r_0)))

####    #Plotting
    system_eq_inst.plot_3d('both', ('Voltage (V)', 'Charge'),
                        scale_factor=(0.05, 3))

    plot_voltage_2d, plot_charge_2d = system_eq_inst.plot_cut_2d(direction=2)
    plot_voltage_2d(0, colorbar_label=r'Voltage (V)')
    plot_charge_2d(0, colorbar_label=r'Charge density($\#.nm^{{{-2}}}$)')
    plt.show()

    #bbox=[(-W/2, W/2, 3000)]
    plot_voltage, plot_charge = system_eq_inst.plot_cut_1d(
            directions=(0, 2), bbox='default', npoints=2000)

    fig2 = plt.figure()
    ax_voltage = fig2.add_subplot(111)

    t, data_volt = plot_voltage(
            (0, 0), ax = ax_voltage, marker='.', color='k',
            label='Voltage simulation', linestyle='None')

    ax_voltage.set_xlabel('{0}(nm)'.format('y'))
    ax_voltage.set_ylabel('Voltage(V)')
    ax_voltage.legend(loc='upper center')
#    ax_voltage.plot(y, V, 'g-')

    #    ax_charge = ax_voltage.twinx()
    fig2 = plt.figure()
    ax_charge = fig2.add_subplot(111)

    tt, data_charge = plot_charge(
            (0, 0), ax = ax_charge, marker='.', color='b',
            label='Charge simulation', linestyle='None')

    ax_charge.set_xlabel('{0}(nm)'.format('y'))
    ax_charge.set_ylabel(r'Charge density $(\#.nm^{{{-2}}})$')
    ax_charge.legend(loc='lower center')

    plt.show()

    post_process.regions_summary_charge(lin_prob_inst=system_eq_inst,
                                        verbose=True)
test_capa_charge()
#test_dipole()
