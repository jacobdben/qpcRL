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

from poisson.continuous import shapes
from poisson.discrete import finite_volume
from poisson.discrete import surface_calc as sc
from poisson.tools import post_process
from poisson.tools import plot as p_plt
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)

####### Helper functions

def make_dipole(l, W, dist, small_step, large_step):
    '''
        Make the DiscretePoisson for the monopole geometry.
    '''
    # Define your grid of points.( not a mesh)
    rect_int = shapes.Rectangle(length=(l, l), corner=(-l/2, dist/2))
    rect_out = shapes.Rectangle(length=(W, W), corner=(-W/2, -W/2))
    rect_int_2 = shapes.Rectangle(length=(l, l), corner=(-l/2, -dist/2 -l))
    bbox =  [-W/2, W/2, -W/2, W/2]
    grid = GridBuilder(meshs=[(bbox, large_step, rect_out, 0),
                         (bbox, small_step, rect_int, 1),
                         (bbox, small_step, rect_int_2, 2)])

    poissonpro_obj = ContinuousGeometry(
            space=rect_out, mixed=[rect_int, rect_int_2])

    discretized_obj = DiscretePoisson(
            poissonpro_obj, grid=grid)

    return bbox,  grid, poissonpro_obj, discretized_obj, rect_out


def make_mixed_2(
        pos_x1, pos_x2, pos_y1=None, pos_y2=None,
        charge_val = [-1, 1], discretized_obj=None, dist=None, l=None):
    '''
        Construct the mixed parameter
    '''

    tree =  cKDTree(discretized_obj.mesh.points)
    if pos_y1 is None:
        pos_y1 = dist/2 + l/2
    if pos_y2 is None:
        pos_y2 = - pos_y1

    t, tt = tree.query([pos_x1, pos_y1])
    t, tt_2 = tree.query([pos_x2, pos_y2])
    mixed = [([tt_2, tt], charge_val)]

    return mixed


def plotter_2D(plot_voltages, title, part_pos_x, points_charges, points_voltages,
               bbox, discretized_obj, cut_charge_2D=0.0, cut_voltage_2D=0.0,
               analytical_voltage=None):
    '''
        Help with plotting
    '''
    fig2 = plt.figure()
    ax_voltage = fig2.add_subplot(111)

    if not isinstance(plot_voltages, (list, tuple)):
        plot_voltages = [plot_voltages]
    if not isinstance(points_charges, (list, tuple)):
        points_charges = [points_charges]
    if not isinstance(points_voltages, (list, tuple)):
        points_voltages = [points_voltages]

    colors = ['r', 'b']
    marker = ['x', '.']
    for i, plot_voltage in enumerate(plot_voltages):
        t, data_volt = plot_voltage(
                variable= [part_pos_x], ax = ax_voltage, return_data=True,
                marker=marker[i],
                label='Voltage {0}'.format(i), linestyle='None', color=colors[i])

    if analytical_voltage is not None:
        ax_voltage.plot(analytical_voltage[0], analytical_voltage[1], 'g-')

    ax_voltage.set_xlabel('{0}(nm)'.format('y'))
    ax_voltage.set_ylabel('Voltage(V)')
    ax_voltage.legend()
    plt.title(title)
    plt.show()

    direction_2d_cut = 2
    axis = ['x', 'y', 'z']
    del axis[direction_2d_cut]

    # Plot 2D cut of charge variation:
    for points_voltage in points_voltages:
        plot_voltage = p_plt.plot_values_2d(
            figsize=(5, 5),
            colorbar_label=r'Voltage (V)',
            bbox=bbox, xlabel='{0}(nm)'.format(axis[0]),
            ylabel='{0}(nm)'.format(axis[1]), points_value=points_voltage,
            class_inst=discretized_obj, npoints=[150, 150],
            direction=direction_2d_cut, cmap='seismic',
            interpolation=None, aspect='equal')

        plot_voltage(cut_voltage_2D)

    for points_charge in points_charges:

        plot_charge = p_plt.plot_values_2d(
            figsize=(5, 5),
            colorbar_label=r'Charge density($\#.nm^{{{-2}}}$)',
            bbox=bbox, xlabel='{0}(nm)'.format(axis[0]),
            ylabel='{0}(nm)'.format(axis[1]), points_value=points_charge,
            class_inst=discretized_obj, npoints=[100, 100],
            direction=direction_2d_cut, cmap='seismic',
            interpolation=None, aspect='equal', vmin=-1e-5, vmax=1e-5 )

        plot_charge(cut_charge_2D)

    plt.show()

####### Simulation

####### Dipole simulation 2D

def dipole_0_Dirc(W=100, l=2, dist=5, step=0.5,
                      return_plot_voltage=False,
                      part_pos_x=0.0, charge_val=[-1, 1]):
    '''
        Charge is fixed at all nodes in the mesh.
        The voltage is not fixed anywhere
        Plot a dipole with a regular mesh.
    '''
    # Define your grid of points
    rect_out = shapes.Rectangle(length=(W, W), corner=(-W/2, -W/2))
    bbox =  [-W/2, W/2, -W/2, W/2]
    grid = GridBuilder(meshs=[(bbox, step, rect_out, 0)])

    poissonpro_obj = ContinuousGeometry(space=rect_out)

    discretized_obj = DiscretePoisson(
            poissonpro_obj, grid=grid)

    mixed = make_mixed_2(
            pos_x1=part_pos_x, pos_x2=part_pos_x, pos_y1=dist/2 + l/2,
            pos_y2=None, charge_val = charge_val,
            discretized_obj=discretized_obj, dist=dist, l=l)

    charge_val_nul = shapes.PiecewiseFunction((rect_out, 0))

    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            charge_val=[charge_val_nul, mixed[0]])
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    # Analytical
    A = -constants.elementary_charge / (2 * np.pi * constants.epsilon_0 * 1e-9)
    y = np.linspace(-W/2, W/2, 10000)
    r_0 = dist/2 + l/2
    V = -A * (np.log(np.abs(y - r_0)) - np.log(np.abs(y + r_0)))

    #Plotting
    plot_voltage = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=points_voltage, bbox=[(-W/2, W/2, 3000)])

    # Plot 2D cut of charge variation:
    if return_plot_voltage:
        return plot_voltage, part_pos_x

    title = ('Dipole 2D - regular - W= {0}, l= {1},'
              ' step = {2}').format(W, l, step)

    plotter_2D(plot_voltage, title, part_pos_x, cut_charge_2D=0.0,
                   cut_voltage_2D=0.0, points_charges=points_charge,
                   points_voltages=points_voltage, bbox=[-W/2, W/2, -W/2, W/2],
                   discretized_obj=discretized_obj, analytical_voltage=(y, V))


def dipole_irreg_0_Dirc(W=150, l=2, dist=5, large_step=2, small_step=0.25,
                       return_plot_voltage=False, part_pos_x=0.0,
                       charge_val=[-1, 1]):
    '''
        Same as test_dipole_2d but with an irregular mesh
        Plot a dipole but with a irregular mesh.
        Voltage is not fixed anywhere.
    '''

    bbox, grid, poissonpro_obj, discretized_obj, rect_out = make_dipole(
            l, W, dist, small_step, large_step)

#    voronoi_plot_2d(discretized_obj.mesh)
#    plt.show()
#    raise

    mixed = make_mixed_2(
            pos_x1=part_pos_x, pos_x2=part_pos_x, pos_y1=dist/2 + l/2,
            pos_y2=None, charge_val = charge_val,
            discretized_obj=discretized_obj, dist=dist, l=l)

    charge_val_nul = shapes.PiecewiseFunction((rect_out, 0))

    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            charge_val=[charge_val_nul, mixed[0]])
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    # Analytical
    A = -constants.elementary_charge / (2 * np.pi * constants.epsilon_0 * 1e-9)

    y = np.linspace(-W/2, W/2, 10000)
    r_0 = dist/2 + l/2
    V = -A * (np.log(np.abs(y - r_0)) - np.log(np.abs(y + r_0)))

    #Plotting
    plot_voltage = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=points_voltage, bbox=[(-W/2, W/2, 3000)])

    # Plot 2D cut of charge variation:
    if return_plot_voltage:
        return plot_voltage, part_pos_x

    title = ('Dipole 2D - W= {0}, l= {1},'
              ' large_step = {2}, small_step={3}').format(W, l, large_step,
                             small_step)

    plotter_2D(plot_voltage, title, part_pos_x, cut_charge_2D=0.0,
                   cut_voltage_2D=0.0, points_charges=points_charge,
                   points_voltages=points_voltage, bbox=[-W/2, W/2, -W/2, W/2],
                   discretized_obj=discretized_obj, analytical_voltage=(y, V))


def dipole_irreg_1_Dirc(
        W=500, l=2, dist=5, large_step=2, small_step=0.25,
        return_plot_voltage=False, part_pos_y=None, part_pos_x = 0.0,
        charge_val=[-1, 1]):
    '''
        With one dirichlet point
    '''

    bbox, grid, poissonpro_obj, discretized_obj, rect_out = make_dipole(
            l, W, dist, small_step, large_step)

    mixed = make_mixed_2(
            pos_x1=part_pos_x, pos_x2=part_pos_x, pos_y1=part_pos_y,
            pos_y2=None, charge_val = charge_val,
            discretized_obj=discretized_obj, dist=dist, l=l)

    charge_val_nul = shapes.PiecewiseFunction((rect_out, 0))

    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            charge_val=[charge_val_nul],
            mixed_val=mixed, pos_voltage_mixed=np.array([mixed[0][0][0]]),
            solve_problem=True)
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    # Analytical
    A = (-constants.elementary_charge / (2 * np.pi * constants.epsilon_0 * 1e-9)
         * -1)
    y = np.linspace(-W/2, W/2, 100000)
    r_0 = discretized_obj.mesh.points[mixed[0][0][0]][1]
    r_1 = discretized_obj.mesh.points[mixed[0][0][1]][1]
    V = -A * (+np.log(np.abs(y - r_0))
              -np.log(np.abs(y - r_1))) - 17.6108
    # Plotting

    post_process.regions_summary_charge(linear_prob_inst)

    plot_voltage = p_plt.plot_1d(
            figsize=(10, 10), directions=[0], class_inst=discretized_obj,
            points_value=points_voltage, bbox=[(-W/2, W/2, 3000)])

    # Update charge values
    linear_prob_inst.update(mixed_val=[([mixed[0][0][0], mixed[0][0][1]],
                                        np.negative(charge_val))])
    linear_prob_inst.solve(new_instance=False, factorize=False, verbose=True)
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    plot_voltage_2 = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=-points_voltage, bbox=[(-W/2, W/2, 3000)])

    # Plot 2D cut of charge variation:
    if return_plot_voltage:
        return plot_voltage, part_pos_x

    title = ('Monopole 2D - W= {0}, l= {1},'
             ' large_step = {2}, small_step={3}').format(W, l, large_step,
                                                         small_step)

    plotter_2D(
            [plot_voltage, plot_voltage_2], title, part_pos_x,
            cut_charge_2D=0.0, cut_voltage_2D=0.0,
            points_charges=points_charge,
            points_voltages=points_voltage, bbox=[-W/2, W/2, -W/2, W/2],
            discretized_obj=discretized_obj,
            analytical_voltage=(y, V))


def dipole_irreg_2_Dirc(
        W=200, l=2, dist=5, large_step=2, small_step=0.25,
        return_plot_voltage=False, part_pos_y=None, part_pos_x = 0.0,
        charge_val=[1, -1]):
    '''
        WIth two dirichlet points
    '''

    bbox, grid, poissonpro_obj, discretized_obj, rect_out = make_dipole(
            l, W, dist, small_step, large_step)

    mixed = make_mixed_2(
            pos_x1=part_pos_x, pos_x2=part_pos_x, pos_y1=part_pos_y,
            pos_y2=None, charge_val = charge_val,
            discretized_obj=discretized_obj, dist=dist, l=l)

    charge_val_nul = shapes.PiecewiseFunction((rect_out, 0))

    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            charge_val=[charge_val_nul],
            mixed_val=mixed, pos_voltage_mixed=np.array([mixed[0][0][0],
                                                         mixed[0][0][1]]),
            solve_problem=True)
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    # Analytical

    A = (-constants.elementary_charge / (2 * np.pi * constants.epsilon_0 * 1e-9)
         * 0.05922381918122215)

    y = np.linspace(-W/2, W/2, 100000)
    r_0 = discretized_obj.mesh.points[mixed[0][0][0]][1]
    r_1 = discretized_obj.mesh.points[mixed[0][0][1]][1]
    V = -A * (-np.log(np.abs(y - r_0))
              +np.log(np.abs(y - r_1)))

    # Plotting

    post_process.regions_summary_charge(linear_prob_inst)

    plot_voltage = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=points_voltage, bbox=[(-W/2, W/2, 3000)])

    linear_prob_inst.update(mixed_val=[([mixed[0][0][0],
                                         mixed[0][0][1]],
                                        np.negative(charge_val))])
    linear_prob_inst.solve(new_instance=False, factorize=False, verbose=True)
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    plot_voltage_2 = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=-points_voltage, bbox=[(-W/2, W/2, 3000)])

    # Plot 2D cut of charge variation:
    if return_plot_voltage:
        return plot_voltage, part_pos_x

    title = ('Monopole 2D - W= {0}, l= {1},'
              ' large_step = {2}, small_step={3}').format(W, l, large_step,
                             small_step)

    plotter_2D(
            [plot_voltage, plot_voltage_2], title, part_pos_x,
            cut_charge_2D=0.0, cut_voltage_2D=0.0, points_charges=points_charge,
            points_voltages=points_voltage, bbox=[-W/2, W/2, -W/2, W/2],
            discretized_obj=discretized_obj,
            analytical_voltage=(y, V))

##### Monopole simulation 2D
def test_monopole_2D_irr(W=30, l=2, dist=5, large_step=0.1, small_step=0.01,
                         return_plot_voltage=False, part_pos_y=0.0,
                         part_pos_x = 0.0):
    '''
        Charge fixed at every node in the mesh.
        Voltage fixed nowhere.
    '''
    bbox, grid, poissonpro_obj, discretized_obj, rect_out = make_dipole(
            l, W, dist, small_step, large_step)

#    voronoi_plot_2d(discretized_obj.mesh)
#    plt.show()
#    raise


    old_mixed = make_mixed_2(
            pos_x1=part_pos_x, pos_x2=part_pos_x, pos_y1=part_pos_y,
            pos_y2=None, charge_val = [-1, 1],
            discretized_obj=discretized_obj, dist=dist, l=l)

    charge_val_nul = shapes.PiecewiseFunction((rect_out, 0))


    mixed = [([old_mixed[0][0][0]], [+1])]
    charge_val_nul = shapes.PiecewiseFunction((rect_out, 0))
    print(mixed)
    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            charge_val=[charge_val_nul, mixed[0]])
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    # Analytical

    A = -constants.elementary_charge / (2 * np.pi * constants.epsilon_0 * 1e-9)

    y = np.linspace(-W/2, W/2, 10000)
    r_0 = dist/2 + l/2
    V = -A * (np.log(np.abs(y - r_0)) - np.log(np.abs(y + r_0)))

    # Plotting
    plot_voltage = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=points_voltage, bbox=[(-W/2, W/2, 3000)])

    linear_prob_inst.update(charge_val=[([old_mixed[0][0][0]], [-1])])
    linear_prob_inst.solve(new_instance=False, factorize=False, verbose=True)
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    plot_voltage_2 = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=-points_voltage, bbox=[(-W/2, W/2, 3000)])

    # Plot 2D cut of charge variation:
    if return_plot_voltage:
        return plot_voltage, part_pos_x

    title = ('Monopole 2D - W= {0}, l= {1},'
              ' large_step = {2}, small_step={3}').format(W, l, large_step,
                             small_step)

    plotter_2D(
            [plot_voltage, plot_voltage_2], title, part_pos_x,
            cut_charge_2D=0.0, cut_voltage_2D=0.0, points_charges=points_charge,
            points_voltages=points_voltage, bbox=[-W/2, W/2, -W/2, W/2],
            discretized_obj=discretized_obj)

##### Tripole simulation 2D
def tripole__irreg_2_Dirc(
        W=200, l=2, dist=5, large_step=2, small_step=0.05,
        return_plot_voltage=False, part_pos_y=None, part_pos_x = 0.0):
    '''
        Fixate voltage to have at least one dirichlet point
    '''

    # Define your grid of points.( not a mesh)
    rect_int = shapes.Rectangle(length=(l, l), corner=(-l/2, dist/2))
    rect_out = shapes.Rectangle(length=(W, W), corner=(-W/2, -W/2))
    rect_middle = shapes.Rectangle(length=(dist, dist), corner=(0.0, -dist/2))
    rect_int_2 = shapes.Rectangle(length=(l, l), corner=(-l/2, -dist/2 -l))
    bbox =  [-W/2, W/2, -W/2, W/2]
    bbox_small = [-(l + dist), l + dist, -(l + dist), (l + dist)]
    grid = GridBuilder(meshs=[(bbox, large_step, rect_out, 0),
                         (bbox_small, small_step, rect_int, 1),
                         (bbox_small, small_step, rect_int_2, 2),
                         (bbox_small, small_step, rect_middle, 3)])

    poissonpro_obj = ContinuousGeometry(
            space=rect_out, mixed=[rect_int, rect_int_2])

    discretized_obj = DiscretePoisson(
            poissonpro_obj, grid=grid)

    tree =  cKDTree(discretized_obj.mesh.points)
    if part_pos_y is None:
        part_pos_y = dist/2 + l/2
    i, center = tree.query([0.0,  0.0])
    t, tt = tree.query([part_pos_x,  part_pos_y])
    t, tt_2 = tree.query([part_pos_x,  -part_pos_y])
    mixed = [([tt_2, tt, center], [0, 0, 1])]
    charge_val_nul = shapes.PiecewiseFunction((rect_out, 0))

    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=False,
            charge_val=[charge_val_nul],
            mixed_val=mixed, pos_voltage_mixed=np.array([tt_2, tt]),
            solve_problem=True)
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    # Analytical
    A = -constants.elementary_charge / (2 * np.pi * constants.epsilon_0 * 1e-9)
    y = np.linspace(-W/2, W/2, 100000)

    r_0 = discretized_obj.mesh.points[tt_2][1]
    r_1 = discretized_obj.mesh.points[tt][1]
    V = -A * (-0.49999999992794597*np.log(np.abs(y - r_0))
              -0.49999999992794597*np.log(np.abs(y - r_1))
              + np.log(np.abs(y))) - 7.43801
    # Plotting

    post_process.regions_summary_charge(linear_prob_inst)

    plot_voltage = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=points_voltage, bbox=[(-W/2, W/2, 3000)])

    linear_prob_inst.update(mixed_val=[([tt_2, tt, center], [0, 0, -1])])
    linear_prob_inst.solve(new_instance=False, factorize=False, verbose=True)
    points_charge = linear_prob_inst.points_charge
    points_voltage = linear_prob_inst.points_voltage

    plot_voltage_2 = p_plt.plot_1d(
            figsize=(10,10), directions=[0], class_inst=discretized_obj,
            points_value=-points_voltage, bbox=[(-W/2, W/2, 3000)])

    # Plot 2D cut of charge variation:
    if return_plot_voltage:
        return plot_voltage, part_pos_x

    title = ('Monopole 2D - W= {0}, l= {1},'
              ' large_step = {2}, small_step={3}').format(W, l, large_step,
                             small_step)

    plotter_2D(
            [plot_voltage, plot_voltage_2], title, part_pos_x,
            cut_charge_2D=0.0, cut_voltage_2D=0.0, points_charges=points_charge,
            points_voltages=points_voltage, bbox=[-W/2, W/2, -W/2, W/2],
            discretized_obj=discretized_obj,
            analytical_voltage=(y, V))


#dipole_0_Dirc()
#dipole_irreg_0_Dirc()
test_monopole_2D_irr()
#dipole_irreg_1_Dirc()
#dipole_irreg_2_Dirc()
#tripole__irreg_2_Dirc()
