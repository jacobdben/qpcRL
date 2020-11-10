'''Parallel plate capacitance example
    from a continuous input.
    Version 2.0
    TODO: Fix missing points on spherical mesh
'''
import warnings
import time

from scipy.spatial import voronoi_plot_2d, cKDTree, Voronoi, qhull
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


def plot_points(point_charge, point_voltage, discretized_obj,
                coordinates, variable, direction, variable_2=None,
                npoints = 20000):

    if variable_2 is None:
        dimension = 2
    else:
        dimension = 3

    directions = np.arange(dimension, dtype=int)[np.logical_not(
            np.in1d(np.arange(dimension, dtype=int), direction))]

####   Plotting - simulation
    coordinates_mesh = discretized_obj.mesh.points[
            discretized_obj.points_system, direction]

    bbox = [(np.min(coordinates_mesh),
            np.max(coordinates_mesh),
            npoints)]

    xlabel= ['x', 'y', 'z']
    plot_voltage = p_plt.plot_1d(
            figsize=(10,10),
            directions=directions,
            class_inst=discretized_obj,
            points_value=point_voltage,
            bbox=bbox)

    plot_charge = p_plt.plot_1d(
        figsize=(10,10),
        directions=directions,
        class_inst=discretized_obj,
        points_value=point_charge,
        bbox=bbox)

    fig2 = plt.figure()
    ax_voltage = fig2.add_subplot(111)

    t, data_volt = plot_voltage([variable, variable_2], ax = ax_voltage,
                    return_data=True,
                    label='Voltage simulation', marker='.', color='k',
                    linestyle='None')

    ax_voltage.set_xlabel('{0}(nm)'.format(xlabel[direction]))
    ax_voltage.set_ylabel('Voltage(V)')

    ax_charge = ax_voltage.twinx()

    t, data_charge = plot_charge([variable, variable_2], ax = ax_charge,
                    return_data=True,
                    label='Charge simulation', marker='.', color='b',
                    linestyle='None')

    ax_charge.set_xlabel('{0}(nm)'.format(xlabel[direction]))
    ax_charge.set_ylabel('Charge(C)')
    ax_charge.legend(loc='lower center')

### Retrieving data - simulation
    coordinates = data_volt[0][np.logical_not(np.isnan(data_volt[1])), :]
    voltage_values = data_volt[1][np.logical_not(np.isnan(data_volt[1]))]

    V_max = np.max(voltage_values)
    V_min = np.min(voltage_values)

    # Because of symmetry around 0
    np_map = [coordinates[:, direction] <= 0, coordinates[:, direction] > 0]

    coordinates_unq = [None] * len(np_map)
    voltage_values_unq = [None] * len(np_map)

    ycoord_min = [None] * len(np_map)
    ycoord_max =  [None] * len(np_map)

    for map_ele_num, mapping_sign in enumerate(np_map):

        voltages = voltage_values[mapping_sign]
        coordinates_pn = coordinates[mapping_sign]

        ycoord_min[map_ele_num] = np.mean(coordinates_pn[
                np.argwhere(voltages == V_min)][:, 0, direction])
        ycoord_max[map_ele_num] = np.mean(coordinates_pn[
                np.argwhere(voltages == V_max)][:, 0, direction])

        # Find the positions with the same voltage value
        index = voltages[:].argsort()
        uniqval, uniqcount = np.unique(voltages[index], return_counts=True)

        cs = np.concatenate(([0], np.cumsum(uniqcount)))
        cs = np.array([cs[0:-1], cs[1:]]).T

        # Find the average of the positions for a constant voltage
        coord = np.ones(len(cs))
        for ele_num, ele_cd in enumerate(cs):

            coord[ele_num] = np.mean(coordinates_pn[index[ele_cd[0]:ele_cd[1]],
                                                 direction])


        coordinates_unq[map_ele_num] = coord
        voltage_values_unq[map_ele_num] = uniqval

#### Value Analytical
    A = [None] * len(ycoord_max)
    C = [None] * len(ycoord_min)
    analytic_charge = [None] * len(ycoord_min)

    for i in range(len(ycoord_max)):
        dim_fact = discretized_obj.distance_normalization
        A[i] = ((V_max - V_min)
             / (1 / ycoord_max[i] - 1 / ycoord_min[i]))

        analytic_charge[i] = A[i] * 4 * np.pi * (constants.epsilon_0 * dim_fact)

        C[i] = (V_min
             - (A[i] / ycoord_min[i]))

####   Plotting - analytical
    for i in range(len(A)):

        points_y_r = np.linspace(ycoord_max[i], ycoord_min[i], 100)

        voltage_th_plot_r = (A[i] / points_y_r
                      + C[i] * np.ones(len(points_y_r)))
        if i == 0:
            ax_voltage.plot(points_y_r,
                            voltage_th_plot_r,
                            'r-',
                            label=r'Voltage analytical')
        else:
            ax_voltage.plot(points_y_r,
                            voltage_th_plot_r,
                            'r-')

    if variable_2 is None:

        plt.title(r'Sphere with $V_1= {{{0:.0f}}}$' \
                  'and $V_2 = {{{1:.0f}}}$ along {3} = {2:.1f}'.format(
                      V_max, V_min, variable, xlabel[(direction + 1) % 2]))
    else:
        plt.title(r'Sphere with $V_1= {{{0:.0f}}}$' \
                  'and $V_2 = {{{1:.0f}}}$ along {3} = {2:.1f} and' \
                  ' {4} = {5:.1f}'.format(
                      V_max, V_min, variable, xlabel[directions[0]],
                      xlabel[directions[1]], variable_2))


#### Error evaluation

    error = [None] * len(ycoord_max)
    error_relat = [None] * len(ycoord_max)

    for i in range(len(ycoord_max)):

        voltage = (A[i] / coordinates_unq[i]
                   + ( C[i] * np.ones(len(coordinates_unq[i]))))

        error[i] = voltage - voltage_values_unq[i]
        error_relat[i] = np.divide(error[i], voltage_values_unq[i])

    print(ycoord_max)
    print(ycoord_min)

    ax_voltage.plot(np.concatenate(coordinates_unq),
                    np.concatenate(voltage_values_unq), 'go',
                    label='Voltage at voronoi cell center', markersize=5)
    ax_voltage.legend(loc='upper center')

#### Total charge - simulation
    point_charge = (point_charge
                    * discretized_obj.mesh.points_hypervolume)

    point_charge = (point_charge * constants.elementary_charge)

    total_charge=0
    relative_factor=0
    comp_Q = list()

    for key, points in discretized_obj.regions.voltage.tag_points.items():
        mask = np.isfinite(point_voltage[points])
        total_charge += np.sum(point_charge[points[mask]])
        relative_factor += np.mean(np.abs(point_charge[points[mask]]))

        comp_Q.append(np.sum(point_charge[points[mask]]))

    relative_total_charge = total_charge/relative_factor

    print('')
    print('------------------------------------------------------------------')
    print('------------------------------------------------------------------')
    if variable_2 is None:
        print('Test sphere -> Cut - along {1} = {0:.5f}'.format(
                variable, xlabel[(direction + 1) % 2]))
    else:
        print('Test sphere -> Cut - along {1} = {0:.1f} and' \
                  ' {2} = {3:.1f}'.format(
                          variable, xlabel[directions[0]],
                          xlabel[directions[1]], variable_2))
    print('')
    print('Error:')
    print(error)
    print('-----------------')
    print('max={0:.5f}'.format(np.max(np.abs(np.concatenate(error)))))
    print('min={0:.5f}'.format(np.min(np.abs(np.concatenate(error)))))
    print('mean={0:.5f}'.format(np.mean(np.abs(np.concatenate(error)))))
    print('-----------------')
    print('')
    print('------------------------------------------------------------------')
    print('')
    print('Relative error:')
    print(error_relat)
    print('')
    print('------------------------------------------------------------------')
    print('')
    print('Error:')
    print('-----------------')
    print('max={0:.5f}'.format(np.max(np.abs(error))))
    print('min={0:.5f}'.format(np.min(np.abs(error))))
    print('mean={0:.5f}'.format(np.mean(np.abs(error))))
    print('-----------------')
    print('Relative error:')
    print('-----------------')
    print('max={0:.5f}'.format(np.nanmax(np.abs(error_relat))))
    print('min={0:.5f}'.format(np.nanmin(np.abs(error_relat))))
    print('mean={0:.5f}'.format(np.nanmean(np.abs(error_relat))))
    print('-----------------')
    print('')
    print('\n', '------------------------------------------------------------')
    print(' Relative Total charge: {0}'.format(relative_total_charge))
    print((' Total charge : {0}').format(total_charge))
    print('\n', '------------------------------------------------------------', '\n')
    print('\n', '------------------------------------------------------------')
    print(' Analytical charge at each electrode: {0}'.format(analytic_charge))
    print(' Simulation charge at each electode: {0}'.format(comp_Q))
    print('\n', '------------------------------------------------------------', '\n')
    print('\n', '------------------------------------------------------------')

    if np.allclose(np.abs(ycoord_min), np.abs(ycoord_min[0])) is False:
            warnings.warn('\n The smallest radius is not the same for all'
                           + ' selected points. \n This migth cause'
                           + ' some problems ')


    plt.show()


def plotting(points_charge, points_voltage, discretized_obj,
             grid, func, bbox, plot_mesh=False, variable=0, variable_2=None,
             direction=1, direction_2d_cut=2, varialbe_2d_cut=0.0):


###### Plotting
#    # Plot voronoi cell , Neuman and Dirichlet regions
    if plot_mesh:
        fig = plt.figure()

        ax = fig.add_subplot(111)
        voronoi_plot_2d(discretized_obj.mesh, ax=ax)
        x, y = list(zip(*grid.mesh_points))
        ax.scatter(x, y, c=grid.point_label, cmap=cm.jet)

        # remove non-calculated points before plot
        t = discretized_obj.points_system

        t = t.astype(int)
        xx = discretized_obj.mesh.points[t, 0]
        yy = discretized_obj.mesh.points[t, 1]
        ax.scatter(xx, yy, c='r', marker='+', s=300)
        plt.axes().set_aspect('equal', 'datalim')

        plt.show()


    plot_points(points_charge, points_voltage, discretized_obj,
                coordinates=discretized_obj.mesh.points, variable=variable,
                direction=direction,
                variable_2=variable_2)

    axis = ['x', 'y', 'z']
    del axis[direction_2d_cut]
    func2 = p_plt.plot_geometry_2d(figsize=(11, 11), bbox=bbox,
                                   fctnames=['elc1', 'isol', 'elec2'],
                                   functions=[func[0], func[1], func[2]],
                                   npoints=[1000, 1000],
                                   xlabel='{0}(nm)'.format(axis[0]),
                                   ylabel='{0}(nm)'.format(axis[1]),
                                   direction=direction_2d_cut,
                                   cmap=ListedColormap(['w','orange', 'blue',
                                                        'red']),
                                   vmin=-2, vmax=2,
                                   aspect='equal')
    func2(varialbe_2d_cut)

    func2 = p_plt.plot_values_2d(figsize=(11, 11),
                                 colorbar_label='Voltage (V)',
                                 bbox=bbox,
                                 xlabel='{0}(nm)'.format(axis[0]),
                                 ylabel='{0}(nm)'.format(axis[1]),
                                 points_value=points_voltage,
                                 class_inst=discretized_obj,
                                 npoints=[1000, 1000],
                                 direction=direction_2d_cut, cmap='seismic',
                                 aspect='equal')

    func2(varialbe_2d_cut)

    points_charge = points_charge * discretized_obj.charge_normalization
    points_charge = points_charge / (discretized_obj.distance_normalization**2)
    func2 = p_plt.plot_values_2d(figsize=(11, 11),
                                   colorbar_label=r'Charge density($C.m^{{{-2}}}$)',
                                   bbox=bbox,
                                   xlabel='{0}(nm)'.format(axis[0]),
                                   ylabel='{0}(nm)'.format(axis[1]),
                                   points_value=points_charge,
                                   class_inst=discretized_obj,
                                   npoints=[1000, 1000],
                                   direction=direction_2d_cut, cmap='seismic',
                                   aspect='equal')

#
    func2(varialbe_2d_cut)


    plt.show()


def test_sphere(**kw):
    '''
        Sphere with a voltage difference betwen its core and its surface
        TODO: Finish removing mutable attributes
    '''

    # build the mesh
    epislon = 1e-14
    r1 = 3.0 + epislon
    r1_2a = 2.90 + epislon
    r1_2b = 3.1 + epislon
    r2 = 8.0 + epislon
    r2_3a = 7.90 + epislon
    r2_3b = 8.1 + epislon
    r3 = 10.0 + epislon

    radius1 = [r1, r1, r1]
    radius1a = [r1, r1, r1]
    radius2 = [r2, r2, r2]
    radius2a = [r2, r2, r2]
    radius3 = [r3, r3, r3]
    radius2_3a = [r2_3a, r2_3a, r2_3a]
    radius2_3b = [r2_3b, r2_3b, r2_3b]
    radius1_2a = [r1_2a, r1_2a, r1_2a]
    radius1_2b = [r1_2b, r1_2b, r1_2b]
    center = [0.0, 0.0, 0.0]

    cercle1 = shapes.Ellipsoid(radius1, center)
    cercle2 = (shapes.Ellipsoid(radius2, center)
                - shapes.Ellipsoid(radius1a, center))
    cercle3 = (shapes.Ellipsoid(radius3, center)
                - shapes.Ellipsoid(radius2a, center))
    cercle_1_2 = (shapes.Ellipsoid(radius1_2b, center)
                - shapes.Ellipsoid(radius1_2a, center))
    cercle_2_3 = (shapes.Ellipsoid(radius2_3b, center)
                - shapes.Ellipsoid(radius2_3a, center))

    sphericalbox = [-10, 10, 0, 2 * np.pi, 0, np.pi]
    step = [0.5, np.pi / 20, np.pi / 20]
    step_2 = [0.01, np.pi/ 20, np.pi/ 20]

    grid = GridBuilder(build_mesh=False)

    grid.add_mesh_spherical(sphericalbox, step, cercle1, 0)
    grid.add_mesh_spherical(sphericalbox, step, cercle2, 1)
    grid.add_mesh_spherical(sphericalbox, step, cercle3, 2)
    grid.add_mesh_spherical(sphericalbox, step_2, cercle_1_2, 3)
    grid.add_mesh_spherical(sphericalbox, step_2, cercle_2_3, 4)
#    mesh_points, l = repeated_values.remove_from_grid(grid.mesh_points)

#    p_plt.points_3D_mavi(grid, scale_factor=.04)

    # Define the system
    die_co = 1
    dielectric = [shapes.PiecewiseFunction(((cercle1 + cercle2 + cercle3),
                                            die_co))]

    poissonpro = ContinuousGeometry(
            space=(cercle1 + cercle2 + cercle3), dielectric=dielectric,
            voltage=[cercle1, cercle3], charge=[cercle2])

 ### Discretize your system and build the capacitance matrix
    discretized_obj = DiscretePoisson(
            poissonpro, grid=grid,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    # Define boundary conditions
    dirichlet_val = [(cercle3, 4.0),
                     (cercle1, 0.0)]
    neuman_val = [(cercle2, 0.0)]

    # build and solve linear system
    t1 = time.time()
    linear_problem_isnt = LinearProblem(
            discretized_obj, dirichlet_val, neuman_val, is_charge_density=True,
            solve_problem=False)
    t2 = time.time()

    points_charge, points_voltage = linear_problem_isnt.solve(solver_type='spsolve')

#
#    plotting(points_charge, points_voltage, discretized_obj,
#             grid, func=[cercle1, cercle2, cercle3], bbox=[-10,10,-10,10],
#             plot_mesh=False, variable=0, variable_2=0.0, direction=1,
#             direction_2d_cut=1, varialbe_2d_cut = 0.0)

    t3 = time.time()
    points_charge, points_voltage = linear_problem_isnt.solve(solver_type='mumps')
    t3b = time.time()

    plotting(points_charge, points_voltage, discretized_obj,
             grid, func=[cercle1, cercle2, cercle3], bbox=[-10,10,-10,10],
             plot_mesh=False, variable=0, variable_2=0.0, direction=1,
             direction_2d_cut=1, varialbe_2d_cut = 0.0)

    voltage_val = [(cercle3, 8.0),
                     (cercle1, 0.0)]

    t4 =  time.time()
    linear_problem_isnt.update(voltage_val=voltage_val)
    t5 = time.time()
    points_charge, points_voltage = linear_problem_isnt.solve(solver_type='mumps',
                                                        factorize=False)
    t6 = time.time()

    print('Number of points: {0}'.format(
            discretized_obj.mesh.npoints))
    print('Build system of equations obj: {0}'.format(t2 - t1))
    print('Solve using scipy sparse obj: {0}'.format(t3 - t2))
    print('Solve using mumps obj: {0}'.format(t3b - t3))
    print('Update system of equations obj: {0}'.format(t5 - t4))
    print('Solve using mumps without factorization obj: {0}'.format(t6 - t5))

    plotting(points_charge, points_voltage, discretized_obj,
             grid, func=[cercle1, cercle2, cercle3], bbox=[-10,10,-10,10],
             plot_mesh=False, variable=0, variable_2=0.0, direction=1,
             direction_2d_cut=1, varialbe_2d_cut = 0.0)

    post_process.regions_summary_charge(linear_problem_isnt)


test_sphere()