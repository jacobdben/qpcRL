''' Circular parallel plate capacitance (2D) example
    Both using a rectangular mesh or a circular mesh
    TODO: Check charge discrepancey ??? Confirm method.
    '''
import warnings

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
from poisson.tools import plot as p_plt
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)

def plot_points(point_charge, point_voltage, discretized_obj,
                coordinates, variable, direction, variable_2=None,
                npoints=20000):

    if variable_2 is None:
        dimension = 2
        variables = [variable]
    else:
        dimension = 3
        variables = [variable, variable_2]


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

    t, data_volt = plot_voltage(variables, ax = ax_voltage,
                    return_data=True,
                    label='Voltage simulation', marker='.', color='k',
                    linestyle='None')

    ax_voltage.set_xlabel('{0}(nm)'.format(xlabel[direction]))
    ax_voltage.set_ylabel('Voltage(V)')

    ax_charge = ax_voltage.twinx()

    t, data_charge = plot_charge(variables, ax = ax_charge,
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

    # Theory
#    A = ((V_max - V_min)
#         / np.log(np.abs(ycoord_max / ycoord_min)))
#
#    print(A * 2 * np.pi * (constants.epsilon_0 * dim_fact))
#    print(comp_Q)
#
#    C = (V_min
#         - A * np.log(np.abs(ycoord_min)))

#### Value Analytical
    A = [None] * len(ycoord_max)
    C = [None] * len(ycoord_min)
    analytic_charge = [None] * len(ycoord_min)

    for i in range(len(ycoord_max)):

        dim_fact = discretized_obj.distance_normalization

        A[i] = ((V_max - V_min)
             / np.log(np.abs(ycoord_max[i] / ycoord_min[i])))

        analytic_charge[i] = A[i] * 2 * np.pi * (constants.epsilon_0 * dim_fact)

        C[i] = (V_min
             - (A[i] * np.log(np.abs(ycoord_min[i]))))

####   Plotting - analytical
    for i in range(len(A)):

        points_y_r = np.linspace(ycoord_max[i], ycoord_min[i], 100)

        voltage_th_plot_r = (A[i] * np.log(np.abs(points_y_r))
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

        plt.title(r'Cylinder with $V_1= {{{0:.0f}}}$' \
                  'and $V_2 = {{{1:.0f}}}$ along {3} = {2:.1f}'.format(
                      V_max, V_min, variable, xlabel[(direction + 1) % 2]))
    else:
        plt.title(r'Cylinder with $V_1= {{{0:.0f}}}$' \
                  'and $V_2 = {{{1:.0f}}}$ along {3} = {2:.1f} and' \
                  ' {4} = {5:.1f}'.format(
                      V_max, V_min, variable, xlabel[directions[0]],
                      xlabel[directions[1]], variable_2))

#### Error evaluation

    error = [None] * len(ycoord_max)
    error_relat = [None] * len(ycoord_max)

    for i in range(len(ycoord_max)):

        voltage = (A[i] * np.log(np.abs(coordinates_unq[i]))
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
        print('Test Cylinder -> Cut - along {1} = {0:.5f}'.format(
                variable, xlabel[(direction + 1) % 2]))
    else:
        print('Test Cylinder -> Cut - along {1} = {0:.1f} and' \
                  ' {2} = {3:.1f}'.format(
                          variable, xlabel[directions[0]],
                          xlabel[directions[1]], variable_2))

    print('')
    print('------------------------------------------------------------------')
    print('------------------------------------------------------------------')
    print(('Cylindrical parallel plate capacitance - cut along'
          + ' {1} = {0:.5f}').format(variable, ['x', 'y' ][(direction + 1) % 2]))
    print('')
    print('Error:')
    print(error)
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
    print('max={0:.5f}'.format(np.nanmax(np.abs(np.concatenate(error)))))
    print('min={0:.5f}'.format(np.nanmin(np.abs(np.concatenate(error)))))
    print('mean={0:.5f}'.format(np.nanmean(np.abs(np.concatenate(error)))))
    print('-----------------')
    print('Relative error:')
    print('-----------------')
    print('max={0:.5f}'.format(np.nanmax(np.abs(np.concatenate(error_relat)))))
    print('min={0:.5f}'.format(np.nanmin(np.abs(np.concatenate(error_relat)))))
    print('mean={0:.5f}'.format(np.nanmean(np.abs(np.concatenate(error_relat)))))
    print('-----------------')
    print('')
    print('\n', '------------------------------------------------------------')
    print('Relative Total charge: {0}'.format(relative_total_charge))
    print(('Total charge : {0}').format(total_charge))
    print('\n', '------------------------------------------------------------', '\n')
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

    return error, error_relat


def plotting(points_charge, points_voltage, discretized_obj,
             grid, func, bbox, plot_mesh=False, direction_1d_cut=1, variable_1d=0.0,
             direction_2d_slice=2, variable_2d=0.0):


###### Plotting
#    # Plot voronoi cell , Neuman and Dirichlet regions
    if plot_mesh:
        fig = plt.figure()

        ax = fig.add_subplot(111)
        voronoi_plot_2d(discretized_obj.mesh, ax=ax)
        x, y = list(zip(*grid.points))
        ax.scatter(x, y, c=grid.point_label, cmap=cm.jet)

        # remove non-calculated points before plot
        t = discretized_obj.points_system

        t = t.astype(int)
        xx = discretized_obj.mesh.points[t, 0]
        yy = discretized_obj.mesh.points[t, 1]
        ax.scatter(xx, yy, c='r', marker='+', s=300)
        plt.axes().set_aspect('equal', 'datalim')

        plt.show()


    error, error_realt = plot_points(
            points_charge, points_voltage, discretized_obj,
            coordinates=discretized_obj.mesh.points, variable=variable_1d,
            direction=direction_1d_cut)

    func2 = p_plt.plot_geometry_2d(figsize=(11, 11), bbox=bbox,
                                     fctnames=['elc1', 'isol', 'elec2'],
                                     functions=[func[0], func[1], func[2]],
                                     npoints=[1000, 1000],
                                     xlabel='x(nm)', ylabel='y(nm)',
                                     direction=direction_2d_slice,
                                     cmap=ListedColormap(['w',
                                      'orange', 'blue', 'red']), vmin=-2, vmax=2,
                                     aspect='equal')
    func2(variable_2d)

    func2 = p_plt.plot_values_2d(figsize=(11, 11), colorbar_label='Voltage (V)',
                                   bbox=bbox, xlabel='x(nm)', ylabel='y(nm)',
                                   points_value=points_voltage,
                                   class_inst=discretized_obj,
                                   npoints=[1000, 1000],
                                   direction=direction_2d_slice, cmap='seismic',
                                   aspect='equal')

    func2(variable_2d)

    points_charge = points_charge * discretized_obj.charge_normalization
    points_charge = points_charge / (discretized_obj.distance_normalization**2)
    func2 = p_plt.plot_values_2d(figsize=(11, 11),
                                   colorbar_label=r'Charge density($C.m^{{{-2}}}$)',
                                   bbox=bbox, xlabel='x(nm)', ylabel='y(nm)',
                                   points_value=points_charge,
                                   class_inst=discretized_obj,
                                   npoints=[1000, 1000],
                                   direction=direction_2d_slice, cmap='seismic',
                                   aspect='equal')

#
    func2(variable_2d)


#    plt.show()

    return  error, error_realt

def test_cyl_rect(**kwargs):
    ''' Not finished
    '''
    # build the mesh
    r1 = 3.0
    r1_2a = 2.90
    r1_2b = 3.10
    r2 = 8.0
    r2_3a = 7.95
    r2_3b = 8.05
    r3 = 10.0
    radius1 = [r1, r1]
    radius1_2a = [r1_2a, r1_2a]
    radius1_2b = [r1_2b, r1_2b]
    radius2 = [r2, r2]
    radius2_3a = [r2_3a,r2_3a]
    radius2_3b = [r2_3b,r2_3b]
    radius3 = [r3, r3]
    center = [0.0, 0.0]
    bstep = 0.5
    #    rstep = 0.8
    bbox = [-10.0, 10.0, -10.0, 10.0]
    #    rbox = [0.0, 11.0]

    cercle1 = shapes.Ellipsoid(radius1, center)
    cercle2 = (shapes.Ellipsoid(radius2, center)
               - shapes.Ellipsoid(radius1, center))
    cercle_1_2 = (shapes.Ellipsoid(radius1_2b, center)
                  - shapes.Ellipsoid(radius1_2a, center))
    cercle_2_3 = (shapes.Ellipsoid(radius2_3b, center)
                  - shapes.Ellipsoid(radius2_3a,center))
    cercle3 = (shapes.Ellipsoid(radius3, center)
               - shapes.Ellipsoid(radius2, center))

    fine_step = kwargs.get('fine_step', 0.005)
    mesh1 = (bbox, bstep, cercle1, 0)
    mesh_int1 = ([-4, 4, -4, 4], fine_step, cercle_1_2, 3)
    mesh2 = (bbox, bstep, cercle2, 1)
    mesh_int2 = ([-8.5, 8.5, -8.5, 8.5], fine_step, cercle_2_3, 4)
    mesh3 = (bbox, bstep, cercle3, 2)

    grid = GridBuilder(meshs=[mesh1, mesh2, mesh3, mesh_int1, mesh_int2],
                  holes=[], points=[])

    # Define the system
    poissonpro = ContinuousGeometry(space=(cercle1 + cercle2 + cercle3),
                                voltage=[cercle1, cercle3])

 ### Discretize your system and build the capacitance matrix
    discretized_obj = DiscretePoisson(
            poissonpro, mesh=finite_volume.Voronoi(grid=grid.points),
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    # Define boundary conditions
    dirichlet_val = [(cercle3, 2.0),
                     (cercle1, 0.0)]
    neuman_val = [shapes.PiecewiseFunction((cercle2, 0.0))]

    # build and solve linear system
    system_eq_obj = LinearProblem(discretized_obj, dirichlet_val,
                                 neuman_val,
                                 is_charge_density=True,
                                 build_equations=True,
                                 solve_problem=False)

    points_charge, points_voltage = system_eq_obj.solve()

    error, error_realt = plotting(
            points_charge, points_voltage, discretized_obj,
            grid, func=[cercle1, cercle2, cercle3], bbox=bbox,
            plot_mesh=False, direction_1d_cut=1, variable_1d=0.0,
            direction_2d_slice=2, variable_2d=0.0)

    return  error, error_realt

def test_cyl_circ(**kw):
    # build the mesh

    epislon = 1e-14
    r1 = 3.0 + epislon
    r1_2a = 2.95 + epislon
    r1_2b = 3.05 + epislon
    r2 = 8.0 + epislon
    r2_3a = 7.95 + epislon
    r2_3b = 8.05 + epislon
    r3 = 10.0 + epislon

    radius1 = [r1, r1]
    radius1a = [r1, r1]
    radius2 = [r2, r2]
    radius2a = [r2, r2]
    radius3 = [r3, r3]
    radius2_3a = [r2_3a,r2_3a]
    radius2_3b = [r2_3b,r2_3b]
    radius1_2a = [r1_2a, r1_2a]
    radius1_2b = [r1_2b, r1_2b]
    center = [0.0, 0.0]

    cercle1 = shapes.Ellipsoid(radius1, center)
    cercle2 = (shapes.Ellipsoid(radius2, center)
               - shapes.Ellipsoid(radius1a, center))
    cercle3 = (shapes.Ellipsoid(radius3, center)
               - shapes.Ellipsoid(radius2a, center))
    cercle_1_2 = (shapes.Ellipsoid(radius1_2b, center)
                  - shapes.Ellipsoid(radius1_2a, center))
    cercle_2_3 = (shapes.Ellipsoid(radius2_3b, center)
                  - shapes.Ellipsoid(radius2_3a,center))

    circularbox = [0, 13, 0, 2 * np.pi]
#    step = [0.5, np.pi /4]
#    step_2 = [0.01, np.pi/20]
    step = [0.5, np.pi /20]
    step_2 = [0.001, np.pi/20]

    # Just a fix for the moment
    grid = GridBuilder(build_mesh=False)

    # This is only temporary
    grid.add_mesh_circular(circularbox, step, cercle1, 0)
    grid.add_mesh_circular(circularbox, step, cercle2, 1)
    grid.add_mesh_circular(circularbox, step, cercle3, 2)
    grid.add_mesh_circular(circularbox, step_2, cercle_1_2, 3)
    grid.add_mesh_circular(circularbox, step_2, cercle_2_3, 4)

#    plt.figure()
#    voronoi_plot_2d(grid.points)
#    plt.plot()
#    points, t = repeated_points.remove_from_grid(grid.points,
#                                               decimals=5)

    # Define the system
    poissonpro = ContinuousGeometry(space=(cercle1 + cercle2 + cercle3),
                                voltage=[cercle1, cercle3])

 ### Discretize your system and build the capacitance matrix
    discretized_obj = DiscretePoisson(
            poissonpro, grid=grid.points,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    # Define boundary conditions
    dirichlet_val = [shapes.PiecewiseFunction((cercle3, 2.0)),
                     shapes.PiecewiseFunction((cercle1, 0.0))]
    neuman_val = [shapes.PiecewiseFunction((cercle2, 0.0))]

    # build and solve linear system
    system_eq_obj = LinearProblem(discretized_obj, dirichlet_val,
                                 neuman_val,
                                 is_charge_density=True,
                                 build_equations=True,
                                 solve_problem=False)

    points_charge, points_voltage = system_eq_obj.solve()

    plotting(points_charge, points_voltage, discretized_obj,
             grid, func=[cercle1, cercle2, cercle3], bbox=[-10,10,-10,10],
             plot_mesh=True, direction_1d_cut=1, variable_1d=0.0,
             direction_2d_slice=2, variable_2d=0.0)


def test_cyl_eps():
        # build the mesh

    epislon = 1e-14
    r1 = 3.0 + epislon
    r1_2a = 2.95 + epislon
    r1_2b = 3.05 + epislon
    r2 = 8.0 + epislon
    r2_3a = 7.95 + epislon
    r2_3b = 8.05 + epislon
    r3 = 10.0 + epislon

    radius1 = [r1, r1]
    radius1a = [r1, r1]
    radius2 = [r2, r2]
    radius2a = [r2, r2]
    radius3 = [r3, r3]
    radius2_3a = [r2_3a,r2_3a]
    radius2_3b = [r2_3b,r2_3b]
    radius1_2a = [r1_2a, r1_2a]
    radius1_2b = [r1_2b, r1_2b]
    center = [0.0, 0.0]

    cercle1 = shapes.Ellipsoid(radius1, center)
    cercle2 = (shapes.Ellipsoid(radius2, center)
               - shapes.Ellipsoid(radius1a, center))
    cercle3 = (shapes.Ellipsoid(radius3, center)
               - shapes.Ellipsoid(radius2a, center))
    cercle_1_2 = (shapes.Ellipsoid(radius1_2b, center)
                  - shapes.Ellipsoid(radius1_2a, center))
    cercle_2_3 = (shapes.Ellipsoid(radius2_3b, center)
                  - shapes.Ellipsoid(radius2_3a,center))

    circularbox = [0, 13, 0, 2 * np.pi]
#    step = [0.5, np.pi /4]
#    step_2 = [0.01, np.pi/20]
    step = [0.25, np.pi /20]
    step_2 = [0.1, np.pi/20]

    # Just a fix for the moment
    grid = GridBuilder(build_mesh=False)

    # This is only temporary
    grid.add_mesh_circular(circularbox, step, cercle1, 0)
    grid.add_mesh_circular(circularbox, step, cercle2, 1)
    grid.add_mesh_circular(circularbox, step, cercle3, 2)
    grid.add_mesh_circular(circularbox, step_2, cercle_1_2, 3)
    grid.add_mesh_circular(circularbox, step_2, cercle_2_3, 4)


#    points, t = repeated_points.remove_from_grid(grid.points,
#                                               decimals=5)

    r1_2b = 3.0 + epislon
    r2 = 5.0 + epislon
    r2_3a = 5.05 + epislon
    r2_3b = 8.00 + epislon

    cercle2_1 = (shapes.Ellipsoid([r2, r2], center)
                 - shapes.Ellipsoid([r1_2b, r1_2b], center))
    cercle2_2 = (shapes.Ellipsoid([r2_3b, r2_3b], center)
                 - shapes.Ellipsoid([r2_3a, r2_3a], center))
    dielectric = [shapes.PiecewiseFunction((cercle1, 2.5)),
                  shapes.PiecewiseFunction((cercle2_1, 5)),
                  shapes.PiecewiseFunction((cercle2_2, 10)),
                  shapes.PiecewiseFunction((cercle3, 2.5))]

    # Define the system
    poissonpro = ContinuousGeometry(space=(cercle1 + cercle2 + cercle3),
                                voltage=[cercle1, cercle3],
                                dielectric=dielectric)

    # Discretize your system and build the capacitance matrix
    discretized_obj = DiscretePoisson(
            poissonpro, grid=grid.points,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    # Define boundary conditions
    dirichlet_val = [shapes.PiecewiseFunction((cercle3, 2.0)),
                     shapes.PiecewiseFunction((cercle1, 0.0))]
    neuman_val = [shapes.PiecewiseFunction((cercle2, 0.0))]

    # build and solve linear system
    system_eq_obj = LinearProblem(discretized_obj, dirichlet_val,
                                 neuman_val,
                                 is_charge_density=True,
                                 build_equations=True,
                                 solve_problem=False)

    points_charge, points_voltage = system_eq_obj.solve()

    plotting(points_charge, points_voltage, discretized_obj,
             grid, func=[cercle1, cercle2, cercle3], bbox=[-10,10,-10,10],
             plot_mesh=False, direction_1d_cut=1, variable_1d=0.0,
             direction_2d_slice=2, variable_2d=0.0)


test_cyl_eps()
test_cyl_circ()
test_cyl_rect()

#fine_step = [0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.0025]
#list_error = list()
#list_error_relat = list()
#time_list = list()
#
#
#for i in fine_step:
#    ti =  time.time()
#    try:
#        erro_tmp, error_relat_temp = test_cyl_rect(fine_step=i)
#    except ValueError:
#        error_temp = []
#        error_relat_temp = []
#    t2 = time.time()
#    time_list.append(t2 - ti)
#    list_error.append(erro_tmp)
#    list_error_relat.append(error_relat_temp)




