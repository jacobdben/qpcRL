'''Parallel plate capacitance example
    from a continuous input.
    Version 2.0
    TODO : check when voronoi cells are not rectangular.
    TODO : Get the total surface of each cell properly. Plot properly.
'''

from scipy.spatial import voronoi_plot_2d, cKDTree
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
                coordinates, variable, direction, variable_2=None):

    ### Selecting coordinates and dirichlet/neuman points
    if variable_2 is not None:
        variables = (variable, variable_2)
    else:
        variables = (variable, )

    dimension = coordinates.shape[1]
    directions = [i for i in range(dimension)]
    del directions[direction]

    # If 3D variable_2 can't be None
    coordinates_1d, indices_1d = post_process.slice_1d_coordinates(
                coordinates=discretized_obj.mesh.points,
                directions = directions,
                variables = (variable, variable_2),
                decimals=5)

    mesh = discretized_obj.mesh

    pos_dirichlet = discretized_obj.points_system_voltage

    pos_dirichlet = pos_dirichlet.astype(int)
    pos_neuman = discretized_obj.points_system_charge


    points_dirichlet = np.intersect1d(indices_1d, pos_dirichlet)
    points_neuman = np.intersect1d(indices_1d, pos_neuman)

    # There should be only one max and min since we cut along one
    # direction and every redundant dirichlet point has alaready been removed.
    V_max = np.max(point_voltage[points_dirichlet])
    V_max_id = np.argwhere(point_voltage[points_dirichlet] == V_max)
    V_min = np.min(point_voltage[points_dirichlet])
    V_min_id = np.argwhere(point_voltage[points_dirichlet] == V_min)

    dist_min = mesh.points[points_dirichlet[V_min_id],
                                   direction]
    dist_max = mesh.points[points_dirichlet[V_max_id],
                                   direction]

    ### Finding the hypersurface (ridge hypersurface)
    point_surface = post_process.surface_between_points(
            points_r=points_dirichlet,
            points_l=points_neuman,
            cls_inst=discretized_obj)

    eta_iso = constants.epsilon_0
    charge_nor = constants.elementary_charge
    dim_nom = 1e-9

    # total charge
    point_charge = (point_charge
                    * discretized_obj.mesh.points_hypervolume)

    #charge_surface = points_out[points_dirichlet][pos_diriclet]/(eta_iso*surface)
    point_charge = (point_charge * charge_nor)

    total_charge=0
    relative_factor=0
    for key, points in discretized_obj.regions.voltage.tag_points.items():
        mask = np.isfinite(point_voltage[points])
        print('Region {} -> Total charge of : {}'.format(
                key, np.sum(point_charge[points[mask]])))
        total_charge += np.sum(point_charge[points[mask]])
        relative_factor += np.mean(np.abs(point_charge[points[mask]]))

    relative_total_charge = total_charge/relative_factor

    #Fit on voltage linenar regime
    coef = np.polyfit(mesh.points[points_neuman][:, 1],
                      point_voltage[points_neuman], deg=1)

    charge_from_slope =(coef[0]
                        * (np.abs(point_surface)*(eta_iso * dim_nom)))


    charge_theo = (np.abs(point_surface)
                   * (eta_iso * dim_nom)
                   * ((V_max - V_min)
                      / (dist_max - dist_min)))

    slope_theo = charge_theo[0] / ((eta_iso * dim_nom) * np.abs(point_surface))
    print(slope_theo)
    if np.allclose(slope_theo,slope_theo[0]) is False:
        print('Test Failed:')
        print('  The theoretical slope is not unique \n')
        raise
    slope_theo = slope_theo[0]

    plot_voltage = p_plt.plot_1d(
            directions=directions, class_inst=discretized_obj,
            points_value=point_voltage)

    plot_charge = p_plt.plot_1d(
            directions=directions, class_inst=discretized_obj,
            points_value=point_charge)

    fig2 = plt.figure()
    xlabel= ['x', 'y', 'z']
    ax_voltage = fig2.add_subplot(111)

    ax_voltage  = plot_voltage(variable=variables, ax=ax_voltage,
                               label='Voltage simulation',
                               marker='.', color='k', linestyle='None')

    ax_voltage.plot(mesh.points[points_neuman][:, 1],
             (slope_theo*mesh.points[points_neuman][:, 1]
              - slope_theo*dist_min[0][0]),
             'r-',
             label=r'Voltage analytical'.format(coef[1], coef[0]))

    ax_voltage.set_xlabel('y(nm)')
    ax_voltage.set_ylabel('Voltage(V)')
    ax_voltage.legend(loc='upper center')

    ax_charge = ax_voltage.twinx()

    ax_charge  = plot_charge(variable=variables, ax=ax_charge,
                               label='Charge simulation',
                               marker='.', color='b', linestyle='None')

    ax_charge.set_xlabel('y(nm)')
    ax_charge.set_ylabel('Charge(C)')
    ax_charge.legend(loc='lower center')

    Vd = np.max(point_voltage[points_dirichlet])
    Vg = np.min(point_voltage[points_dirichlet])


    if variable_2 is None:

        plt.title(r'Parallel Capacitance with $V_g= {{{0:.0f}}}$' \
                  'and $V_d = {{{1:.0f}}}$ along {3} = {2:.1f}'.format(
                      Vd, Vg, variable, xlabel[(direction + 1) % 2]))
    else:
        plt.title(r'Parallel Capacitance with $V_g= {{{0:.0f}}}$' \
                  'and $V_d = {{{1:.0f}}}$ along {3} = {2:.1f} and' \
                  ' {4} = {5:.1f}'.format(
                      Vd, Vg, variable, xlabel[directions[0]],
                      xlabel[directions[1]], variable_2))

    tol = 1e-5
    print('')
    print('----------------------------------------------------------------')
    if variable_2 is None:
        print(('Test parallalel plate capacitance ->'
               + ' Cut along {1} = {0:.5f}').format(
                variable, xlabel[(direction + 1) % 2]))
    else:
        print(('Test parallalel plate capacitance -> Cut along {1} = {0:.1f}'
               + ' and {2} = {3:.1f}').format(
                          variable, xlabel[directions[0]],
                          xlabel[directions[1]], variable_2))
    print('')
    print('\n  Total charge (in Coulombs): \n \n  {0} \n '.format(
            np.sum(total_charge)))
    print('\n  Relative total charge: \n \n  {0} \n '.format(
            relative_total_charge))
    print('\n Slope of the voltage variation from fit: \n \n {0} \n '.format(
            charge_from_slope))
    print(('\n  Numerically calculated total charge'))
    print(point_charge[points_dirichlet])
    print(('\n  Analytically calculated surface charge '
          + ' : \n  \n {0} \n '.format(charge_theo)))

    if (np.allclose(np.abs(charge_from_slope)/np.abs(point_charge[points_dirichlet]),
                    1,
                    atol=tol)
        and np.isclose(relative_total_charge, 0.0, atol=tol)
        and np.allclose(np.abs(charge_theo[0])/np.abs(point_charge[points_dirichlet]),
                        1, atol=tol)):

        print(('\n Result: Passed with normalized charge tolerance'
              +' of \n \n {0} \n').format(
                tol))
    else:

        print('\n Result: Not Passed \n')


    print('\n Error of : \n \n {0} \n'.format(np.abs(charge_theo) -
                                  np.abs(point_charge[points_dirichlet])))
    print('\n Relative error of : \n \n {0} \n'.format(
            ((np.abs(charge_theo)
             - np.abs(point_charge[points_dirichlet]))
             / np.abs(charge_theo))))
    print(('\n This migth depend on how the distance between the two '
           + 'electrodes were calculated and also the distance between the '
           +' dirichlet points and their nearest neuman points \n'))
    print('')

    plt.show()


def plotting(discretized_obj, grid, points_charge, points_voltage,
             func, bbox, plot_mesh=False, plot_density=False,
             variable_2d_slice=0, variable_1d_cut=1.5, variable_1d_cut_2=None,
             direction_2d_slice=2, direction_1d_slice=1):

##     Plot voronoi cell , Neuman and Dirichlet regions
    if plot_mesh:
        fig = plt.figure()

        ax = fig.add_subplot(111)
        voronoi_plot_2d(discretized_obj.mesh, ax=ax)
        x, y = list(zip(*grid.points))
        ax.scatter(x, y, c=grid.point_label, cmap=cm.jet)
        # remove non-calculated points before plot
        t = np.concatenate((discretized_obj.points_system_charge,
                            discretized_obj.points_system_voltage))
        t = t.astype(int)

        xx = discretized_obj.mesh.points[t, 0]
        yy = discretized_obj.mesh.points[t, 1]
        ax.scatter(xx, yy, c='r', marker='+')
        plt.show()

#    # plot charge density everywhere
    if plot_density:

        fig = plt.figure()

        ax = fig.add_subplot(111)
        voronoi_plot_2d(discretized_obj.mesh, ax=ax)
        x, y = list(zip(*grid.points))
        ax.scatter(x, y, c=grid.point_label, cmap=cm.jet)
        points_system = discretized_obj.points_system

        xx = discretized_obj.mesh.points[points_system, 0]
        yy = discretized_obj.mesh.points[points_system, 1]
        ax.scatter(xx, yy, c=points_charge[points_system],
                   marker='o', s=500)
        plt.show()

    # Plot result and compare it with analytical calculation
    plot_points(points_charge, points_voltage, discretized_obj,
                coordinates=discretized_obj.mesh.points, variable=variable_1d_cut,
                direction=direction_1d_slice, variable_2=variable_1d_cut_2)

    func2 = p_plt.plot_geometry_2d(figsize=(10, 10), bbox=bbox,
                                   fctnames=['elc1', 'isol', 'elec2'],
                                   functions=[func[0], func[1], func[2]],
                                   npoints=[100, 14],
                                   xlabel='x(nm)', ylabel='y(nm)',
                                   direction=direction_2d_slice,
                                   cmap=ListedColormap(['w',
                                  'orange', 'blue', 'red']), vmin=-2, vmax=2,
                                   interpolation='none', aspect='equal')
    func2(variable_2d_slice)

    func2 = p_plt.plot_values_2d(figsize=(10, 10),
                                 colorbar_label='Voltage (V)',
                                 bbox=bbox, xlabel='x(nm)', ylabel='y(nm)',
                                 points_value=points_voltage,
                                 class_inst=discretized_obj,
                                 npoints=[500, 500],
                                 direction=direction_2d_slice,
                                 cmap='seismic',
                                 aspect='equal')

    func2(variable_2d_slice)

    points_charge = points_charge * discretized_obj.charge_normalization
    points_charge = points_charge / (discretized_obj.distance_normalization**2)
    func2 = p_plt.plot_values_2d(figsize=(10, 10),
                                 colorbar_label=r'Charge density($C.m^{{{-2}}}$)',
                                 bbox=bbox, xlabel='x(nm)', ylabel='y(nm)',
                                 points_value=points_charge,
                                 class_inst=discretized_obj,
                                 npoints=[500, 500],
                                 direction=direction_2d_slice,
                                 cmap='seismic',
                                 aspect='equal')

#
    func2(variable_2d_slice)


    plt.show()


def test_capa_2d(**kw):
    '''
        Parallel plate capacitance test in a 2D mesh.
    '''
### Build the grid

    length_1 = [5, 1]
    length_2 = [5, 7]
    corner_2 = [0, 1.1]
    corner_3 = [0, 8.2]
    length_3 = [5, 2]
    rect1 = shapes.Rectangle(length=length_1)
    rect2 = shapes.Rectangle(length=length_2, corner=corner_2)
    rect3 = shapes.Rectangle(length=length_3, corner=corner_3)
    bbox = [0, 5, 0, 9]

    bstep=0.25
    grid1 = (bbox, bstep, rect1, 0)
    grid2 = (bbox, bstep, rect2, 5)
    grid3 = (bbox, bstep, rect3, 0)

    grid = GridBuilder(meshs=[grid1, grid3, grid2], holes=[], points=[])


### Define the continuous system

    poissonpro = ContinuousGeometry(space=(rect1 + rect2 + rect3),
                                   voltage=[rect1, rect3])

 ### Discretize your system and build the capacitance matrix
    discretized_obj = DiscretePoisson(
            poissonpro, grid=grid,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    # Define boundary conditions
    dirichlet_val = [(rect3, 20.0),
                     (rect1, 0.0)]

    # build and solve linear system
    system_eq_obj = LinearProblem(discretized_obj, dirichlet_val,
                                 is_charge_density=True,
                                 build_equations=True,
                                 solve_problem=False)

    points_charge, points_voltage = system_eq_obj.solve()

    plotting(discretized_obj, grid, points_charge, points_voltage,
             func=[rect1, rect2, rect3], bbox=bbox,
             variable_2d_slice=0, variable_1d_cut=1.5,
             direction_2d_slice=2, direction_1d_slice=1)

def test_capa_3d(**kw):
    '''
        Parallel plate capacitance test in a 3D mesh.
    '''
    # build the mesh
    length_1 = [5, 3, 5]
    length_2 = [5, 10, 5]
    corner_1 = [0, 3.1, 0]
    corner_2 = [0, 13.1, 0]
    length_3 = [5, 3, 5]
    rect1 = shapes.Rectangle(length=length_1)
    rect2 = shapes.Rectangle(length=length_2, corner=corner_1)
    rect3 = shapes.Rectangle(length=length_3, corner=corner_2)
    bbox = [0, 5, 0, 17, 0, 5]

    mesh1 = (bbox, 0.25, rect1, 0.5)
    mesh2 = (bbox, 0.25, rect2, 1)
    mesh3 = (bbox, 0.25, rect3, 0.5)
    grid = GridBuilder(meshs=[mesh1, mesh2, mesh3], holes=[], points=[])

    p_plt.points_3D_mavi(grid)

### Define the continuous system
    poissonpro = ContinuousGeometry(space=(rect1 + rect2 + rect3),
                                   voltage=[rect1, rect3])

 ### Discretize your system and build the capacitance matrix
    discretized_obj = DiscretePoisson(
            poissonpro, grid=grid,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    # Define boundary conditions
    dirichlet_val = [(rect3, 20.0),
                     (rect1, 0.0)]

    # build and solve linear system
    system_eq_obj = LinearProblem(discretized_obj, dirichlet_val,
                                 is_charge_density=True,
                                 build_equations=True,
                                 solve_problem=False)

    points_charge, points_voltage = system_eq_obj.solve()
    plotting(discretized_obj, grid, points_charge, points_voltage,
             func=[rect1, rect2, rect3], bbox=bbox[:4],
             variable_2d_slice=1.0, variable_1d_cut=1.5,
             variable_1d_cut_2=1.5,
             direction_2d_slice=2, direction_1d_slice=1)

def test_capa_1d():
    '''
            Parallel plate capacitance test in a 1D mesh.
    '''
    eta_vide = constants.epsilon_0
    charge = constants.elementary_charge
    dist_nor = 1e-9

    xx = np.sort(np.linspace(-10, 10, 25))
    xx = xx[:, None]

    space = lambda x: (-10 <= x[:,0]) * (x[:,0] <= 10)
    dirichlet_1 = lambda x: (-10 <= x[:,0]) * (x[:,0] <= -5)
    dirichlet_2 = lambda x: (5 <= x[:,0]) * (x[:,0] <= 10)
    neuman = lambda x: (-5 < x[:, 0]) * (x[:, 0] < 5)
    dielectric = lambda x: np.ones(x.shape[0], dtype=bool)

### Define the continuous system

    poissonpro = ContinuousGeometry(space=space,
                                   voltage=[dirichlet_1, dirichlet_2],
                                   dielectric=[(dielectric, 1)],
                                   charge=[neuman])

 ### Discretize your system and build the capacitance matrix
    discretized_obj = DiscretePoisson(
            poissonpro, grid=xx,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    dirichlet_val = [(dirichlet_1, 0),
                     (dirichlet_2, 5)]
    neuman_val = [(neuman, 0.0)]

     # build and solve linear system
    system_eq_obj = LinearProblem(discretized_obj, dirichlet_val,
                                 charge_val=neuman_val,
                                 is_charge_density=True,
                                 build_equations=True,
                                 solve_problem=False)

    points_charge, points_voltage = system_eq_obj.solve()

     # Solving it analytically from the input data
    dist_diff = np.diff(
            discretized_obj.mesh.points[discretized_obj.points_system_voltage],
            axis=0)[:, 0]

    voltage_diff = np.diff(points_voltage[discretized_obj.points_system_voltage],
                           axis=0)

    charge_teo = (((voltage_diff/dist_diff) * (eta_vide * dist_nor))
                  / (charge))

    # Compare with simulation
    points_charge = (points_charge * discretized_obj.mesh.points_hypervolume)
    if np.allclose(
            np.abs(points_charge[discretized_obj.points_system_voltage])/charge_teo,
            1):

        print('\n 1D parallel capacitance test passed')
    else:

        print('1D paralle capacitance test not passed')

    print(' \n Error of : {0}'.format(
            np.abs(np.abs(points_charge[discretized_obj.points_system_voltage])
            - charge_teo)))
    print('\n Relative error: {0} \n '.format(
            np.abs(np.abs(points_charge[discretized_obj.points_system_voltage])
            - charge_teo)/charge_teo))

test_capa_1d()
test_capa_2d()
test_capa_3d()