'''
    Simple example in 3D.
    Sphere
'''
import numpy as np
import matplotlib.pyplot as plt
import os

from poisson import plot as p_plt
from poisson.continuous import shapes
from poisson.tools import post_process
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)


def sphere_3D():

####  Build the grid of poihnts. Finer mesh at interface.
    epislon = 1e-14
    r1 = 3.0 + epislon
    r1_2a = 2.90 + epislon
    r1_2b = 3.1 + epislon
    r2 = 8.0 + epislon
    r2_3a = 7.90 + epislon
    r2_3b = 8.1 + epislon
    r3 = 10.0 + epislon

    center = [0.0, 0.0, 0.0]

    cercle1 = shapes.Ellipsoid((r1, r1, r1), center)
    cercle2 = (shapes.Ellipsoid((r2, r2, r2), center)
               - shapes.Ellipsoid((r1, r1, r1), center))
    cercle3 = (shapes.Ellipsoid((r3, r3, r3), center)
               - shapes.Ellipsoid((r2, r2, r2), center))

    cercle_1_2 = (shapes.Ellipsoid((r1_2b, r1_2b, r1_2b), center)
                  - shapes.Ellipsoid((r1_2a, r1_2a, r1_2a), center))
    cercle_2_3 = (shapes.Ellipsoid((r2_3b, r2_3b, r2_3b), center)
                  - shapes.Ellipsoid((r2_3a, r2_3a, r2_3a),center))

    sphericalbox = [-10, 10, 0, 2 * np.pi, 0, np.pi]
    step = [0.1, np.pi / 20, np.pi / 20]
    step_2 = [0.01, np.pi/ 20, np.pi/ 20]

    grid = GridBuilder(build_mesh=False)

    grid.add_mesh_spherical(sphericalbox, step, cercle1, 0)
    grid.add_mesh_spherical(sphericalbox, step, cercle2, 1)
    grid.add_mesh_spherical(sphericalbox, step, cercle3, 2)
    grid.add_mesh_spherical(sphericalbox, step_2, cercle_1_2, 3)
    grid.add_mesh_spherical(sphericalbox, step_2, cercle_2_3, 4)

    # Plot the grid
    p_plt.points_3D_mavi(cls_inst=grid, scale_factor = 0.05)
    plt.show()

#### Construct the poisson problem and solve it
    geometry = ContinuousGeometry(space=(cercle1 + cercle2 + cercle3),
                                voltage=[cercle1, cercle3])

    # Visualize the geometry of a 2D system
    bbox = [-10, 10, -10, 10]
    plot_geometry = geometry.plot(direction=2, bbox=bbox, plot_type='2D')
    plot_geometry(variable=0)
    geometry.plot(points=grid.points, plot_type='3D', scale_factor = 0.05)
    plt.show()
    # Or
    plot_geometry = p_plt.plot_continuous_geometry(geometry_inst=geometry,
                                                   bbox=bbox)
    plot_geometry(variable=0)
    p_plt.points_3D_mavi(cls_inst=geometry, points=grid.points,
                         scale_factor = 0.05)
    plt.show()

    # Discretize the continuous geometry using a voronoi finite volume mesh
    sys_instance = DiscretePoisson(
            geometry, grid=grid,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    # Build the A and B matrix (A = xB) and solve for x
    linear_prob_inst = LinearProblem(sys_instance, is_charge_density=True,
                                  voltage_val=[(cercle3, 4.0),
                                              (cercle1, 0.0)])

    # Save data to vtk file
    current = '/'.join((os.path.dirname(os.path.abspath(__file__)),
                                        'example_sphere3D'))
    linear_prob_inst.save_to_vtk(filename=current)

#### Plotting
        # Plot 2D cut
    plot_volt, plot_charge = linear_prob_inst.plot_cut_2d(direction=2,
                                                          npoints=(1000,1000))
    plot_volt(variable=0, colorbar_label='Voltage (V)')
    plot_charge(variable=0, colorbar_label='Charge')
    plt.show()

    plot_voltage, plot_charge = linear_prob_inst.plot_cut_1d(
            directions=(1, 2), bbox='default', npoints=2000)

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

sphere_3D()