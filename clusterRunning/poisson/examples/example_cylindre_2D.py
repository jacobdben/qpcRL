'''
    Example of a 2D cylinder
'''
import matplotlib.pyplot as plt
import os

from poisson import plot as p_plt
from poisson.continuous import shapes
from poisson.tools import post_process
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)

def cylinder_2D():

    # Define the grid points
    center = (0.0, 0.0)
    fine_step = 0.005
    bstep = 0.25
    bbox = [-10.0, 10.0, -10.0, 10.0]

    cercle1 = shapes.Ellipsoid((3.0, 3.0), center)
    cercle2 = (shapes.Ellipsoid((8.0, 8.0), center)
               - shapes.Ellipsoid((3.0, 3.0), center))
    cercle3 = (shapes.Ellipsoid((10.0, 10.0), center)
               - shapes.Ellipsoid((8.0, 8.0), center))

    # So the mesh will be finer at the interface between
    # the electrodes and the insulator region.
    cercle_1_2 = (shapes.Ellipsoid((3.10, 3.10), center)
                  - shapes.Ellipsoid((2.90, 2.90), center))
    cercle_2_3 = (shapes.Ellipsoid((8.05,8.05), center)
                  - shapes.Ellipsoid((7.95,7.95),center))

    # Construct the set of points that are at the center of the
    # volume ement in the finite volume mesh.
    grid = GridBuilder(
            meshs=[(bbox, bstep, cercle1, 0),
                   ([-4, 4, -4, 4], fine_step, cercle_1_2, 3),
                   (bbox, bstep, cercle2, 1),
                   ([-8.5, 8.5, -8.5, 8.5], fine_step, cercle_2_3, 4),
                   (bbox, bstep, cercle3, 2)])

    # Default value for the dielectric constant 8.85e-12
    geometry = ContinuousGeometry(
            space=(cercle1 + cercle2 + cercle3), voltage=[cercle1, cercle3])

    # Visualize the geometry of a 2D system
    geometry.plot(bbox=bbox, plot_type='2D')
    # Or
    # p_plt.plot_continuous_geometry(geometry_inst=geometry, bbox=bbox)
    plt.show()

    discretized_obj = DiscretePoisson(
            geometry, grid=grid,
            selection={'Neuman-Dirichlet':[['voltage', '*']]})

    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=True,
            voltage_val = [(cercle3, 20.0), (cercle1, 0.0)])

    # Save data into vtk format ((.vtu))
    current = '/'.join((os.path.dirname(os.path.abspath(__file__)),
                                        'example_cylindre'))
    linear_prob_inst.save_to_vtk(filename=current)

####    #Plotting
    # Plot 2D cut
    plot_volt, plot_charge = linear_prob_inst.plot_cut_2d()
    plot_volt(colorbar_label='Voltage(V)')
    plot_charge(colorbar_label='Charge')
    plt.show()
    # Or p_plt.plot_linear_problem_2d(linear_problem_inst=linear_prob_inst)

    # Plot 1D cut
    # If bbox not specified -> plot the value at the center of the volume
    # cell.
    plot_voltage, plot_charge = linear_prob_inst.plot_cut_1d(directions=[1])
    # If bbox specified -> generates a list of points and find which volume
    # cell they belong to. Take the value at the center of that cell as their
    # voltage and charge value.
    plot_voltage_bb, plot_charge_bb = linear_prob_inst.plot_cut_1d(
            directions=[1], bbox='default')

    fig2 = plt.figure()
    ax_voltage = fig2.add_subplot(111)

    t, data_volt = plot_voltage(
            variables= [0], ax = ax_voltage, marker='x',
            label='Voltage simulation - center', linestyle='None', color='r',)

    t, data_volt = plot_voltage_bb(
            variables= [0], ax = ax_voltage, marker='.',
            label='Voltage simulation - bbox', linestyle='None', color='k',)

    ax_voltage.set_xlabel('{0}(nm)'.format('y'))
    ax_voltage.set_ylabel('Voltage(V)')

    ax_charge = ax_voltage.twinx()

    t, data_charge = plot_charge(
            variables = [0], ax = ax_charge, marker='x',
            label='Charge simulation - center', linestyle='None',  color='m',)

    t, data_charge = plot_charge_bb(
            variables = [0], ax = ax_charge, marker='.',
            label='Charge simulation - bbox', linestyle='None',  color='b',)

    ax_charge.set_xlabel('{0}(nm)'.format('y'))
    ax_charge.set_ylabel(r'Charge density $(\#.nm^{{{-2}}})$')
    ax_charge.legend(loc='lower center')

    plt.show()

cylinder_2D()