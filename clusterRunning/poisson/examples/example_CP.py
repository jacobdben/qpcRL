'''
    Example with a trivial geometry and minimal code usage
'''
import matplotlib.pyplot as plt
import os

from poisson import plot as p_plt
from poisson.continuous import shapes
from poisson.tools import post_process
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)


def parallel_plate_2D():
    '''
        2D exapmle of parallel plate capacitance
    '''
    # Define your grid of points.( not a mesh)
    rect1 = shapes.Rectangle(length=(5, 1))
    rect2 = shapes.Rectangle(length=(5, 7), corner=(0, 1.01))
    rect3 = shapes.Rectangle(length=(5, 2), corner=(0, 8.02))

    bbox = [0, 5, 0, 9]
    bstep=0.5

    # Mesher name will change
    grid = GridBuilder(meshs=[(bbox, bstep, rect1, 0),
                         (bbox, bstep, rect2, 5),
                         (bbox, bstep, rect3, 0)])
    # In this example we only need to define the voltage regions, since
    # the rest is by default a charge region. It is important to remark here
    # that the total space must be given, otherwise the by default charge regions
    # cannot be known. If a charge and voltage regions are defined
    # one does not need to define a space.
    # By defining the region_names manually the user changes the region names
    # from voltage -> electrodes, charge->insulator and mixed->mixed. This is
    # of course not nescessary.
    geometry = ContinuousGeometry(
            space=(rect1 + rect2 + rect3), voltage=[rect1, rect3],
            dielectric=[shapes.PiecewiseFunction((rect1 + rect3, 2.5)),
                        (rect2, 2.5)])
    geometry = ContinuousGeometry(
            space=(rect1 + rect2 + rect3), voltage=[rect1, rect3],
            default_relative_permittivity=12)

    # Visualize the geometry of a 2D system
    geometry.plot(bbox=bbox, plot_type='2D')
    # Or
    # p_plt.plot_continuous_geometry(geometry_inst=geometry, bbox=bbox)
    plt.show()

    # PoissonProblem is used to defined the "continuous" geometry.
    # DiscretizedPoisson will apply it to a given mesh.
    discretized_obj = DiscretePoisson(geometry,
                                      grid=grid,
                                      discretize=False,
                                      construct_capacintace_matrix=False)

    # Selection - Apply Neuman-Dirichlet function to all sub-regions
    # within the voltage region. If selection is not given, no function is
    # applied. We notice here that since the first eleemnt in
    # region names is electrodes, the voltage regions are now called
    # electrodes, so whenever the latter is called, the user defined name
    # must be given. The steps are done when initializing System.
    discretized_obj.discretize(selection={'Neuman-Dirichlet':[['voltage', '*']]})
    discretized_obj.construct_capacitance_mat()

    # Build the linear system of equations and solve it.
    # If not value is given to charge_val then the default value of 0 is taken
    linear_prob_inst = LinearProblem(
            discretized_obj, is_charge_density=True,
            voltage_val = [(rect1, 20.0),
                           (rect3, 0.0)],)

    # Save data into vtk format ((.vtu))
    current = '/'.join((os.path.dirname(os.path.abspath(__file__)),
                                        'example_CP'))
    linear_prob_inst.save_to_vtk(filename=current)

#####Plotting
    # Plot 2D cut
    plot_volt, plot_charge = linear_prob_inst.plot_cut_2d()
    plot_volt(colorbar_label='Voltage(V)')
    plot_charge(colorbar_label='Charge')
    plt.show()
    # Or p_plt.plot_linear_problem_2d(linear_problem_inst=linear_prob_inst)

    # Plot 1D cut
    # If bbox not specified -> plot the value at the center of the volume
    # cell.
    plot_voltage, plot_charge = linear_prob_inst.plot_cut_1d(directions=[0])
    # If bbox specified -> generates a list of points and find which volume
    # cell they belong to. Take the value at the center of that cell as their
    # voltage and charge value.
    plot_voltage_bb, plot_charge_bb = linear_prob_inst.plot_cut_1d(
            directions=[0], bbox='default')

    fig2 = plt.figure()
    ax_voltage = fig2.add_subplot(111)

    t, data_volt = plot_voltage(
            variables= [2], ax = ax_voltage, marker='x',
            label='Voltage simulation - center', linestyle='None', color='r',)

    t, data_volt = plot_voltage_bb(
            variables= [2], ax = ax_voltage, marker='.',
            label='Voltage simulation - bbox', linestyle='None', color='k',)

    ax_voltage.set_xlabel('{0}(nm)'.format('y'))
    ax_voltage.set_ylabel('Voltage(V)')

    ax_charge = ax_voltage.twinx()

    t, data_charge = plot_charge(
            variables = [2], ax = ax_charge, marker='x',
            label='Charge simulation - center', linestyle='None',  color='m',)

    t, data_charge = plot_charge_bb(
            variables = [2], ax = ax_charge, marker='.',
            label='Charge simulation - bbox', linestyle='None',  color='b',)

    ax_charge.set_xlabel('{0}(nm)'.format('y'))
    ax_charge.set_ylabel(r'Charge density $(\#.nm^{{{-2}}})$')
    ax_charge.legend(loc='lower center')

    plt.show()
    plt.show()

parallel_plate_2D()