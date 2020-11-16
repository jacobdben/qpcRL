'''
    Example using a MOS structure
'''
import os

from matplotlib import pyplot as plt

from poisson.continuous import shapes
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)

### Geomtry

# make bridge gate
bridge = shapes.Rectangle([20, 4, 2], corner=[-10, 0, 0])
bridge_bbox = [-20, 20, 0, 4, 0, 3]

# insulator1
insulator = shapes.Rectangle([20,40,2], corner=[-10,-20,-2.1])
insulator_box = [-20,20,-20,20,-2,0]

# make Graphene
plane_1 = shapes.Rectangle([20, 20, 2], corner=[-10, -20, -4.4])
plane_2 = shapes.Rectangle([20, 20, 2], corner=[-10, 0, -4.4])
hole1 = shapes.Rectangle([5, 14, 2], corner=[-10.1, -5, -4.4])
hole2 = shapes.Rectangle([5, 14, 2], corner=[5.1, -5, -4.4])
hole3 = shapes.Rectangle([10, 14, 2], corner=[-5, -5, -4.4])

graphene_elc1 = plane_1 - (hole1 + hole2 + hole3)
graphene_elc2 = plane_2 - (hole1 + hole2 + hole3)

graphene_bbox = [-10, 10, -20, 20, -4.5, -2]

# GRID
grid = GridBuilder()
grid.add_mesh_square(bridge_bbox, 1, bridge, 1.5)
grid.add_mesh_square(graphene_bbox, 1, graphene_elc1, 1)
grid.add_mesh_square(graphene_bbox, 1, graphene_elc2, 1)
grid.add_mesh_square(insulator_box, 0.25, insulator, 0.5)
grid.add_mesh_square(graphene_bbox, 1, hole1, 0.5)
grid.add_mesh_square(graphene_bbox, 1, hole2, 0.5)
grid.add_mesh_square(graphene_bbox, 0.1, hole3, 0.7)

# Geometry
die_insulator = 12
die_bridge = 3
die_graphene = 1.5
geometry = ContinuousGeometry(space = bridge + insulator + plane_1 + plane_2,
                              voltage={'bridge':bridge, 'Elec_1':graphene_elc1,
                                       'Elec_2':graphene_elc2},
                              dielectric=[(insulator, 12),
                                          (bridge, die_bridge),
                                          (hole3, die_graphene)])

bbox = [-20,20, -20,20]
plot_geometry = geometry.plot(direction=2, bbox=bbox, plot_type='2D')
plot_geometry(variable=0)
plot_geometry(variable=-2)
plot_geometry(variable=4)
plot_geometry(variable=-4.5)
geometry.plot(points=grid.points, plot_type='3D', scale_factor = 0.2)
plt.show()

# Discrete system
discrete_poi = DiscretePoisson(
        geometry, grid=grid,
        selection = {'Neuman-Dirichlet':[['voltage', '*']]})

# Build the linear problem and solve it
linear_prob_inst = LinearProblem(discrete_poi,
                                 voltage_val = [(bridge, 10.0),
                                                (graphene_elc1, 5.0),
                                                (graphene_elc2, -5.0)],
                                 charge_val = [(hole3, 10e7)],
                                 is_charge_density=True)

# Save data to vtk file
#current = '/'.join((os.path.dirname(os.path.abspath(__file__)),
#                                    'example_sphere3D'))
#linear_prob_inst.save_to_vtk(filename=current)

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

