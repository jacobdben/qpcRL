import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from poisson import plot as p_plt
from poisson.continuous import shapes
from poisson.tools import post_process
from poisson import (DiscretePoisson, GridBuilder, ContinuousGeometry,
                     LinearProblem)

def make_QPC():

    ### gates
    gatez = 5  # lower position of the gate in z (on top of gas)
    gateh = 1  # heigth of the gates
    gateextw = 15  # width of the gate at side opposed to gas
    gateintw = 5  # width of the gate at side facing the gas
    gatedepth = 6   # x extension of the gate
    gatexpos = 3   # x position of the interior of the gate

    ### gas
    gasz = 0  # gas position in z
    gasx = 10   # gas extension along x
    gasy = 30  # gas extension along y

    ### refinement
    alphab = 1
    alphag = 0.25
    alphagas = 0.25

    ### external box
    extbox = np.array(
            [-(2 * gatexpos + gatedepth), (2 * gatexpos + gatedepth),
             -.75 * gasy, .75 * gasy, -10 * alphag, 2 * (gateh + gatez)])

    ### gates
    gatel = shapes.Delaunay(
            [[-(gatexpos + gatedepth) , -gateextw/2, gatez],
              [-(gatexpos + gatedepth) , gateextw/2, gatez],
              [-(gatexpos) , -gateintw/2, gatez],
              [-(gatexpos) , gateintw/2, gatez],
              [-(gatexpos + gatedepth) , -gateextw/2, gatez + gateh],
              [-(gatexpos + gatedepth) , gateextw/2, gatez + gateh],
              [-(gatexpos) , -gateintw/2, gatez + gateh],
              [-(gatexpos) , gateintw/2, gatez + gateh]])
    gatel_mesh = [extbox, alphag, gatel, 1]

    gater = copy.deepcopy(gatel)
    gater.rotate([0, 0, np.pi], center=[0, 0, 0])
    gater_mesh = [extbox, alphag, gater, 2]

    ### gas
    gas = shapes.Delaunay(
            [[-gasx/2, -gasy/2, gasz + 4*alphag],
             [gasx/2, -gasy/2, gasz + 4*alphag],
             [-gasx/2, gasy/2, gasz + 4*alphag],
             [gasx/2, gasy/2, gasz + 4*alphag],
             [-gasx/2, -gasy/2, gasz - 4*alphag],
             [gasx/2, -gasy/2, gasz - 4*alphag],
             [-gasx/2, gasy/2, gasz - 4*alphag],
             [gasx/2, gasy/2, gasz - 4*alphag]])
    gas_mesh = [extbox, alphagas, gas, 3]

    ### exterior box
    extbbox = shapes.Delaunay(
            [extbox[[0, 2, 4]], extbox[[0, 2, 5]],
             extbox[[0, 3, 4]], extbox[[0, 3, 5]],
             extbox[[1, 2, 4]], extbox[[1, 2, 5]],
             extbox[[1, 3, 4]], extbox[[1, 3, 5]]])
    ext_mesh = [extbox, alphab, extbbox, 0]

    ### mesh
    return (GridBuilder(meshs=[ext_mesh, gatel_mesh, gater_mesh, gas_mesh]),
            gatel, gater, gas, extbbox)


def QPC():

    grid, gatel, gater, gas, extbbox = make_QPC()

    dielectric = (extbbox - (gatel + gater + gas))
    geo_inst = ContinuousGeometry(
            space=(gatel + gater + gas + dielectric),
            voltage = {'gater':gater,
                       'gatel':gatel})

    geo_inst.plot(direction=2, bbox=extbbox.get_bbox()[:4], plot_type='2D')(6)
    geo_inst.plot(points=grid.points, plot_type='3D', scale_factor=0.1)

    sys_inst = DiscretePoisson(geo_inst, grid=grid,
                             selection={'Neuman-Dirichlet':[['voltage', '*']]})

    print(len(sys_inst.points_system))
    print(len(sys_inst.mesh.points))

    sys_eq_inst = LinearProblem(sys_inst,
                                voltage_val=[(gatel, 5), (gater, 5)],
                                charge_val=[(gas, 0.01)])

    current = '/'.join((os.path.dirname(os.path.abspath(__file__)),
                                        'example_QPC'))

    sys_eq_inst.plot_3d('both', ('Voltage (V)', 'Charge'),
                        scale_factor=(0.05, 3))

    sys_eq_inst.save_to_vtk(filename=current)

    plot_voltage, plot_charge = sys_eq_inst.plot_cut_1d(
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

    plot_voltage_2d, plot_charge_2d = sys_eq_inst.plot_cut_2d(direction=2)

    plot_voltage_2d(0, colorbar_label=r'Voltage (V)')
    plot_charge_2d(0, colorbar_label=r'Charge density($\#.nm^{{{-2}}}$)')

    plot_voltage_2d(5, colorbar_label=r'Voltage (V)')
    plot_charge_2d(5, colorbar_label=r'Charge density($\#.nm^{{{-2}}}$)')

    plt.show()


QPC()
