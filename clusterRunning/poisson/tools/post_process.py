'''
    Functions that can be used to extract voltage and charge
        values from a given coordinate or input points.
    TODO: rename slice_1d_coordinates and slice_coordinates
'''

# charge_from_all and voltage_from_all are
# use internally in LinearProblem.
__all__ = ['value_from_coordinates',
           'slice_coordinates',
           'surface_between_points',
           'regions_summary_charge',
           'slice_1d_coordinates']


from scipy.spatial import cKDTree

import numpy as np

from . import _post_process as ptl
from poisson.discrete import FiniteVolume, DiscretePoisson, LinearProblem


def slice_1d_coordinates(coordinates, directions, variables, decimals=5):
    '''
        From  a set of coordinates (numpy.array) find a cut along one direction
            and recover its coordinates.

        Parameters:
        -----------
        coordinates: numpy.ndarray of 2 or 3 dimensions,
        directions: tuple or list contaning one or two int, e.g. (1, 2)
            The int can take values of 0, 1, 2.
            They correspond to the axis (columns) in coordinates where
            the componets will remain the same for all points,
            e.g for a cut along 1  one writes directions=(0, 2)
        variables: tuple or list containing one or two floats
            Where the cut will be made, the values for
            the components of coordinates that will remain constant,
            e.g. if i want to make a cut with y = 10.0 and z = 15.0
            variables=(10.0, 15.0)
        decimals: c.f. description in slice_coordinates. Default is 5.
    '''

    dimension = np.asarray(coordinates).shape[1]

    # Creat a function allowing us to select the set of coordinates
    # where one of their components is the same.
    slice_coord_1 = slice_coordinates(
                coordinates=coordinates,
                decimals=decimals,
                direction=directions[0],
                return_indices=True)

    if dimension == 3:
        # Coordinates where one among the three components are the same.
        coordinates_2d, indices_2d = slice_coord_1(variables[0])

        # Idem as slice_coord_1
        slice_coord_2 = slice_coordinates(
            coordinates=coordinates_2d,
            decimals=decimals,
            direction=directions[1],
            return_indices=True)

        # Coordinates where two among the three components are the same.
        coordinates_1d, indices_1d = slice_coord_2(variables[1])
        indices_1d = indices_2d[indices_1d]

    if dimension == 2:
        coordinates_1d, indices_1d = slice_coord_1(variables[0])

    return coordinates_1d, indices_1d


def sub_region_summary_charge(discrete_poisson_inst, region_ndt, points_charge,
                              verbose=True):
    '''
        Reads the charge in each sub_region.

        Parameters:

            discrete_poisson_inst : poisson.DiscretePoisson instance

            points_charge: numpy array

            region_ndt : namedtuple
                region_ndt = discrete_poisson_inst.regions.charge is the namedtuple
                for the charge region.
    '''
    hyper_volume = list()
    charges = list()
    charge_tot = list()

    for key, points in region_ndt.tag_points.items():

        mask = np.isfinite(points_charge[points])
        charge_tot_temp = np.sum(points_charge[points[mask]])
        hyper_volume.append((
                key, discrete_poisson_inst.mesh.points_hypervolume[
                        points[mask]]))
        charge_tot.append((key, charge_tot_temp))
        charges.append((key, points_charge[points[mask]]))

        if verbose:
            print('\n Sub-region {} : \n'.format(key))
            print('    Total charge of: {}'.format(
                    charge_tot_temp))
            print('    Total Hypervolume: {}'.format(
                    np.sum(discrete_poisson_inst.mesh.points_hypervolume[
                            points[mask]])))
            print('    Number of points: {}'.format(
                    len(discrete_poisson_inst.mesh.points_hypervolume[
                            points[mask]])))

    return hyper_volume, charges, charge_tot


def regions_summary_charge(lin_prob_inst, verbose=True):
    '''
        Returns the charge in each sub-region as well as the hypervolume.

        Parameters:
        -----------

            lin_prob_inst : poisson.LinearProblem instance
            verbose: boolean

        Returns:
        --------
            hyper_volume: Dict containing the hypervolume for all
                volume elemnets within each sub-region.
            charges: Idem as hyper_volume but containing the charge
            charges_tot: Dict containing the total charge in each sub-region

    '''
    if isinstance(lin_prob_inst, LinearProblem):
        discrete_poisson_inst = lin_prob_inst.discrete_poisson
        points_charge = lin_prob_inst.points_charge
    else:
        TypeError('lin_prob_inst must be an instance of poisson.LinearProblem')

    hyper_volume = dict()
    charges = dict()
    charges_tot = dict()
    regions_names = [discrete_poisson_inst.regions_names_dict[key] for key in [
             'dirichlet', 'neuman', 'mixed']]

    if verbose:
        print('#'*50)
    for region_name in regions_names:
        if verbose:
            print('Region: {} ##################################### \n'.format(
                region_name))

        surf, charg, charg_tot = sub_region_summary_charge(
                discrete_poisson_inst,
                region_ndt=getattr(discrete_poisson_inst.regions, region_name),
                verbose=True,
                points_charge=points_charge)

        hyper_volume.update({region_name: surf})
        charges.update({region_name: charg})
        charges_tot.update({region_name: charg_tot})

        if verbose:
            print('')
    if verbose:
        print('#'*50)

    return hyper_volume, charges, charges_tot

def surface_between_points(points_r, points_l, cls_inst):
    '''
        TODO: Doc
        obs: points_r and points_l does not need to be
        given in any specific order.
    '''
    if isinstance(cls_inst, DiscretePoisson):
        mesh = cls_inst.mesh
    elif isinstance(cls_inst, FiniteVolume):
        mesh = cls_inst

    point_surface = np.zeros(len(points_r))
    for t, point_r in enumerate(points_r):
        surfaces = np.zeros(len(mesh.points_firstneig[point_r]))
        for i, neig in enumerate(mesh.points_firstneig[point_r]):
            if np.any(points_l == neig):
                ridge_di = mesh.point_ridges[point_r]
                mask = [pos  for pos, ridge in enumerate(ridge_di)
                        if bool(np.any(mesh.ridge_points[ridge] == neig))]
                surfaces[i] = mesh.ridge_hypersurface[ridge_di[mask[0]]]
        point_surface[t] = np.sum(surfaces)

    return point_surface

def slice_coordinates(coordinates, decimals=3, direction=1,
                      return_indices=False):
    '''
        TODO: DOC
    '''

    rounded_coord = coordinates.round(decimals=decimals)
    arg_sorted = rounded_coord[:, direction].argsort()
    uniqval, uniqcount = np.unique(rounded_coord[arg_sorted, direction],
                                   return_counts=True)


    # Creat indptr vector marking the end and beggining of each uniqval
    indtpr = np.concatenate(([0], np.cumsum(uniqcount)))
    indtpr = np.array([indtpr[0:-1], indtpr[1:]]).T

    def f(variable):
        pos = np.arange(0, len(uniqval))[uniqval == round(variable, decimals)]

        coordinates_1d =  coordinates[
                arg_sorted[indtpr[pos][0][0]:indtpr[pos][0][1]]]
        if return_indices:
            indices = arg_sorted[indtpr[pos][0][0]:indtpr[pos][0][1]]

            return coordinates_1d, indices
        else:
            return coordinates_1d
    return f


def value_from_coordinates(points_value, class_inst, deg_interpolation=0,
                           index_points=None):
    '''
        Given a set of inputs, outputs and coordinates uses the information
        on class_inst class to find the charge points or voltages
        in each point in coordinates.
        Uses the function get_value_from_points to do so.

        Parameters:
            points_value: np.ndarray of 1D. Returned by solver. Either
                points_charges or points_voltages
            class_inst: instace of a poisson.DiscretePoisson
                or
                        instance of a poisson.FiniteVolume
            deg_interpolation: int
                e.g.  0 for no interpolation. It indicates the degree of
                interpolation.
            index_of_points: np.ndarray with the index of the points to be used
                among points_value.
        Returns:
            function that takes as input a set of 2D or 3D coordinates(
            np.ndarray) and returns values for the voltage or charges
            in each point in the coordinates (as an np.ndarray).

    '''

    if isinstance(class_inst, DiscretePoisson):
        mesh = class_inst.mesh
    elif isinstance(class_inst, FiniteVolume):
        mesh = class_inst
    else:
        raise TypeError('class_inst must be of type DiscretePoisson \
                        or FiniteVolume')

    if index_points is None:
        index_points = np.arange(mesh.npoints)
        first_neig = mesh.points_firstneig
    else:
        index_points = np.asanyarray(index_points)
        first_neig = mesh.fneigs(index_points)

    tree = cKDTree(mesh.points[index_points])

    def f(coordinates):
        coord_val = ptl.value_from_points(x=coordinates, discret_val=points_value,
                                          mapping=index_points,
                                          firstneig = first_neig, tree=tree,
                                          vector=True,
                                          interpolate=deg_interpolation)
        return np.ones(len(coord_val), dtype=bool), coord_val
    return f

def charge_from_all(linear_problem_inst, discrete_poisson_inst):
    '''
        Given a set of inputs and outputs it uses the information
        on DiscretePoisson class to find the charge and charge points
        in each point in the mesh.points.

        Parameters:
            linear_problem_inst: poisson.LinearProblem instance
            discrete_poisson_inst: poisson.DiscretePoisson instance

        Returns:
            points_charge: values for the charge in each point in the mesh.
                Zero for points where the voronoi cell is either open
                or belonging to a dirichlet region not in the surface

    '''

    points_output = linear_problem_inst.points_output
    points_input = linear_problem_inst.points_input

    if points_output is None:
        raise ValueError(( 'linear_problem_inst.points_output is None.'
                          + ' The system has not yet been solved. '))

    pos_dirichlet = discrete_poisson_inst.points_system_voltage.astype(int)
    pos_neuman = discrete_poisson_inst.points_system_charge
    pos_voltage_mixed = linear_problem_inst.pos_voltage_mixed.astype(int)
    pos_charge_mixed = linear_problem_inst.pos_charge_mixed.astype(int)

    # NAN if the point is not calculated/ defined
    points_charge = np.zeros(discrete_poisson_inst.mesh.npoints,
                             dtype=np.float) * np.NAN

    points_charge[pos_dirichlet] = points_output[pos_dirichlet]
    points_charge[pos_neuman] = points_input[pos_neuman]

    if len(discrete_poisson_inst.points_system_mixed) > 0:

        points_charge[pos_charge_mixed] = points_input[pos_charge_mixed]
        points_charge[pos_voltage_mixed] = points_output[pos_voltage_mixed]

    return points_charge

def voltage_from_all(linear_problem_inst, discrete_poisson_inst):
    '''
        Given a set of inputs and outputs it uses the information
        on DiscretePoisson class to find the charge and voltage points
        in each point in the mesh.

        Parameters:
            linear_problem_inst: poisson.LinearProblem instance
            discrete_poisson_inst: poisson.DiscretePoisson instance

        Returns:
            points_voltage: values for the charge in each point in the mesh.
                Zero for points where the voronoi cell is either open
                or belonging to a dirichlet region not in the surface

    '''

    points_output = linear_problem_inst.points_output
    points_input = linear_problem_inst.points_input

    if points_output is None:
        raise ValueError(( 'linear_problem_inst.points_output is None.'
                          + ' The system has not yet been solved. '))

    pos_dirichlet = discrete_poisson_inst.points_system_voltage.astype(int)
    pos_neuman = discrete_poisson_inst.points_system_charge
    pos_voltage_mixed = linear_problem_inst.pos_voltage_mixed.astype(int)
    pos_charge_mixed = linear_problem_inst.pos_charge_mixed.astype(int)

    # NAN if the point is not calculated/ defined
    points_voltage = np.zeros(discrete_poisson_inst.mesh.npoints,
                             dtype=np.float) * np.NAN
    points_voltage[pos_neuman] = points_output[pos_neuman]
    points_voltage[pos_dirichlet] = points_input[pos_dirichlet]

    if len(discrete_poisson_inst.points_system_mixed) > 0:
        points_voltage[pos_charge_mixed] = points_output[pos_charge_mixed]
        points_voltage[pos_voltage_mixed] = points_input[pos_voltage_mixed]

    return points_voltage