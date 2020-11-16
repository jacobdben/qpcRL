'''
   TODO : Add proper description
   TODO : Add verification that the function is vectorizable
   TODO : Add test to neuman_dirichlet_selection/ voronoi_selection and
           delaunay_selection. I think this has been already done.
   TODO : Use eval vect of mesher to read the region functions
   TODO : set the dielectric function of the dirichelt point
   to the same value as its neighbour
   TODO: check for convergence with different dielectric constants.
   TODO: test different methods

'''


__all__ = ['DiscretePoisson',
           'keep_in_fneigs']


from collections import namedtuple
import itertools
from operator import itemgetter
#import warnings

from scipy.spatial import qhull
import scipy.sparse as sps
from scipy import constants

import numpy as np

import matplotlib.pyplot as plt

from . import _finite_volume as vscython
from . import _discrete_poisson as dscython
from . import finite_volume


def keep_in_fneigs(index_of_points_to_keep, points_firstneig, npoints):
    '''
        Wrapper around cython function.
        Makes a copy of self.points_firstneig and eliminates
            any point not belonging to index_of_points

        npoints number of points in the original mesh.

        index_of_point = np.ndarray[np.int_t, ndim=1]

        Cython version - 5x faster for large numbers (> 50K)
            for smaller number either faster (2x) or same order of mag.

        pure python version:

            points_in = np.zeros(npoints,
                                 dtype=int)
            points_in[index_of_points] = np.ones(len(index_of_points),
                                                 dtype=int)

            index_of_points_fneigs = []
            for point in range(len(self.points_firstneig)):
                index_of_points_fneigs.append([])
                for fneig in self.points_firstneig[point]:
                    if points_in[fneig]:
                        index_of_points_fneigs[point].append(fneig)

        TODO: add test ?
    '''

    new_points_fneigs = vscython.fneig(
            index_of_points_to_keep,
            points_firstneig,
            npoints)

    return new_points_fneigs


def discrete_value_from_func(functions, grid, default=0):
    ''' Recover the value of a function at each point in the mesh
        ----------------------------------------------------------
        Parameters:
            functions: dicitonary with as a key the tag of the region
                and as the value the function associted to that region
            grid: array
                mesh that will be applied to functions
            default: float
                default value
        ---------------------------------------------------------
        Returns :
            point_value: array
                point_value[î] contains the value of mesh[î]
            point_tag:
                point_tag[i] contains the tag of mesh[î]
    '''

    point_value = np.ones(grid.shape[0]) * default
    point_tag = np.zeros(grid.shape[0], dtype=object)

    functions_info = [[region_func(grid), region_name]
                      for region_name, region_func in functions.items()]

    for function_info in functions_info:

        mask = np.arange(len(grid))[function_info[0][0]]

        if isinstance(function_info[0][1], int):

            point_value[mask] = (point_value[mask]
                                 * function_info[0][1][mask])

        else:
            point_value[mask] = point_value[mask] * function_info[0][1][mask]

        point_tag[mask] = (np.ones(len(mask), dtype=object)
                           * function_info[1])

    return point_value, point_tag


def assemble_points_system(region_ndtuple, surface_points, region_name,
                           sub_region_names, **kwargs):
    '''
        Assemble points system in function of the type of
        finite volume mesh. If the latter is composed of vornoi cells,
        the closed voronoi regions must be provided in kwargs.
    '''

    try:
        closed_region_points = kwargs['closed_regions_point']
    except KeyError:
        raise AttributeError(('\n If Voronoi finite volue mesh the '
                              + ' closed voronoi regions must have been '
                              + ' selected.\n '))

    if closed_region_points[region_name]:
        region_ndtuple.index_of_closed_points = np.hstack(
            closed_region_points[region_name])
    else:
        region_ndtuple.index_of_closed_points = np.array([], dtype=int)

    surface_point_region = []
    for sub_region_name in sub_region_names[region_name]:
        if sub_region_name in surface_points.keys():
            surface_point_region.append(surface_points[sub_region_name])

    if len(surface_point_region) > 0:

        surface_point_region = np.hstack(surface_point_region)

        region_ndtuple.index_of_system_points = np.intersect1d(
                surface_point_region,
                region_ndtuple.index_of_closed_points)
    else:
        region_ndtuple.index_of_system_points = \
            region_ndtuple.index_of_closed_points

    return region_ndtuple

def create_regions_nmtp(regions_point, regions_properties,
                        region_specific_properties, surface_points,
                        mesh_points, sub_region_names, **kwargs):
    '''
        TODO: Add description
        In kwargs add specific information depending on how the
        mesh was created
    '''

    point_tags = np.zeros(mesh_points.shape[0], dtype=object)
    region_ndtuple = {}

    # FIll in the data for charge and voltage functions
    for region_name, region_point_dict in regions_point.items():

        regions_properties_current = ( regions_properties +
                                      region_specific_properties[region_name])

        region_ndtuple[region_name] = namedtuple(
                region_name, regions_properties_current)

        dict_tag = {}
        for sub_region_name in sub_region_names[region_name]:

            point_tags[region_point_dict[sub_region_name]] = (
                    np.ones(len(region_point_dict[sub_region_name]),
                            dtype=object)
                    * sub_region_name)

            dict_tag[sub_region_name] = np.array(
                    region_point_dict[sub_region_name])

        region_ndtuple[region_name].tag_points = dict_tag

        # In case no function is assigned index_of_mesh_points is np.array([])
        if region_point_dict:
            region_ndtuple[region_name].index_of_mesh_points = np.hstack(
                list(region_point_dict.values()))
        else:
            region_ndtuple[region_name].index_of_mesh_points = np.array(
                [], dtype=int)

        # Depends on the type of mesh used.
        # There is no real need to return anything as reguin_ndtuple is
        # modified in define_points_system without making a deep copy
        # but for clarty it migth be iteresting ?
        region_ndtuple[region_name] = assemble_points_system(
                region_ndtuple=region_ndtuple[region_name],
                surface_points=surface_points,
                region_name=region_name,
                sub_region_names=sub_region_names,
                **kwargs)

        if len(region_ndtuple[region_name].index_of_system_points) == 0:
            region_ndtuple[region_name].index_of_system_points = \
                region_ndtuple[region_name].index_of_mesh_points

    return region_ndtuple, point_tags

def neuman_dirichlet_selection(dirichlet_points,
                               neuman_points, points_firstneig, **kwargs):
    ''' Remove any dirichlet point not
        connected to a neuman point

        Parameters:
            dirichlet_points: list.
                List containing arrays containing points
                    belonging to dirichlet regions (voltage regions)

            neuman_points: array.
                numpy array containing the neuman points in the mesh.

            points_firstneig: list of arrays.
                Each element contains the first neig of each point in the mesh

            **kwargs there so its compatible with delaunay_selection and voronoi
                selection

        Returns:
            dirichlet_points_surface: list of same size as dirichlet_points.
                Each element contains the points in its corresponding
                    dirichlet_points element that are next to a point in
                    neuman_points.

    '''

    points_neuman = neuman_points
    if isinstance(dirichlet_points, list) is False:
        dirichlet_points = [dirichlet_points]

    dirichlet_poits_surface = [[] for i in range(len(dirichlet_points))]

    for pos, points_dirichlet in enumerate(dirichlet_points):

        # Select the elements in points_firstneig belonging
        # to points_dirichlet
        points_fneigs_temp = [points_firstneig[pdirichlet]
                              for pdirichlet in points_dirichlet]

        # For every element in points_fneigs_tmp
        # keep only points belonging to points_neuman in
        list_keep = keep_in_fneigs(
                            points_neuman,
                            points_fneigs_temp,
                            len(points_firstneig))

        # Find the non empty elements in list_keep.
        # Their position correspond to the points_dirichlet next to a
        # points_neuman.
        mapping = dscython.find_non_empty(
                list_keep,
                index_to_test=np.arange(len(points_dirichlet), dtype=int))

        dirichlet_poits_surface[pos] = points_dirichlet[mapping]


    print('Done Neuman - Dirichlet refinement')
    return dirichlet_poits_surface

def delaunay_selection(convex_voltage_funcs, mesh_points, **kwargs):
    ''' Keep only the surfaces of dirichlet regions.
    This is done by making a delaunay triangulation and
    keeping only the simpleces at the boundary, the latter being
    detected by a -1 in the neighbour list.
    **kwargs there so its compatible with neuman_dirichlet_selection and voronoi
        selection
    '''

    try:
        points_index = [np.arange(len(mesh_points))[conv_fun(mesh_points)]
                    for conv_fun in convex_voltage_funcs]
    except TypeError:
        points_index = [np.arange(len(mesh_points))[
                convex_voltage_funcs(mesh_points)]]

    surface_points = []
#    not_points = []
    for points_id in points_index:

        points_id = np.asanyarray(points_id)
        points = mesh_points[points_id]

        delaunay = qhull.Delaunay(points)

        external_simpleces = np.array([
            pos for pos, del_pos in enumerate(delaunay.neighbors)
            if any(del_pos == -1)])

        external_points = points_id[np.unique(delaunay.simplices[
            external_simpleces])]

        surface_points.append(external_points)

    print('Done Delaunay refinement')
    return surface_points

def voronoi_selection(convex_voltage_funcs, mesh_points, delaunay=False,
                      **kwargs):
    ''' Keep only the surfaces of dirichlet regions.
    This is done by creating a voronoi diagram and deleting any point in the
    mesh not belonging to a open voronoi region
    **kwargs there so its compatible with neuman_dirichlet_selection and
        delaunay_selection
    '''
    try:
        points_index = [np.arange(len(mesh_points))[conv_fun(mesh_points)]
                    for conv_fun in convex_voltage_funcs]
    except TypeError:
        points_index = [np.arange(len(mesh_points))[
                convex_voltage_funcs(mesh_points)]]

    surface_points = []
    for points_id in points_index:
        points_id = np.asanyarray(points_id)
        points = mesh_points[points_id]

        # c.f. docstring in deulaunay_refinement
        if delaunay:
            delaunay_obj = qhull.Delaunay(points)

            external_simpleces = np.array([
                pos for pos, del_pos in enumerate(delaunay_obj.neighbors)
                if any(del_pos == -1)])

            external_points_delaunay = np.unique(delaunay_obj.simplices[
                external_simpleces])
            points = points[external_points_delaunay]
        # If the latter is not enough one can either create a voronoi diagram
        # from the mesh_points[points_id] and then take the points at
        # the open voronoi regions as the ones at the surface.
        # Another option is to make a voronoi diagram from
        # mesh_points[external_points_delaunay] and from that take the points
        # that are associated to open voronoi regions.
        # In the latter the voronoi region will be created for a potentially
        # much smaller number of points. Nontheless, delaunay triangulation
        # will be performed twice : first to obtain external_points_delaunay
        # and a second time to further filter the mesh_points points.
        voronoi = qhull.Voronoi(points, qhull_options=b"Qbb Qc Qx Qz")

        open_voronoi = np.array([pos
                                 for pos, del_pos in enumerate(voronoi.regions)
                                 if any(np.array(del_pos) == -1)])

        # to avoid having to use an np.where to find which voronoi region
        # is associated with wich point in points
        region_point = np.ones(len(voronoi.regions), dtype=int)
        for i, region in enumerate(voronoi.point_region):
            region_point[region] = i

        external_points = points_id[region_point[open_voronoi]]

        if delaunay:
            external_points = points_id[external_points_delaunay[
                    region_point[open_voronoi]]]

        surface_points.append(external_points)

    print('Done Voronoi refinement')
    return surface_points


def prepare_capacitance_sparse(points_firstneig, dielectric_val,
                               points_ridges, ridges_capacitance):

    number_elements = len(list(itertools.chain.from_iterable(
            points_firstneig))) + len(points_firstneig)
    column = np.zeros(number_elements, dtype = int)
    row = np.zeros(number_elements, dtype = int)
    data = np.zeros(number_elements, dtype=float)

    element_id = 0
    for point, fneigs in enumerate(points_firstneig):
        diag_data = 0
        # Avoid empty fneigs (due to Qz otpion in qhull) and open regions
        if bool(fneigs):
            for fneig in fneigs:

                point_ridges = points_ridges[point]
                fneig_ridges = points_ridges[fneig]
                ridge = fneig_ridges[[index for index, element in
                                      enumerate(fneig_ridges)
                                      if element in point_ridges][0]]

                ridge = fneig_ridges[[index for index, element in
                                      enumerate(fneig_ridges)
                                      if element in point_ridges][0]]

                column[element_id] = fneig
                row[element_id] = point

                dielectric_factor = (2 * (dielectric_val[point]
                                         * dielectric_val[fneig])
                                     / (dielectric_val[point]
                                        + dielectric_val[fneig]))

                data[element_id] = -1 * (ridges_capacitance[ridge]
                                         * dielectric_factor)

                # Diagonal elements don't have the minus sign.
                diag_data = (diag_data + (-1 * data[element_id]))

                element_id +=1

            column[element_id] = point
            row[element_id] = point
            data[element_id] = diag_data

            element_id +=1

    #       TODO: Understand why  we would need to remove extra zeros
    data = np.delete(data, np.arange(element_id, number_elements))
    column = np.delete(column, np.arange(element_id, number_elements))
    row = np.delete(row, np.arange(element_id, number_elements))

    return data, column, row


class DiscretePoisson():
    '''
        Generates instance containing the capacitance matrix
        associated with a continuous geometry and the relevant
        mesh information.

            version 0.1 -> 0.2 : Added discretization step and
                capacitance matrix construction in __init__ by default. Removed
                build_fv_mesh module, the mesh is now given as input when
                initialized.
            version 0.2 -> 0.3: Instead of taking as parameter a
                finite_volume.FiniteVolume mesh, takes either a numpy array
                containing a set of coordinates or a grid.GridBuilder instance.
                The possibility of giving a mesh still remains. Otherwise
                the mesh is constructed internally.
    '''

    version ='0.3'
    print(('DiscretePoisson version ' + version))

    def __init__(self, geometry, grid=None, discretization_method='Voronoi',
                 mesh=None, test_geometry=True, discretize=True,
                 selection=None, construct_capacintace_matrix=True):
        '''

            Must define, at least, either:
                1 - geometry, grid
                2 - geometry, mesh

            Parameters:
            -----------

            geometry: poisson.ContinuousGeometry instance

            grid: numpy array or callable object returning a numpy array.
                The numpy array must contain the coordinates for the points
                in the system.
            discretization_method: string.
                Method used to generate the mesh.
                Accepted inputs: 'Voronoi'
                Default: 'Voronoi'

            mesh: poisson.discrete.finite_volume.Voronoi() instance.

            selection: dict()
                The key indicates which function is going to be used.
                Each element is a list containing other lists.
                    Each list within the previous list has as a fisrt
                    element the tag given to the region to which the
                    function will be applied to and as a second element
                    the concerned sub_regions.
                example :
                        selection = {'Neuman-Dirichlet':[
                                ['voltage', ['voltage_0', 'voltage_1']],
                                ['charge', ['charge_0']]]}

                If one writes ['voltage','*'] then all
                    sub_regions within voltage are called
                Default: None

            test_geometry: boolean.
                Verify if the regions defined by the geometry parameter do
                not intersect for a given grid.
                Default: True

            discretize: boolean
                If the discretization of the geometry will be done
                when the class is initialized.
                Default: True
            construct_capacitance_matrix: boolean
                If the construction of the capacitance matrix will be done
                when the class is initialized.
                Default: True
        '''

        self.grid = grid
        if callable(grid):
            self.points = self.grid() #Retuns a np.array of points when called
        elif isinstance(grid, np.ndarray):
            self.points = self.grid

        self.discretization_methods = {'Voronoi': finite_volume.Voronoi}
        if grid is not None:
            self.mesh = self.discretization_methods[discretization_method](
                    grid=self.points)
        elif mesh is not None:
            self.mesh = mesh
        else:
            raise ValueError('Neither mesh nor grid are defined')

        self.geometry = geometry
        self.regions_functions = geometry.regions_functions
        self.dielectric_functions = geometry.dielectric_functions
        self.regions_names_dict = geometry.regions_names_dict
        self.sub_region_names = geometry.sub_region_names


        # Verify that the functions are chosen correctly, i.e. :
        #   That the intersection of convex_voltage_funcs, other_voltage_funcs
        #       and charge_funcs is empty.
        #   That dielectric_funcs union with the latter three lists of functions
        #   is dielectric_funcs
        #   That every point in the mesh is has a dielectric and charge/votlage
        #   value.
        if test_geometry:
            (self.regions_intersection,
            self.sub_regions_intersection) = geometry.check_functions(
                    self.mesh.points)
        else:
            self.regions_intersection = None
            self.sub_regions_intersection = None

        # contains all the information of each region including points index
        # tags and values (for dielectric only)
        self.regions = None
        self.dielectric = None

        self.ridges_capacitance = np.array([])
        self.points_capacitance_sparse = []
        self.points_dielectric = np.array([])
        self.points_system_firstneig = None
        self.points_system = None
        self.points_system_charge = None
        self.points_system_voltage = None
        self.points_system_mixed = None

        self.capa_normalization = 1

        self.selection = selection
        if discretize:
            self.discretize(selection=self.selection)
            if construct_capacintace_matrix:
                self.construct_capacitance_mat()

    def __validate_functions(self):

        if ((self.sub_regions_intersection is not None)
            or (self.regions_intersection is not None)):
            print('\n')
            print('-'*48)
            print('Sub_regions_intersection')
            print('\n')
            print(self.sub_regions_intersection)
            print('-'*48)
            print('regions_intersection')
            print('\n')
            print(self.regions_intersection)
            print('-'*48)
            print('\n')
            raise ValueError((' \n \n Functions are not well defined.'
                              + '\n There is one or more points being'
                              + ' defined simultaneously by the same function'
                              + ' \n Use check_functions module in'
                              + ' geometryblem class to see which ones.\n'))


    def construct_capacitance_mat(
            self, charge_normalization=constants.elementary_charge,
            distance_normalization=1e-9,
            voltage_normalization = 1):
        '''
            Constructs the capacitance matrix.

            Parameters:
            -----------
                charge_normalization: float.
                    Default: scipy.constants.elementary_charge
                distance_normalization: float
                    Default: 1e--9
                voltage_normalization: float
                    Default: 1
        '''

        # check function inputs:
        self.__validate_functions()

        dielectric_name = self.regions_names_dict['dielectric']

        self.charge_normalization=charge_normalization
        self.distance_normalization=distance_normalization
        self.voltage_normalization=voltage_normalization

        points_ridges = self.mesh.point_ridges

        ridges_hypersurf = self.mesh.ridge_hypersurface
        print('Done calculating surface')

        ridgepoints_dist = self.mesh.ridgepoints_dist
        print('Done calculating distance')

        self.ridges_capacitance = np.true_divide(
            ridges_hypersurf, ridgepoints_dist)

        dielectric_region = getattr(self.regions, dielectric_name)

        assert (self.mesh.points.shape[0]
                == len(dielectric_region.point_value)), (
                        'There are some points wehere the dielectric'
                        + ' constant is undefined')

        dielectric_val = dielectric_region.point_value[:, None]

## Create sparse matrix. - There is probably a better way of initializing it
#       # number of elements in each column/row
        # change this in order to dela with onlt the closed
        # regions which will have non zero values
        self.points_system = np.concatenate(
                [getattr(self.regions, region_name).index_of_system_points
                 for region_name in self.regions_functions]).astype(int)

        self.points_system_firstneig = self.mesh.fneigs(
                self.points_system)

        data, column, row = prepare_capacitance_sparse(
                self.points_system_firstneig, dielectric_val,
                points_ridges, self.ridges_capacitance)

        # Normalization  -> c.f. report
        data = (data * self.distance_normalization)
        data = ((data * self.voltage_normalization)
                / self.charge_normalization)

        self.points_capacitance_sparse = sps.csc_matrix((data, (row, column)),
                                                   shape=(self.mesh.npoints,
                                                          self.mesh.npoints))

        print('Done calculating capacitance matrix')


    def discretize(self, selection=None,
                   default_dielectric=constants.epsilon_0):
        ''' Takens the initial input ( functions defining the geometry
        and dielectric constants) and tags each point in the mesh and
        associates to it a dielectric constant.
        --------------------------------------------
        Parameters:
            selection = dictionary, c.f. redundant_point_selection for
                more information or __init__()
                Default: The one given when initializing a
                    DiscretePoisson instace, which has the default value
                    of None.
            default_dielectric = float
                Default: scipy.constants.epsilon_0
                Every point in the mesh is set to the default dielectric
                unless the value is specified in Geometry.dielectric_functions,
                in which case the value of the dielectric constant is
                default_dielectric * val. Val is the value specified
                by Geometry.dielectric_functions.
        --------------------------------------------
        The discretized system is given by self.regions :
            ----------------------------------------
        self.regions: namedtuple
            ----------------------------------------
            Attributes:
                voltage: nametuple
                charge: -//-
                mixed = -//-
                dielectric = -//-
                point_tags = np.array giving the tag (string) associated to
                    each point in the grid (self.mesh.points)
        ---------------------------------------------
        Concerning voltage, charge and mixed : nametuple
            -----------------------------------------
            Attributes:
                index_of_mesh_points: np.array - index of points associated with
                    a dirichlet region (for voltage) or a neuman region
                    (for charge) or a mixed region (neuman or dirichlet).
                index_of_system_points : np.array - idem as above but refering
                    only to points passed down to the solver and which
                    will be used to solve the system.
                index_of_closed_points: idem as above but only referring to
                    points associated to closed regions (no vertex at infinity)
                resized_index_of_points: idem as above but after removing
                    certain dirichlet points ( see neuman_dirichlet_selection)
                    for more information. obs: this will probably be removed.
                tag_points: dictionary - key work is the tag and
                    the corresponding value are the points associated with
                    that tag.
        ---------------------------------------------
        Concerning dieletric: namedtuple
            -----------------------------------------
            Attributes:
                point_value: gives the dielectric constant associated with
                    each point in self.mesh.points
                point_tag: gives the tag associated with each point
        ----------------------------------------------
        Concerning the tags :
            For example, lets say there is three functions defining
            the voltage regions. The tag associated with the points for
            which the first function returns True is 'voltage_0'; for those
            returning True for the second function - 'voltage_1', ...
            Same thing for charge (charge_0 ...), mixed(mixed_0 ...)
            and dielectric (dielectric_0 ...)    negative_neuman = points_neuman[

        ----------------------------------------------
        In order to make this function independent of self.regions_names_dict
        or self.regions_functions it uses self.regions_names_dict indices in
        order to select voltage, charge and mixed_BC regions.

        For clarity, self.regions_names_dict is a dictionary with the following
        keys:
                    'dirichlet' -> voltage (by default)
                    'neuman' -> charge
                    'mixed' -> mixed
                    'dielectric' -> dielectric

        In any case it is worth verifying how these properties are
        constructed in  geometry.Geometry.

        Moreover, self.region_names contains the name for the
        voltage, charge and mixed regions.
        Self.region_functions does not contain any information about the
        dielectric functions.

        **********************************************************************
        TODO: Update
        '''

        # check function inputs:
        self.__validate_functions()

        (voltage_name, charge_name, mixed_name,
         dielectric_name) = [self.regions_names_dict[key] for key in [
             'dirichlet', 'neuman', 'mixed', 'dielectric']]

        self.regions = namedtuple(
                'Regions', [*list(self.regions_names_dict.values()), 'point_tags'])

        # function must be vectorizable
        # points for each region with. For now ignore dielectric functions
        # in self.regions_functions there are not dielectric functions
        regions_point = {region_name:
            {sub_region_name:
                np.arange(len(self.mesh.points))[
                        sub_region_function(self.mesh.points)]
                for sub_region_name, sub_region_function in region_func.items()}
            for region_name, region_func in self.regions_functions.items()}

        # find the closed points for each region (dirichlet, neuman, mixed)
        # in case fv_mesh is made by finding voronoi cells

        closed_regions_point = self._get_closed_points()
        region_specific_properties = {}
        for key in self.regions_functions:
            if key is mixed_name:
                region_specific_properties.update({key : []})
            else:
                region_specific_properties.update(
                        {key : ['index_of_closed_points']})

        # There migth be some redundant data being saved
        # Create namedtuple to contain all the data
        # Needs to be in the same order as region_funcs
        regions_properties = ['index_of_mesh_points','tag_points',
                              'index_of_system_points']

        # Find the indices not corresponding to the surface points
        # of convex_dirichlet and for other_dirichlet
        if selection is None:
            selection = self.selection
        if selection is not None:
            print(selection)
            chosen_points = self.search_points(
                    selection=selection, regions_point=regions_point)
        else:
            chosen_points = {}
        # Create the named tuples
        (region_ndtuple,
         point_tags) = create_regions_nmtp(
                 regions_point, regions_properties,
                 region_specific_properties,
                 surface_points=chosen_points,
                 mesh_points=self.mesh.points,
                 sub_region_names=self.sub_region_names,
                 closed_regions_point=closed_regions_point)

        for key in self.regions_functions:
            setattr(self.regions, key, region_ndtuple[key])

        pdie_val, pdie_tag = discrete_value_from_func(
                self.dielectric_functions,
                self.mesh.points,
                default=default_dielectric)

        dielectric_nmd = namedtuple(dielectric_name,
                                     ['point_value', 'point_tag'])
        dielectric_nmd.point_value = pdie_val
        dielectric_nmd.point_tag = pdie_tag
        setattr(self.regions, dielectric_name, dielectric_nmd)

        self.regions.point_tags = point_tags

        # For easy of acces, although rather redundant
        self.points_system_voltage = getattr(
                self.regions, voltage_name).index_of_system_points

        self.points_system_charge = getattr(
                self.regions, charge_name).index_of_system_points
        self.points_system_mixed = getattr(
                self.regions, mixed_name).index_of_system_points

        self.points_system = np.concatenate(
                [getattr(self.regions, region_name).index_of_system_points
                 for region_name in self.regions_functions]).astype(int)

    def _get_closed_points(self):
        '''
            Get the index of the closed points for a mesh
            within the region defined by each function in
            self.regions_functions.
            TODO : Find a better way of doing this instead of re-evaluating
            the function at each closed point in the mesh

        '''
        # check function inputs:
        self.__validate_functions()

        closed_point_index = self.mesh.region_point[
                self.mesh.closedregionindex]

        closed_points_function = lambda fun: np.arange(len(self.mesh.points))[
                closed_point_index[fun(self.mesh.points[closed_point_index])]]

        closed_regions_point = {
                region_name: [np.sort(closed_points_function(function))
                              for i, function in region_func.items()]
                for region_name, region_func in self.regions_functions.items()}

        return closed_regions_point

    def search_points(self, selection, regions_point):
        '''
            Find the redundant points within a region defined by
            the user pre-defined functions (in poisson problem)
            and give the indices of the remaining non-redundant points in the
            mesh. There are three methods that can be used:

                i) Voronoi -> voronoi_selection()
                ii) Delaunay -> delaunay_selection()
                iii) Neuman-Dirichlet -> neuman_dirichlet_selection()

            For more information on each of thoses methods c.f. docstring
            of each one of them.

            Parameters:
            -----------
                selection :  dict()
                    example :
                            selection = {'Neuman-Dirichlet':[
                                    ['voltage',
                                     ['convex_voltage_0', 'convex_voltage_1']],
                                    ['charge', ['charge_0']]
                                                           ]
                                        }
                    The key indicates which function is going to be used.
                    Each element is a list containing other lists.
                        Each list within the previous list has as a fisrt
                        element the tag given to the region to which the
                        function will be applied to and as a second element
                        the concerned sub_regions.

                    If one writes ['convex_voltage','*'] then all
                        sub_regions within convex_voltage are called

                regions_point: given by discretize module, c.f. the latter.

            Returns:
            --------
                chosen_points: dict()
                    Contains as key the tag of each sub_region and
                        as element associated with that key the indices
                        for the non-redundant points.

        '''
        # check function inputs:
        self.__validate_functions()

        (voltage_name, charge_name, mixed_name,
         dielectric_name) = [self.regions_names_dict[key] for key in [
             'dirichlet', 'neuman', 'mixed', 'dielectric']]

        selection_functions = {'Voronoi': voronoi_selection,
                           'Delaunay': delaunay_selection,
                           'Neuman-Dirichlet': neuman_dirichlet_selection}

        chosen_points = {}
        for selec_fun, regions_to_select in selection.items():
            if selec_fun in selection_functions:
                for region in regions_to_select :

                    if region[1] is '*':
                        region[1] = list(
                                self.regions_functions[region[0]].keys())

                    try:
                        func_tuple = itemgetter(*region[1])(
                                self.regions_functions[region[0]])
                    except KeyError:
                        raise ValueError(('One of the functions given for '
                                         +'{0} does not exist or was not'
                                         + ' named correctly').format(
                                                 region[0]))

                    dirichlet_points = itemgetter(*region[1])(
                            regions_point[region[0]])

                    # It is either a tuple or a np.array
                    if isinstance(dirichlet_points, tuple):
                        dirichlet_points = [*dirichlet_points]
                    elif isinstance(dirichlet_points, np.ndarray):
                        dirichlet_points = [dirichlet_points]

                    neuman_points = np.concatenate(
                            list(regions_point[charge_name].values()))

                    chosen_points_CD = selection_functions[selec_fun](
                            convex_voltage_funcs=func_tuple,
                            mesh_points=self.mesh.points,
                            dirichlet_points=dirichlet_points,
                            neuman_points=neuman_points,
                            points_firstneig=self.mesh.points_firstneig)

                    chosen_points.update({
                            region[1][sub_region_num]: sub_region_points
                            for sub_region_num, sub_region_points in enumerate(
                                    chosen_points_CD)})
            else:
                raise ValueError('\n Selection key must be either "Voronoi"'
                                   + ' "Delaunay" or "Neuman-Dirichlet" \n')

        return chosen_points