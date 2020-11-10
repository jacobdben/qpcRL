''' Contains the class PoissonPorblem which has as inputs
   a set of continous functions defining the poisson problem.
   TODO: Add function to visualize the regions.
'''

__all__ = ['ContinuousGeometry']


import warnings
import copy
import itertools

import numpy as np

from matplotlib.colors import ListedColormap, BoundaryNorm

from poisson.continuous import shapes
from poisson.continuous.shapes import asshape, aspiecewise
from poisson.tools import plot as p_plt

def check_intersection(list_array, warning_message, names_list,
                       arrays_intersection=None):
    '''
        Check if all arrays is the list_arrays intersect with one another.
        Sends an warning-message in case they do.

        Parameters:
            list_array: list or array of n elements where each elements is an
                np.ndarray[dtype=bool, dim] and each array has the same
                number of elements.
            warning_message: string
            names_array: list with n elements where each element is an string
                that corresponds to the indentification of each array
                in list_array

        Returns:
            arrays_intersection: list of z elements where z corresponds
                to the number of "intersection events". Each element is a
                list containing as the first element the name of the
                intersecting arrays and as a second element the
                intersecting index
    '''
    if arrays_intersection is None:
        arrays_intersection = list()

    list_array_copy = copy.deepcopy(list_array)
    for i, array_1 in enumerate(list_array):
        for pos_array_2, array_2 in enumerate(list_array_copy):

            if np.array_equal(array_1, array_2) is False:
                # check for intersection
                mapping = np.prod((array_1, array_2),
                                  axis=0).astype(bool)
                # index that intersect
                intra_intersect = np.arange(len(array_1))[mapping]
                if len(intra_intersect) > 0:
                    arrays_intersection.append(
                            ['{0} intersection with {1}'.format(
                                    names_list[i],
                                    names_list[i + pos_array_2]),
                                    intra_intersect])
        del list_array_copy[0]

    if bool(arrays_intersection):
        print('\n')
        print('-'*48)
        print('\n')
        warnings.warn(warning_message, RuntimeWarning)
        print('\n')

    return arrays_intersection


class ContinuousGeometry(object):

    ''' Contains all nescessary information nescessary to obtain a discretized
      problem with exception.
   '''


    def __init__(self, space=None, voltage=None, charge=None, mixed=None,
                 dielectric=None, default_relative_permittivity=1):
        '''
        Parameters:
        ------------
        space -> function defining the total system or
            a shapes.Space class or subclass.

         voltage -> list, tuple or dict of functions, Shape class or one of
             its subclasses defining the voltage region (dirichlet region).
             Each function must return True for a point within the region
             defined by that function. Each element will form one subregion
             of the voltage region.
             In case it is a dictionary, the keys of each element
             will be used as the name of the subregion.

         charge -> the same as above but for regions where the charge is
             defined (Neuamn region).
             By default any region defined in space but not in voltage,
             charge or mixed is of charge type.

         mixed -> the same as above but for points where the boundary
               conditions are mutable ( such as for kwant points)

         dielectric -> list, tuple or dict. Each element is either a
             shapes.PieceiwiseFunction class, tuple or list:

                 e.g. ((func, val), ((func, val), (func, val)), ...),
                      [(func, val), ((func, val), (func, val)), ...]
                      {key: (func, val), key2: ((func, val), (func, val)), ...}

             An element of dielectric can thus be a tuple that has as elements:
                 (c.f. parameters of shapes.PiecewiseFunction)
                   1 -> tuples containing:
                        1. function (such as the one defined for
                           Shape.geometry(x)) or shapes.Shape class (func).
                        2. constant or a function that can be evaluated
                            in a set of coordinates x (np.ndarray) returing
                            a number (val).

                        e.g. dielectric[i][0] = (func, constant)
                             dielectric[i][0] = (func, func)
                             dielectric[i][0] = (shapes.Shape, constant)
                             dielectric[i][0] = (shapes.Shape, func)

                   2 -> two elements:
                        list_input[i][0]: function (such as the one defined for
                            Shape.geometry(x)) or shapes.Shape class as
                            the first element

                        list_input[i][1]: constant or a function that can be
                            evaluated in a set of coordinates x (np.ndarray)
                            returing a number.
                        (c.f it is the value parameter in shapes.WrapperShape)

         default_relative_permittivity = 1

        Obs 1: If a region in space is not defined in neither voltage, charge
            or mixed then it is defined by default as charge. This happnes
            if space is defined.

        Obs 2: dielectric functions do not need to cover the entire system.
            In case space is not given and dielectric functions do not cover
            the entire system, the missing points will have de default value
            of scipy.constants.epsilon_0 (8.85e-12).
            In case the space is given, and 'default_dielectric' will be
            function will be created with a default value of 1. Thus the points
            within space will have as a value the one given by
            scipy.constants.epsilon_0.
            The default value can be changed by adding a
            default_relative_permittivity option in the attributes.

        '''

        if voltage is None:
            voltage = []
        if charge is None:
            charge = []
        if mixed is None:
            mixed = []
        if dielectric is None:
            dielectric = []

        voltage = asshape(voltage)
        charge = asshape(charge)
        mixed = asshape(mixed)
        dielectric = aspiecewise(dielectric)
        if (not isinstance(space, shapes.Shape)) and (space is not None):
            if callable(space) and (not isinstance(space, (list, tuple))):
                space = shapes.General(space)
            else:
                raise ValueError(('space cannot be a tuple or list and it must'
                                  + ' either be a function, shape.Shape class'
                                  + ' or one of its sublcasses'))

        # Inputs
        self.space = space
        self.regions_functions = None
        self.sub_region_names = None
        self.dielectric_functions = None
        self.default_relative_permittivity = default_relative_permittivity

        self.regions_names = ['voltage', 'charge', 'mixed']

        self.regions_names_dict = {'dirichlet':self.regions_names[0],
                           'neuman':self.regions_names[1],
                           'mixed':self.regions_names[2],
                           'dielectric':'dielectric'}

        regions_functions_temp = {self.regions_names[0]: voltage,
                                  self.regions_names[1]: charge,
                                  self.regions_names[2]: mixed
                                  }

        self.__tag(regions_functions_temp=regions_functions_temp,
                   dielectric=dielectric)

    def __tag(self, regions_functions_temp, dielectric):
        '''
            TODO: doc
        '''
        self.sub_region_names = dict()
        self.regions_functions = dict()

        # Going through voltage, charge and mixed regions'
        for region_number, region_name in enumerate(self.regions_names):

            sub_region_func_all = regions_functions_temp[region_name]

            if isinstance(sub_region_func_all, (list, tuple)):

                # Create tag
                names_list_temp = [(self.regions_names[region_number]
                                    + '_{0:d}').format(i)
                                   for i in range(len(sub_region_func_all))]

                self.sub_region_names.update({region_name: names_list_temp})

                # creating a dictonary associating the region_name to its
                # function
                sub_regions_functions = {}
                for sub_reg_number, sub_reg_key in enumerate(names_list_temp):
                    sub_regions_functions.update({
                            sub_reg_key: sub_region_func_all[sub_reg_number]})

                self.regions_functions.update({
                        region_name: sub_regions_functions})

            elif isinstance(sub_region_func_all, dict):

                self.regions_functions.update({
                        region_name: sub_region_func_all})

                self.sub_region_names.update({
                        region_name: list(sub_region_func_all.keys())})

        # Add undefined region as the difference between the intersection of
        # the charge, voltage and mixed functions with the complete space
        # function.The tag is 'default'
        list_reg_shapes = list(itertools.chain.from_iterable(
                [reg.values() for reg in self.regions_functions.values()]))
        if self.space is not None:
            # In case charge, voltage and mixed regions are not associated to
            # any function each list will be empty, which would return an error
            if bool(list_reg_shapes):
                # regions_functions_temp is a dict containing as
                # values the lists containing the Shape objects (
                # therefore two np.sum are needed).
                # space_temp is only a list
                self.regions_functions[self.regions_names[1]].update({
                        'charge_default': (self.space
                                           - np.sum(list_reg_shapes))})

            elif (bool(list(self.regions_functions['voltage'].values()))
                  and bool(list(self.regions_functions['mixed'].values()))):
                raise ValueError(('The voltage value must be fixed on at least'
                                 + ' one point in the system'))
            else:
                self.regions_functions[self.regions_names[1]].update({
                        'charge_default': self.space})
#                raise ValueError(('The voltage value must be fixed on at least'
#                                 + ' one point in the system'))
            self.sub_region_names[self.regions_names[1]].append(
                    'charge_default')

        # Going through dielectric functions
        if isinstance(dielectric, (tuple, list)):
            self.dielectric_functions = {'_'.join(
                    [self.regions_names_dict['dielectric'],
                     '{0:d}'.format(number)]):
                     function for number, function in enumerate(dielectric)}
        elif isinstance(dielectric, dict):
            self.dielectric_functions = dielectric

        if bool(self.dielectric_functions):

            if self.space is not None:
                total_shape = self.space
            elif bool(list_reg_shapes):
                total_shape = np.sum(list_reg_shapes)
            else:
                raise ValueError('No region is defined')

            default_die_shape = (
                    total_shape
                    - np.sum([shapes.General(die_pieceiwse.shape_function())
                    for die_pieceiwse in self.dielectric_functions.values()]))

            self.dielectric_functions.update({'default_dielectric':
                shapes.PiecewiseFunction((default_die_shape,
                                         self.default_relative_permittivity))})

        elif bool(self.dielectric_functions) is False:

            if self.space is not None:
                total_shape = self.space
            elif bool(list_reg_shapes):
                total_shape = np.sum(list_reg_shapes)
            else:
                raise ValueError('No region is defined')

            self.dielectric_functions.update({'default_dielectric':
                shapes.PiecewiseFunction((total_shape,
                                         self.default_relative_permittivity))})


    def check_functions(self, grid):
        '''
            Verify that the functions are chosen correctly, i.e. that the
            intersection of convex_voltage_funcs, other_voltage_funcs,
            charge_funcs and mixed are empty for a given grid.

            It tests if for each region (among convex_voltage, other_voltage,
            charge_funcs, mixed) there are no two sub_regions defining
            the same point in the grid.

            It tests if for all regions (among convex_voltage, other_voltage,
            charge_funcs, mixed) there are no region defining the same
            point in the grid.

            Parameter:
                grid: np.ndarray of nDimensions, i.e. the grid coordinates

            Returns:
                regions_intersection: list containing as:
                    first element -> name of the regions intersecting
                    second element -> points intersecting

                sub_regions_intersection: same as regions_intersection
            TODO: proper testing ?
       '''

        regions_boolean = [[function(grid)
                            for i, function in
                            self.regions_functions[region].items()]
                           for region in self.regions_functions
                           if bool(self.regions_functions[region])]

        partial_regions_names = [key for key in self.regions_functions
                         if bool(self.regions_functions[key])]

        sub_regions_intersection = self.check_all_subregions(
                regions_boolean=regions_boolean,
                partial_regions_names=partial_regions_names,
                grid=grid)

        regions_intersection = self.check_regions(
                regions_boolean=regions_boolean,
                partial_regions_names=partial_regions_names,
                grid=grid)

        return regions_intersection, sub_regions_intersection


    def check_regions(self, grid, regions_boolean=None,
                      partial_regions_names=None):
        '''
            TODO: Doc
        '''

        if (regions_boolean is None) and (partial_regions_names is None):

            regions_boolean = [[function(grid)
                                for i, function in
                                self.regions_functions[region].items()]
                               for region in self.regions_functions
                               if bool(self.regions_functions[region])]

            partial_regions_names = [key for key in self.regions_functions
                                     if bool(self.regions_functions[key])]


        for pos_region, region in enumerate(regions_boolean):
            if len(region) > 1:
                regions_boolean[pos_region] = np.any(region, axis=0)
            else:
                # They need to be a list[np.ndarray] and not a
                # list[list[np.ndarray]]
                regions_boolean[pos_region] = region[0]

        warning_message = (' \n \n Charge(Neuman), Voltage(Dirichlet) or mixed'
                           + 'functions'
                           + ' are not well defined. \n Their intersection is not'
                           + ' empty. \n Check poisson.geometry.Geometry or '
                           + ' poisson.system.System class attribute'
                           + ' \n regions_intersection for a list of the index of'
                           + ' the points belonging to the \n intersection \n ')

        regions_intersection = check_intersection(
                regions_boolean, warning_message, partial_regions_names)

        if bool(regions_intersection) is False:
            regions_intersection=None

        return regions_intersection

    def check_all_subregions(self, grid, regions_boolean=None,
                             partial_regions_names=None):
        '''
            TODO: Doc
        '''
        if (regions_boolean is None) and (partial_regions_names is None):

            regions_boolean = [[function(grid)
                                for i, function in
                                self.regions_functions[region].items()]
                               for region in self.regions_functions
                               if bool(self.regions_functions[region])]

            partial_regions_names = [key for key in self.regions_functions
                             if bool(self.regions_functions[key])]

        sub_regions_intersection = list()

        warning_message = (' \n \n {0} functions are not well defined.'
                           + '\n Their intersection is not empty. \n Check'
                           + ' poisson.system.System or poisson.geometry.Geometry'
                           + ' class attribute \n sub_regions'
                           + '_intersection for a list of the index of the'
                           + ' points belonging to \n the intersection \n ')

        for pos_region, region in enumerate(regions_boolean):

            warning_message = warning_message.format(
                    partial_regions_names[pos_region])
            #Check intersection of sub_regions for region
            sub_regions_intersection = check_intersection(
                    region, warning_message,
                    names_list=self.sub_region_names[
                            partial_regions_names[pos_region]],
                    arrays_intersection=sub_regions_intersection)

        if bool(sub_regions_intersection) is False:
            sub_regions_intersection = None

        return sub_regions_intersection

    def plot(self, direction=2, plot_type='2D', bbox=None,
             npoints=(100, 100), xlabel='x(nm)', ylabel='y(nm)', points=None,
             figsize=(10, 10), **kwargs):
        '''
        Plot the geometry.
            If '2d' plot -> One must give a bbox and a direction
            If '3d' plot -> One must give points.

            Parameters:
            -----------
                plot_type = str. Default is '2D'
                    '2D' -> 2D cut
                    '3D' -> 3D plot using mayavi

              -> Valid for 2D cut :

                  direction = int. Specifies the axis that is kept constant.
                        0 -> first column in the coordinates numpy array, i.e.
                            axis 'x',
                        1 - second column ..., i.e. axis 'y'
                        2 - third column ..., i.e. axis 'z'
                        Obs: No need to specify direction for  2D system

                  bbox = [1.min, 1.max, 2.min, 2.max].
                        Bounding box along directions 1 or 2.
                        If direction = 3 -> 1='x' and 2='y'
                                       2 -> 1='x' and 2='z'
                                       1 -> 1='y' and 2='z'
                  npoints = tuple, list.
                        Number of points in the plot.
                  xlabel = str
                        Default : 'x(nm)'
                  ylabel = str
                        Default: 'y(nm)'
                  figsize = tuple or list
                    **kwargs -> options sent to plot_toolbox.sliced_geometry_2d
                    obs: give cmap as option to control the colors.

              -> Valid for 3D plot:
                  points: np.array containing the coordinates for
                      which the geometry will be applied to. Cannot be None
                      if direction is None.
                 **kwargs -> options sent to plot_toolbox.sliced_geometry_2d
                    obs: give cmap as option to control the colors.

            Returns:
            --------
            If plot_type is '2D':
                Returns:
                --------
                 Function that has as:
                        Parameters:
                        -----------
                            variable: float
                                Value that determines at which point the
                                axis defined in direction will be cut.
                            ax_val: ax class from matplotlib.pyplot.
                                Option. In case not specified a figure will be
                                    created.
                            xlabel, ylabel: str
                                Optional. In case not specified the values
                                given as paramters in plot_geometry_2d will
                                be used.
                            figsze: (float, float). Same as above
                            colorbar_label: str
                            **kwargs_2 : set of parameters passed to
                                matplotlib.pyplot.imshow. Takes preference over
                                kwargs sent to plot_geometry_2d
                        Returns:
                        --------
                            imshow: imshow object of matplotlib.pyplot
                                containing the plot.object
                            ax: ax object of matplotlib.pyplot containing
                                the plot.object
                            colorbar: the colorbar object

            If plot_type is '3D':
                A 3D plot of the geometry.
                Uses mayavi.
        '''
        if plot_type is '2D':

            if not isinstance(direction, int):
                raise ValueError('Direction must be an int')
            return p_plt.plot_continuous_geometry(
                    self, direction=direction, bbox=bbox,
                    npoints=npoints, xlabel=xlabel, ylabel=ylabel,
                    figsize=figsize, **kwargs)
        elif plot_type is '3D':
            if not isinstance(points, np.ndarray):
                raise ValueError('Points must be a numpy array')
            return p_plt.points_3D_mavi(
                    cls_inst=self, points=points, **kwargs)



