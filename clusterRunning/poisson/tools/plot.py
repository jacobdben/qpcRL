'''
    Plotting functions toolbox
'''


__all__ = ['plot_continuous_geometry',
           'plot_linear_problem_3d',
           'plot_linear_problem_2d',
           'plot_linear_problem_1d',
           'points_3D_mavi',
           'plot_1d',
           'plot_geometry_2d',
           'plot_values_2d']


import itertools
import warnings

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from . import post_process
from poisson.discrete import FiniteVolume, DiscretePoisson
from poisson.continuous import ContinuousGeometry, GridBuilder
from poisson.continuous import grid


def plot_linear_problem_3d(linear_problem_inst, plot_='both', titles=None,
                           scale_factor=1, **kwargs):

    '''
        Wrapper around poisson_toolbox.plot_toolbox.points_3D_mavi().
        Parameters:
        ------------
            linear_problem_inst: instance of poisson.LinearProblem
            plot_: str. Default 'both'
                'both': plot both voltage and charge
                'voltage': plot voltage dispersion
                'charge': plot charge dispersion.
            title: str or tuple/list
                If plot_ is 'both'
                    titles can be a tuple or list containing two
                        titles with the folowing order:
                        (titles_voltage, titles_charge)
            scale_factor: float or tuple/list
                If plot_ is 'both'
                    scale_factor can be a tuple or list containing two
                    scale_factor with the folowing order:
                        (scale_factor_voltage, scale_factor_charge)
            kwargs: arguments passed to mayavi.mlab.points3d
                Add colormap parameter to change the default
                colormap.

    '''
    if 'colormap' not in kwargs.keys():
        kwargs.update({'colormap':'blue-red'})

    if titles is None:
        title_v, title_c = ('Voltage', 'Charge')
    elif isinstance(titles, (tuple, list)):
        title_v, title_c = titles
    else:
        title_v, title_c = (titles, titles)

    if isinstance(titles, (tuple, list)):
        scale_factor_v, scale_factor_c = scale_factor
    else:
        scale_factor_v, scale_factor_c = (scale_factor, scale_factor)

    if (plot_ is 'voltage') or (plot_ is 'both'):

        points_3D_mavi(
                points=linear_problem_inst.discrete_poisson.mesh.points,
                markers=linear_problem_inst.points_voltage,
                title=title_v, scale_factor= scale_factor_v,
                **kwargs)
    if (plot_ is 'charge') or (plot_ is 'both'):
        points_3D_mavi(
                points=linear_problem_inst.discrete_poisson.mesh.points,
                title=title_c, scale_factor= scale_factor_c,
                markers=linear_problem_inst.points_charge,
                **kwargs)


def plot_linear_problem_2d(
        linear_problem_inst, plot_type='both', direction=2, xlabel=None,
        ylabel=None, npoints=(100, 100), bbox=None, figsize=(11, 11)):
    '''
        Wrapper around poisson.plot_toolbox.sliced_values_2d.
        Plot a 2D imshow plot of the charge and voltage dispersion.

        Parameters:
        ------------
        linear_problem_inst: instance of poisson.LinearProblem
        plot_type: str. Default 'both'
            'both': will plot voltage and charge
            'voltage': will plot votage
            'charge': will plot charge
        directions: int, tuple or list.
            Correspond to the directions, i.e. axis, that will be held
            constant.
        xlabel: str
            If not specified -> '{0}(nm)'.format(axis[0])
        ylabel: str
            If not specified -> '{0}(nm)'.format(axis[1])
        bbox: list of tuples. Default is None
            If None -> default bbox is built using LinearProblem
                properties.
            If list
                bbox = [xmin, xmax, ymin, ymax], so that bbox can
                be used to build the list of coordinates at which the
                voltage and charge values will be plotted.
        npoints: int
            Number of points to be plotted. Only relevant if
            bbox='default' is used.

        Returns:
        --------
            If plot_type is 'both':
                Two functions, the first corresponds to the voltage plot
                    and the second to the charge plot.

                Each function take as:

                    Parameters:
                    ------------
                        variable: float, list or tuple of floats
                            Selects the position where the axis defined in
                            directions will be cut. It has the same type
                            and size as directions, e.g.if directions is a
                            tuple containing two elements, so must be
                            variables.
                            Default is 0.
                            Obs: If 2D problem no need to specify variable
                        ax: ax object of matplotlib.pyplot
                        xlabel: str (if not specified takes the one
                             given to plot_cut_2d)
                        ylabel: str (if not specified takes the one
                                 given to plot_cut_2d)
                        colorbar_label: str (if not specified takes the one
                                 given to plot_cut_2d)
                        kwargs: parameters passed to matplotlib.pyplot.plot

                The function will call ax.plot(.., kwargs) and:

                    Returns:
                    --------
                        imshow: imshow object of matplotlib.pyplot
                            containing the plot.object
                        ax: ax object of matplotlib.pyplot containing
                            the plot.object
                        colorbar: the colorbar object

            If plot_type is ('votage' or 'charge'):
               Either the function corresponding to the voltage plot or
               to the charge plot.
    '''
    axis = ['x', 'y', 'z']
    del axis[direction]

    if xlabel is None:
        xlabel_parent='{0}(nm)'.format(axis[0])
    else:
        xlabel_parent=xlabel
    if ylabel is None:
        ylabel_parent='{0}(nm)'.format(axis[1])
    else:
        ylabel=ylabel

    dimension = linear_problem_inst.discrete_poisson.mesh.points.shape[1]
    if dimension == 3:
        direc = np.delete(
                np.arange(linear_problem_inst.discrete_poisson.mesh.\
                          points.shape[1]),
                np.array([direction]))
    elif dimension == 2:
        direc = [0, 1]
    else:
        raise ValueError('Cannot make a 2D plot of a {}D system'.format(
                dimension))

    if bbox is None:
        direc = np.delete(
                np.arange(linear_problem_inst.discrete_poisson.mesh.\
                          points.shape[1]),
                np.array([direction]))
        bbox = [np.min(linear_problem_inst.discrete_poisson.mesh.points[
                :, direc[0]]),
                np.max(linear_problem_inst.discrete_poisson.mesh.points[
                        :, direc[0]]),
                np.min(linear_problem_inst.discrete_poisson.mesh.points[
                        :, direc[1]]),
                np.max(linear_problem_inst.discrete_poisson.mesh.points[
                        :, direc[1]])]

    if (plot_type is 'both') or (plot_type is 'voltage'):

        plot_voltage = plot_values_2d(
                bbox=bbox,
                points_value=linear_problem_inst.points_voltage,
                class_inst=linear_problem_inst.discrete_poisson,
                npoints=npoints, direction=direction)

        def func_voltage(variable=0, ax=None, xlabel=xlabel_parent,
                         ylabel=ylabel_parent, figsize=figsize,
                         colorbar_label=None, **kwargs):
            '''
                Parameters:
                ------------
                    variable: float, list or tuple of floats
                        Selects the position where the axis defined in
                        directions will be cut. It has the same type
                        and size as directions, e.g.if directions is a
                        tuple containing two elements, so must be
                        variables.
                        Default is 0.
                        Obs: If 2D problem no need to specify variable
                    ax: ax object of matplotlib.pyplot
                    xlabel: str (if not specified takes the one
                         given to plot_cut_2d)
                    ylabel: str (if not specified takes the one
                             given to plot_cut_2d)
                    colorbar_label: str (if not specified takes the one
                             given to plot_cut_2d)
                    kwargs: parameters passed to matplotlib.pyplot.plot

                The function will call ax.plot(.., kwargs) and:

                Returns:
                --------
                    imshow: imshow object of matplotlib.pyplot
                        containing the plot.object
                    ax: ax object of matplotlib.pyplot containing
                        the plot.object
            '''
            if dimension < 3:
                variable = 0
            if 'aspect' not in kwargs.keys():
                kwargs.update({'aspect':'equal'})
            if 'cmap' not in kwargs.keys():
                kwargs.update({'cmap':'seismic'})
            imag, ax_val, colobar = plot_voltage(
                    variable, ax, xlabel=xlabel, ylabel=ylabel,
                    colorbar_label=colorbar_label, figsize=figsize,
                    **kwargs)

            return imag, ax_val, colobar

    if (plot_type is 'both') or (plot_type is 'charge'):

        plot_charge = plot_values_2d(
                bbox=bbox,
                points_value=linear_problem_inst.points_charge,
                class_inst=linear_problem_inst.discrete_poisson,
                npoints=npoints, direction=direction)

        def func_charge(variable=0, ax=None, xlabel=xlabel_parent,
                         ylabel=ylabel_parent, figsize=figsize,
                         colorbar_label=None, **kwargs):
            '''
                Parameters:
                ------------
                    variable: float, list or tuple of floats
                        Selects the position where the axis defined in
                        directions will be cut. It has the same type
                        and size as directions, e.g.if directions is a
                        tuple containing two elements, so must be
                        variables.
                        Default is 0.
                        Obs: If 2D problem no need to specify variable
                    ax: ax object of matplotlib.pyplot
                    xlabel: str (if not specified takes the one
                         given to plot_cut_2d)
                    ylabel: str (if not specified takes the one
                             given to plot_cut_2d)
                    colorbar_label: str (if not specified takes the one
                             given to plot_cut_2d)
                    kwargs: parameters passed to matplotlib.pyplot.plot

                The function will call ax.plot(.., kwargs) and:

                Returns:
                --------
                    imshow: imshow object of matplotlib.pyplot
                        containing the plot.object
                    ax: ax object of matplotlib.pyplot containing
                        the plot.object
            '''
            if dimension < 3:
                 variable = 0
            if 'aspect' not in kwargs.keys():
                 kwargs.update({'aspect':'equal'})
            if 'cmap' not in kwargs.keys():
                 kwargs.update({'cmap':'seismic'})

            imag, ax_val, colorbar = plot_charge(
                    variable, ax, xlabel=xlabel, ylabel=ylabel,
                    colorbar_label=colorbar_label, figsize=figsize,
                    **kwargs)

            return imag, ax_val, colorbar

    if plot_type is 'both':
        return func_voltage, func_charge,
    elif plot_type is 'voltage':
        return func_voltage
    elif plot_type is 'charge':
        return func_charge
    else:
        raise('plot_type {} does not correspond to any option'.format(
                plot_type))


def plot_linear_problem_1d(
        linear_problem_inst, directions, plot_type='both', bbox=None,
        npoints=200, figsize=(10, 10), interpolation_data=0, decimals=5):
    '''
        Wrapper around plot_toolbox.sliced_values_1d and
        the function it returns.
        Parameters:
        -----------
        linear_problem_inst: instance of poisson.LinearProblem
        directions = list or tuple with one or two elements (int).
            Specifies the axis that is kept constant.
            0 -> first column in the coordinates numpy array, i.e.
                axis 'x',
            1 - second column ..., i.e. axis 'y'
            2 - third column ..., i.e. axis 'z'
        plot_type: str. Default 'both'
            'both': will plot voltage and charge
            'voltage': will plot votage
            'charge': will plot charge
        bbox: list of tuples or 'default'.
            If 'default' -> default bbox is built using LinearProblem
                properties.
            If list of tuples, each tuple contaings three elements, i.e.
                [(min, max, npoints), ...], so that bbox can
                be used to build the list of coordinates at which the
                voltage and charge values will be plotted, i.e.
                coordinates = np.linspace(*bbox[i])
            If not specified, LinearProblem.system.mesh.points
                is used.
        npoints: int
            Number of points to be plotted. Only relevant if
            bbox='default' is used.
        interpolation_data: by default 0, i.e. no interpolation.
            Not properly implemented
        decimals: int -> c.f.  poisson.post_process.slice_coordinates

        Returns:
        --------
            If plot_type is 'both':
                Two functions, the first corresponds to the voltage plot
                    and the second to the charge plot.

                Each function take as :

                    Parameters:
                    ------------
                        variables: float, list or tuple of floats
                            Selects the position where the axis defined in
                            directions will be cut. It has the same type
                            and size as directions, e.g.if directions is a
                            tuple containing two elements, so must be
                            variables.
                        ax: ax object of matplotlib.pyplot

                        kwargs: parameters passed to
                            matplotlib.pyplot.plot

                The function will call ax.plot(.., kwargs) and

                    Returns:
                        ax: ax object of matplotlib.pyplot containing
                            the plot.
                        list: contaning two numpy arrays
                            with the data points plotted.

            If plot_type is ('votage' or 'charge'):
                Either the function corresponding to the voltage plot or
                    to the charge plot.

    '''
    dimension = linear_problem_inst.discrete_poisson.mesh.points.shape[1]
    if bbox is 'default':
        direc = np.arange(dimension, dtype=int)[np.logical_not(
                np.in1d(np.arange(dimension, dtype=int), directions))][0]
        bbox = [(np.min(linear_problem_inst.discrete_poisson.mesh.points[:, direc]),
                 np.max(linear_problem_inst.discrete_poisson.mesh.points[:, direc]),
                 npoints)]

    if (plot_type is 'both') or (plot_type is 'voltage'):
        plot_voltage = plot_1d(
            figsize=figsize, directions=directions,
            class_inst=linear_problem_inst.discrete_poisson,
            points_value=linear_problem_inst.points_voltage, bbox=bbox,
            interpolation_data=interpolation_data, decimals=decimals)

        def func_volt(variables, ax, **kwargs):
            '''
                Parameters:
                ------------
                    variables: float, list or tuple of floats
                        Selects the position where the axis defined in
                        directions will be cut. It has the same type
                        and size as directions, e.g.if directions is a
                        tuple containing two elements, so must be
                        variables.
                    ax: ax object of matplotlib.pyplot

                    kwargs: parameters passed to
                        matplotlib.pyplot.plot

                The function will call ax.plot(.., kwargs) and

                Returns:
                    ax: ax object of matplotlib.pyplot containing
                        the plot.
                    list: contaning two numpy arrays
                        with the data points plotted.
            '''
            t, data_volt = plot_voltage(
                    variables, ax = ax, return_data=True, **kwargs)

            return  t, data_volt

    if (plot_type is 'both') or (plot_type is 'charge'):
        plot_charge = plot_1d(
            figsize=figsize, directions=directions,
            class_inst=linear_problem_inst.discrete_poisson,
            points_value=linear_problem_inst.points_charge, bbox=bbox,
            interpolation_data=interpolation_data, decimals=decimals)

        def func_charge(variables, ax, **kwargs):
            '''
                Parameters:
                ------------
                    variables: float, list or tuple of floats
                        Selects the position where the axis defined in
                        directions will be cut. It has the same type
                        and size as directions, e.g.if directions is a
                        tuple containing two elements, so must be
                        variables.
                    ax: ax object of matplotlib.pyplot

                    kwargs: parameters passed to
                        matplotlib.pyplot.plot

                The function will call ax.plot(.., kwargs) and

                Returns:
                    ax: ax object of matplotlib.pyplot containing
                        the plot.
                    list: contaning two numpy arrays
                        with the data points plotted.
            '''

            t, data_charge = plot_charge(
                    variables, ax = ax, return_data=True, **kwargs)

            return t, data_charge

    if plot_type is 'both':
        return func_volt, func_charge,
    elif plot_type is 'voltage':
        return func_volt
    elif plot_type is 'charge':
        return func_charge
    else:
        raise('plot_type {} does not correspond to any option'.format(
                plot_type))


def plot_continuous_geometry(
        geometry_inst, direction=2, bbox=None, npoints=(100, 100),
        xlabel='x(nm)', ylabel='y(nm)', figsize=(10, 10), **kwargs):
    '''
    Plot the geomtry.

        Parameters:
        -----------
            geometry_inst: poisson.ContinuousGeometry class

            direction = int. Specifies the axis that is kept constant.
                0 -> first column in the coordinates numpy array, i.e.
                    axis 'x',
                1 - second column ..., i.e. axis 'y'
                2 - third column ..., i.e. axis 'z'

                Obs: if it is a 2D system, direction must be 2, which
                    is the default value.

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

            **kwargs -> options sent to plot_toolbox.plot_geometry_2d
            obs: give cmap as option to control the colors.

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
                        Optional. In case not specified the values given
                            as paramters in plot_geometry_2d will be used.
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
    '''
    list_reg_func = list(itertools.chain.from_iterable(
            [reg.values() for reg in geometry_inst.regions_functions.values()]))
    list_reg_names = list(itertools.chain.from_iterable(
            [reg.keys() for reg in geometry_inst.regions_functions.values()]))

    if isinstance(direction, int):

        interpolation = kwargs.get('interpolation', 'none')
        if 'interpolation' in kwargs.keys():
            del kwargs['interpolation']
        aspect = kwargs.get('aspect', 'equal')
        if 'aspect' in kwargs.keys():
            del kwargs['aspect']

        plot_fun = plot_geometry_2d(
              figsize=figsize, bbox=bbox,
              npoints=npoints,
              fctnames=list_reg_names,
              functions=list_reg_func,
              xlabel=xlabel, ylabel=ylabel, direction=direction,
              vmin=-2, vmax=2,
              interpolation=interpolation, aspect=aspect)

        return plot_fun
    else:
        raise ValueError('Direction must be defined')


def points_3D_mavi(cls_inst=None, points=None, markers=None,
                   title=None, **kwargs):
    '''
        Parameters:
        -----------
            cls_inst: poisson.FiniteVolume,
                poisson.DiscretePoisson instance,
                poisson.ContinuousGeometry or one of its subclasses
            points: np.ndarray with the coordinates of the points
            markers: marker corresponding to each point

            kwargs as options passed to mayavi.mlab.points3d

        obs: If geometry_inst is not None then points must not
            be none neither.

        TODO:Check if the point_label part is working
    '''

    try:
        from mayavi import mlab
    except:
        warnings.warn(('mayavi can"t be imported.'
                          + ' The 3D plot will not be shown.'),
                      ImportWarning)
        return None

    if points is not None:
        points=points
    elif isinstance(cls_inst, FiniteVolume):
        points = cls_inst.points
    elif isinstance(cls_inst, DiscretePoisson):
        points = cls_inst.mesh.points
        if markers is None:
            markers = make_markers(cls_inst, points)
    elif isinstance(cls_inst, GridBuilder):
        points = cls_inst.points
        if markers is None:
            markers = make_markers(cls_inst)
    else:
        raise ValueError('Please give an either a mesh_inst, \
                         a system_inst or a a set of points coordinates')

    if isinstance(cls_inst, ContinuousGeometry):
        if markers is None:
            markers = make_markers(cls_inst, points)

    if markers is None:
        p3Dobj = mlab.points3d(np.reshape(points[:, 0],
                                          (len(points[:, 0]), 1)),
                               np.reshape(points[:, 1],
                                          (len(points[:, 1]), 1)),
                               np.reshape(points[:, 2],
                                          (len(points[:, 2]), 1)),
                               **kwargs)
    else:
        p3Dobj = mlab.points3d(np.reshape(points[:, 0],
                                          (len(points[:, 0]), 1)),
                               np.reshape(points[:, 1],
                                          (len(points[:, 1]), 1)),
                               np.reshape(points[:, 2],
                                          (len(points[:, 2]), 1)),
                               markers[:, None],
                              **kwargs)

    mlab.colorbar(p3Dobj, title=title)
    mlab.show()

    return p3Dobj


def make_markers(cls_inst, points=None):
    '''
        Create a numpy array marking each point in the system with a tag.
        This is supposed to be used as a marker in mlab.points3d

        Parameters:
        -----------
            cls_inst: instance of type:
                poisson.DiscretePoisson,
                poisson.ContinuousGeometry,
                poisson.GridBuilder
            points: np.ndarray with the coordinates of the points
                None by default.
                Not used if poisson.GridBuilder is given.

    '''
    if isinstance(cls_inst, (ContinuousGeometry, DiscretePoisson)):
        list_reg_func = list(itertools.chain.from_iterable(
                    [reg.values()
                     for reg in cls_inst.regions_functions.values()]))
        markers = np.ones(points.shape[0])
        colors = np.linspace(101,
                             len(list_reg_func) + 101,
                             len(list_reg_func)) / 100
        for pos, func in enumerate(list_reg_func):
            mapping = func(points)
            markers[mapping] = markers[mapping] * colors[pos]
    elif isinstance(cls_inst, GridBuilder):
        markers = cls_inst.point_label
    else:
        raise TypeError('cls_inst must be an instance of {}'.format(
                ' or '.join(['poisson.DiscretePoisson',
                             'poisson.ContinuousGeometry',
                             'poisson.GridBuilder'])))

    return markers


def plot_1d(directions=(0), coordinates=None, bbox=None,
                     function=None, class_inst=None, points_value=None,
                     interpolation_data=0, decimals=5,
                     figsize=(5, 5), xlabel=None, ylabel=None, **kwargs):
    '''
        Plot cut along 1 direction.

        Here must define either:
            directions, coordinates, function;
            directions, bbox, function;
            directions, class_inst, function;
            directions, class_inst, points_value.

        Parameters:
        -----------
            directions = list or tuple with one or two elements (int).
                Specifies the axis that is kept constant.
                0 -> first column in the coordinates numpy array, i.e.
                    axis 'x',
                1 - second column ..., i.e. axis 'y'
                2 - third column ..., i.e. axis 'z'

        ----- One must only give one of these three parameters:

                coordinates: numpy array

                class_inst: instance of a poisson.DiscretePoisson
                    or instance of a poisson.FiniteVolume

                bbox: list of tuples.
                    In each tuple there are three elements:
                        bbox[i] = (min, max, npoints)
                        such that coordinates = np.linspace(*bbox[i])
                        At least dimensions - 1  tupes must be defined in the bbox.

        ----- One must only give one of these two parameters:

            function: function taking as paramter a numpy array and
                returning two numpy arrays, the first being a bollean
                and the second specifying the value at each coordinate
                in the numpy array given as parameter.

                It Also accepts a function that only returns one
                numpy array contaning the values.

            points_value: numpy array.
                Can only be given if class_inst is given.
                It gives the value for a point in the mesh defined
                by DiscretePoisson  or FiniteVolume$.

        interpolation_data: c.f. post_process.value_from_coordinates
        decimals: c.f. description in slice_coordinates. Default is 5.

    -------- Parameters that do not need to be set when initializing ----------
    -------- plot_geometry_2d -----------------------------------------------

            figsize= tuple -> (5, 5)
                Optional (can be specified later).
            xlabel and ylabel: string
                Can be specified later
            direction_label=['x', 'y', 'z'] (default).
            kwargs: set of parameters passed to matplotlib.pyplot.plot

        Returns:
        --------
            Function.
                c.f. plot_plot_1d for more info.


    '''
    if 'direction_label' not in kwargs.keys():
        direction_label = np.array(['x', 'y', 'z'])[list(directions)]
    else:
        direction_label = kwargs['direction_label']

    if class_inst is not None:
        if isinstance(class_inst, DiscretePoisson ):
            mesh = class_inst.mesh
        elif isinstance(class_inst, FiniteVolume):
            mesh = class_inst
        coordinates = mesh.points
        dimension = coordinates.shape[1]
    elif coordinates is not None:
        dimension = coordinates.shape[1]
    else:
        raise ValueError(('Please provide either a class_inst,'
                          + '  a set of coordinates or a bbox'))

    if isinstance(directions, (tuple, list)):
        if len(directions) != (dimension -1):
            raise ValueError(
                    ('directions must have {} elements'
                     + ' for a {}D problem').format(
                            (dimension -1), dimension))

    if points_value is not None:
        if function is not None:
            raise ValueError('Both functions and points_value are defined')
        if class_inst is None:
            raise ValueError('points_value defined but not class_inst')
        function = post_process.value_from_coordinates(
                points_value,
                class_inst,
                interpolation_data)

    return plot_sliced_values_1d(
            bbox=bbox, coordinates=coordinates, function=function,
            dimension=dimension, directions=directions, decimals=decimals,
            figsize=figsize, direction_label=direction_label,
            xlabel=xlabel, ylabel=ylabel, **kwargs)

def  plot_sliced_values_1d(bbox=None, coordinates=None, function=None,
                           dimension=None, directions=0, decimals=5,
                           parent_dict=None, figsize=(5, 5),
                           direction_label=['x', 'y', 'z'], xlabel=None,
                           ylabel=None, **kwargs):
    '''
        Returns a function that allows to plot a cut along one direction
        of a set of coordinates or one that is generated from a bbox.
        The values of the coordinates are calculated by applying the
        numpy array coordinates to function.

        Parameters:
        -----------
            bbox: list of tuples.
                In each tuple there are three elements:
                    bbox[i] = (min, max, npoints)
                such that coordinates = np.linspace(*bbox[i])
                At least dimensions - 1  tupes must be defined in the bbox.
            coordinates: numpy array
            function: function taking as paramter a numpy array and
                returning two numpy arrays, the first being a bollean
                and the second specifying the value at each coordinate
                in the numpy array given as parameter.

                It Also accepts a function that only returns one
                numpy array contaning the values.
            dimension: int
                2, 3
            directions: tuple or list contaning one or two int, e.g. (1, 2)
                The int can take values of 0, 1, 2.
                They correspond to the axis (columns) in coordinates where
                the componets will remain the same for all points,
                e.g for a cut along 1  one writes directions=(0, 2)
            decimals: c.f. description in slice_coordinates. Default is 5.
            figsize: tuples, e.g. (5, 10)
            direction_label: e.g. ['x', 'y', 'z']
            xlabel and ylabel strings.
            kwargs: set of parameters passed to matplotlib.pyplot.plot
        Returns:
        --------
            A function that take as:
                Parameters:
                -----------
                    variable: float
                        Value that determines at which point the
                        axis defined in direction will be cut.
                    ax_val: ax class from matplotlib.pyplot.
                        Option. In case not specified a figure will be
                            created.

            -------- All parameters bellow are option and replace -------------
            --- the ones defined in plot_sliced_values_1d ---------------------
                    xlabel, ylabel: str
                        Optional. In case not specified the values given
                            as paramters in plot_geometry_2d will be used.
                    figsze: (float, float). Same as above
                    direction_label: e.g. ['x', 'y', 'z']
                    **kwargs_2 : set of parameters passed to
                        matplotlib.pyplot.plot. Takes preference over kwargs

                Returns:
                ---------
                    ax: ax class from matplotlib.pyplot.
                        Either the one given as parameter or
                            the one created to make the plot.
                    tuple containing the data used to make the plot, i.e.
                    (coordinates_1d, coordinates_values)
    '''

    direction = np.arange(dimension, dtype=int)[np.logical_not(
            np.in1d(np.arange(dimension, dtype=int), directions))][0]

    def f(variable, ax=None, return_data=False, **kwargs_2):
        '''
            Parameters:
                -----------
                    variable: float
                        Value that determines at which point the
                        axis defined in direction will be cut.
                    ax_val: ax class from matplotlib.pyplot.
                        Option. In case not specified a figure will be
                            created.
                    return_data: boolean
                        Default: False
                        If True the function returns the data
                        used to make the plot.

            -------- All parameters bellow are option and replace -------------
            --- the ones defined in plot_sliced_values_1d ---------------------
                    xlabel, ylabel: str
                        Optional. In case not specified the values given
                            as paramters in plot_geometry_2d will be used.
                    figsze: (float, float). Same as above
                    direction_label: e.g. ['x', 'y', 'z']
                    **kwargs_2 : set of parameters passed to
                        matplotlib.pyplot.plot. Takes preference over kwargs

                Returns:
                ---------
                    ax: ax class from matplotlib.pyplot.
                        Either the one given as parameter or
                            the one created to make the plot.

                    If return_data is True:
                        tuple: containing the data used to make the plot, i.e.
                            (coordinates_1d, coordinates_values)
        '''

        if (kwargs is not None) and kwargs_2: # check if kwargs_2 is empty
            kwargs.update(kwargs_2)
        kwargs_2 = kwargs

        if isinstance(variable, (tuple, list)):
            if len(variable) != (dimension -1):
                raise ValueError(
                        ('variable must have at least {} elements'
                         + ' for a {}D problem').format(
                                (dimension -1), dimension))
        elif dimension !=2:
            raise ValueError(('variable must be a tuple or list with'
                              + ' {0} elements for a {1}D problem').format(
                                      (dimension -1), dimension))

        if bbox is not None:
            coordinates_1d = make_coordinates_1d(
                    bbox=bbox, dimension=dimension,
                    directions=directions, variables=variable)
        elif coordinates is not None:
            coordinates_1d, indices_1d = post_process.slice_1d_coordinates(
                    coordinates = coordinates,
                    directions=directions, variables=variable,
                    decimals=decimals)
        else:
           raise ValueError(('Please provide either a class_inst,'
                             + '  a set of coordinates or a bbox'))

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            if dimension ==1:
                plt.title('{} = {}'.format(direction_label[0], variable[0]))
            elif dimension == 2:
                plt.title('{} = {} and {} = {}'.format(
                        direction_label[0], variable[0],
                        direction_label[1], variable[1]))

        coordinates_results = function(coordinates_1d)
        if len(coordinates_results) == 1:
            coordinates_values = coordinates_results
        elif len(coordinates_results) == 2:
            coordinates_values = coordinates_results[1]
        else:
            raise ValueError('The functionre returns more than 2 objects')
        print(kwargs_2)
        ax.plot(coordinates_1d[:, direction], coordinates_values, **kwargs_2)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if return_data:
            return ax, (coordinates_1d, coordinates_values)
        else:
            return ax

    return f


def plot_geometry_2d(bbox, direction=0, functions=None, fctnames=None,
                       npoints=(100, 100), figsize=(5, 5), xlabel=None,
                       ylabel=None, direction_label=['x', 'y', 'z'],
                       **kwargs):
    '''
        Wrapper around call_slice_2d.

        Take poisson.Shapes class or another function
        behaving in a similar way, i.e. returining True or False and
        taking as parameter a set of coordinates.

        Plot the geometry specified by that set of functions and/or set
        of Shapes classes ( or its subclasses).

        Here one must define: bbox, direction, functions and fctnames

        Parameters:
        -----------
            direction = int. Specifies the axis that is kept constant.
                0 -> first column in the coordinates numpy array, i.e.
                    axis 'x',
                1 - second column ..., i.e. axis 'y'
                2 - third column ..., i.e. axis 'z'
            functions = list of poisson.Shapes class or
                any function accepting a set of coordinates and returining
                True or False.
            fctnames = list of strings. Names the functions in the previous
                parameter.
            npoints =  tuple -> (100, 100).
                Specifies the number of points in the imshow that
                will be generated.

    -------- Parameters that do not need to be set when initializing ----------
    -------- plot_geometry_2d -----------------------------------------------

            figsize= tuple -> (5, 5)
                Optional (can be specified later).
            xlabel and ylabel: string
                Can be specified later
            direction_label=['x', 'y', 'z'] (deufalt).
            kwargs: set of parameters passed to matplotlib.pyplot.imshow

        Returns:
        ---------
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
                        Optional. In case not specified the values given
                            as paramters in plot_geometry_2d will be used.
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

    '''
    return call_slice_2d(
            plot='Geometry', functions=functions, direction=direction,
            fctnames=fctnames, bbox=bbox, npoints=npoints,
            xlabel=xlabel, ylabel=ylabel, figsize=figsize,
            direction_label=direction_label, **kwargs)


def plot_values_2d(bbox, npoints=(100, 100), direction=0, points_value=None,
                     functions=None, class_inst=None,
                     direction_label=['x', 'y', 'z'],
                     figsize=(5, 5), xlabel=None, ylabel=None,
                     interpolation_data=0, **kwargs):
    '''
        Wrapper around call_slice_2d.

        Takes a list of functions returning two numpy arrays, the
        first with booleans and the other with floats, and taking
        as parameter a numpy array with a set of coordinates. Plots an
        imshow of the values received from the function for a
        set of coordinates inside the bbox.

        One can also give a set of values in a numpy array instead of
        a set of functions.

        A third option is to only give a poisson.DiscretePoisson
        object instead.

        Plot the geometry specified by that set of functions and/or set
        of Shapes classes ( or its subclasses).

        Here must define either:
            bbox, npoints, direction, points_value, class_inst
            bbox, npoints, direction, functions

        Parameters:
        -----------
            bbox = [1.min, 1.max, 2.min, 2.max]. Bounding box along directions
                the two axis that will not be kept constant.
            direction = int. Specifies the axis that is kept constant.
                0 -> first column in the coordinates numpy array, i.e.
                    axis 'x',
                1 - second column ..., i.e. axis 'y'
                2 - third column ..., i.e. axis 'z'
            class_inst: instace of a poisson.DiscretePoisson
                    or instance of a poisson.FiniteVolume
            points_value: numpy array.
                Can only be given if class_inst is given.
                It gives the value for a point in the mesh defined
                by DiscretePoisson  or poisson.FiniteVolume.
            functions = list of poisson.Shapes class or
                any function accepting a set of coordinates and returining
                True or False.
            npoints =  tuple -> (100, 100).
                Specifies the number of points in the imshow that
                will be generated.
            interpolation_data: c.f. post_process.value_from_coordinates

    -------- Parameters that do not need to be set when initializing ----------
    -------- plot_geometry_2d -----------------------------------------------

            figsize= tuple -> (5, 5)
                Optional (can be specified later).
            xlabel and ylabel: string
                Can be specified later
            direction_label=['x', 'y', 'z'] (deufalt).
            kwargs: set of parameters passed to matplotlib.pyplot.imshow

        Returns:
        ---------
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
                        Optional. In case not specified the values given
                            as paramters in plot_geometry_2d will be used.
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
    '''
    if points_value is not None:
        if functions is not None:
            raise ValueError('Both functions and points_value are defined')
        if class_inst is None:
            raise ValueError('points_value defined but not class_inst')

        functions = [post_process.value_from_coordinates(
                points_value, class_inst,interpolation_data)]

    return call_slice_2d(
            plot='Value', functions=functions, direction=direction,
            bbox=bbox, npoints=npoints,
            direction_label=direction_label, xlabel=xlabel, ylabel=ylabel,
            figsize=figsize, **kwargs)


def make_coordinates_1d(bbox, dimension, directions, variables):
    '''
        Get a set of coordinates from a bbox.
        The coordinates will have one or two constant componants.
        These components are given in variables.

        Parameters:
        ------------
            bbox: list of tupes, e.g. [(min, max, npoints), ...]
            directions: tuple or list contaning one or two int, e.g. (1, 2)
                The int can take values of 0, 1, 2.
                They correspond to the axis (columns) in coordinates where
                the componets will remain the same for all points,
                e.g for a cut along 1  one writes directions=(0, 2)
            dimension: 2, 3
                The dimensions of the set of coordinates that will
                be returned
            variables: tuple or list containing one or two floats.
                Specifies the values for the constant componants of
                the coordinates.

        Returns:
        --------
            A numpy array with dimension given by the dimension
            parameter.

    '''
    direction = np.arange(dimension, dtype=int)[np.logical_not(
            np.in1d(np.arange(dimension, dtype=int), directions))][0]
    list_to_concatenate = [[] for i in range(dimension)]

    # So it accepts two or more intervals.
    for pos_bbox in range(len(bbox)):

        if len(bbox[pos_bbox]) != 3:
            raise ValueError(
                    'Element {0} in bbox is badly defined'.format(
                            pos_bbox))

        list_coord = [None] * int(dimension)
        list_coord[direction] = np.linspace(*bbox[pos_bbox])[:, None]

        for numb, i in enumerate(directions):

            list_coord[i] = (np.ones(bbox[pos_bbox][2])
                             * variables[numb])[:, None]

        for i in range(dimension):
            list_to_concatenate[i].append(list_coord[i])

    list_coord = [list(itertools.chain.from_iterable(list_co))
                  for list_co in list_to_concatenate]

    coordinates = np.hstack(list_coord)

    return coordinates


def call_slice_2d(plot='Value', functions=None, fctnames=None, direction=1,
          bbox=None, npoints=None, xlabel=None, ylabel=None,
          colorbar_label=None, figsize=None, direction_label=['x', 'y', 'z'],
          **kwargs):
    '''
        Wrapper around slice_2d to avoid code redundancy in
        plot_values_2d and plot_geometry_2d.

        Parameters, c.f. slice_2d, plot_values_2d, plot_geometry_2d

        Returns:
        ---------
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
                        Optional. In case not specified the values given
                            as paramters in plot_geometry_2d will be used.
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
    '''

    def f(variable, ax_val=None, xlabel=xlabel, ylabel=ylabel,
          colorbar_label=None, figsize=figsize, **kwargs_2):
        '''
             Parameters:
            -----------
                variable: float
                    Value that determines at which point the
                    axis defined in direction will be cut.
                ax_val: ax class from matplotlib.pyplot.
                    Option. In case not specified a figure will be
                        created.
                xlabel, ylabel: str
                    Optional. In case not specified the values given
                        as paramters in plot_geometry_2d will be used.
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
        '''

        if (kwargs is not None) and kwargs_2: # check if kwargs_2 is empty
            kwargs.update(kwargs_2)
        kwargs_2 = kwargs

        if ax_val is None:
            fig = plt.figure(figsize=figsize)
            ax_val = fig.add_subplot(111)
            plt.title('{} = {}'.format(direction_label[direction], variable))

        img, ax_val = slice_2d(
                    bbox=bbox, functions=functions, fctnames=fctnames,
                    npt=npoints, direction=direction, variable=variable,
                    ax=ax_val, plot=plot, **kwargs_2)
        colobar = plt.colorbar(img, ax=ax_val)

        if colorbar_label is not None:
            colobar.set_label(colorbar_label)
        if xlabel is not None:
            ax_val.set_xlabel(xlabel)
        if ylabel is not None:
            ax_val.set_ylabel(ylabel)

        return img, ax_val, colobar

    return f


def slice_2d(bbox, functions, npt=(None, None), direction=2,
                      variable=None, cmap='Set1', ax=None, plot='Value',
                      fctnames=None, **kwargs):
    '''
    Returns a function that accept as parameter a point along
        "direction" (c.f. Parameters) and that returns a cut of the
        values or geometry defined by "functions" along that direction
        at that point.

    Parameters:

        bbox = [1.min, 1.max, 2.min, 2.max]. Bounding box along directions
            the two axis that will not be kept constant.
        direction = int. Specifies the axis that is kept constant.
                0 -> first column in the coordinates numpy array, i.e.
                    axis 'x',
                1 - second column ..., i.e. axis 'y'
                2 - third column ..., i.e. axis 'z'
        npt =  tuple -> (100, 100).
                Specifies the number of points in the imshow that
                will be generated.

                plot: string. Can be either 'Values' or 'Geometry'.
        cmap: str.
            c.f. https://matplotlib.org/examples/color/colormaps_reference.html
        ax: ax class from matplotlib.pyplot.
        plot = str.
            'Value' or 'Geometry'.
        functions = list of functions.
            If plot is 'Value':
                Each function returns two numpy arrays, one with
                dtype boolean and the other with a float and has as
                parameters a numpy array that corresponds to the
                coordinates of a set of points. The boolean np array indicates
                if the point is to be considered within the shape
                defined by the function and the second numpy array gives
                the value associated to that point.

            If plot is 'Geometry' then only a numpy array of booleans is
            returned.
        fctnames = list of strings. Names the functions in the previous
                parameter.
    Example:
        See plot_geometry_2d and plot_values_2d, call_slice_2d
    '''

    if isinstance(npt, (tuple, list)):
        if len(npt) == 2:
            npt1, npt2 = npt
        else:
            raise ValueError('npt must have two elements')
    else:
        raise ValueError('npt must be a list')

    if ax is None:
        raise ValueError('No axis is defined')

    if variable is None:
        raise ValueError('Variable is None. No point along {0} was chosen.'.format(
                direction))

    if fctnames is None:
        fctnames = np.arange(1, len(functions) + 1)

    if isinstance(cmap, str):
        cmap = cm.ScalarMappable(cmap=cmap).cmap
    elif not callable(cmap):
        raise ValueError('cmap must be callable or a string')

    cmap_nan_color = kwargs.get('cmap_nan_color', 'k')
    cmap_nan_alpha = kwargs.get('cmap_nan_alpha', 0.5)
#    cmap.set_bad(color=cmap_nan_color, alpha=cmap_nan_alpha)

    # Creat a 2D mesh
    meshplot = grid._2d_mesh(np.linspace(bbox[0], bbox[1], npt1),
                                    np.linspace(bbox[2], bbox[3], npt2))
    # Add third coordinate to 2D mesh.
    meshplot = np.hstack((meshplot,
                          np.ones(npt1 * npt2)[:, None] * variable))
    meshplot = np.roll(meshplot, 2 - direction)

    if plot is 'Value':
        values = [[] for i in range(npt2)]

        for i in range(npt2):
            val_tmp = np.zeros(npt1)

            for f in functions:
                # Try:except So it can do a "2D" slice of a "2D" problem
                # I can't know in advance if the function will only accep
                # a 2d or 3d numpy array
                try:
                    boolean, val = f(meshplot[i * npt1 : (i + 1) * npt1])
                except ValueError:
                    boolean, val = f(meshplot[i * npt1 : (i + 1) * npt1, 0:2])

                val_tmp[boolean] = val[boolean]

            values[i] = val_tmp

    elif plot is 'Geometry':

        # To add the labels to each function in the imshow.
        # TODO: better solution
        for i, f in enumerate(functions):
            ax.plot(0, 0, c=cmap(np.linspace(0, 1, len(functions) + 1)[i+1]),
                     label=fctnames[i], lw=10)

        try :
            values = sum(
                    [(i + 1) * f(meshplot) for i, f in enumerate(functions)]
                    ).reshape(npt2, npt1)
        except (AttributeError, ValueError):
            values = sum([(i + 1) * f(meshplot[:, 0:2])
                          for i, f in enumerate(functions)]
                        ).reshape(npt2, npt1)

        kwargs.update({'vmin':0, 'vmax':len(functions)})
        ax.legend()

    else:
        raise ValueError('plot must be either "Value" or "Geometry"')

    img = ax.imshow(values, extent=bbox, cmap=cmap, origin='lower', **kwargs)
    return img, ax
