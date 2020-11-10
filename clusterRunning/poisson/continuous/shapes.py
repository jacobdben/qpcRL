'''
    ABC class for shape definition.
    Geometrical engine

    TODO: Add the tests that pacome made for geometry_tooblox.
          Check thoses tests to see if any problem exists.
'''

__all__ = ['aspiecewise',
           'asshape',
           'get_bbox',
           'Shape',
           'PiecewiseFunction',
           'General',
           'Rectangle',
           'Ellipsoid',
           'InHull',
           'Delaunay',
           'RegularPolygon',
           'ExtrudedRegularPolygon']


from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import Delaunay as DelaunayScipySpatial
from scipy.spatial import qhull

import matplotlib.pyplot as plt
from poisson import continuous
from .grid import GridBuilder


def get_bbox(object_):
    '''
    Uses a convex hull to get the bounding box of the hull made by
    list of points.

    Parameters:
    -----------
        object_: scipy.spatial.Deulaunay;
                 numpy array containing the coordinates of points;
                 scipy.spatial.qhull.ConvewHull

    '''
    if isinstance(object_, InHull):
        hull = qhull.ConvexHull(object_.points_coordinates)
    elif not isinstance(object_, qhull.ConvexHull):
        hull = qhull.ConvexHull(object_)
    else:
        hull=object_
    return np.array(list(zip(hull.min_bound, hull.max_bound))).flatten()


def __convert_to_class(input_data, convert_into='General'):
    '''
        Converts elements in input_data to a class of
        convert_to type.

        input_data elements depend on which class they will
        be converted to. See parameters of each class for more information

        Parameters:
        -----------
            input_data: list, tuple or dict.
            c.f. parameters in the class that has the name
            given in convert_to

            convert_into: str. Default is 'General'
                Can be 'General' or 'PiecewiseFunction'.

    '''
    call_class = {'General': General,
                  'PiecewiseFunction': PiecewiseFunction}

    if isinstance(input_data, dict):
        input_data.update({
                key: val for key, val in zip(
                        input_data.keys(),
                        __convert_to_class(list(input_data.values()),
                                           convert_into=convert_into))})

    elif isinstance(input_data, (list, tuple)):
        for pos, data_el in enumerate(input_data):
            if not isinstance(data_el, call_class[convert_into]):
                input_data[pos] = call_class[convert_into](data_el)
    else:
        raise TypeError('input_data can"t be of type: {}'.format(
                str(type(input_data))))
    return input_data


def asshape(functions):
    '''
        Transform [function, ...], (function, ...) or {key: function, ...}
            elemnts into shape.General

        Parameters:
        -----------
        functions: list or dict
                Each element must be a function as required by General
        Returns:
        --------
            If element is a:
                function, then:
                    -> replaces the element by a shapes.General object
            else:
                Does nothing to the element in question

    '''
    return __convert_to_class(functions, convert_into='General')


def aspiecewise(input_data):
    '''
        Transform [(function, value), ((function, value), (function, value)),
                  ...]
              or  ((function, value), ((function, value), (function, value)),
                  ...)
              or {key: (function, value), key_2: ((function, value),
                                                 (function, value)),
                 ...}
         elements into shapes.PiecewiseFunction

        Parameters:
        -----------
        input_data: list, tuple or dict
            If each element is a tuple that has as elements:
                   -> tuples containing:
                        1. function (such as the one defined
                            for Shape.geometry(x)) or shapes.Shape class.
                        2. constant or a function that can be evaluated
                            in a set of coordinates x (np.ndarray) returing
                            a number.

                        e.g. input_data[i][0] = (func, constant)
                             input_data[i][0] = (func, func)
                             input_data[i][0] = (shapes.Shape, constant)
                             input_data[i][0] = (shapes.Shape, func)

                    -> two elements:
                        input_data[i][0]: function (such as the one defined for
                            Shape.geometry(x)) or shapes.Shape class as
                            the first element

                        input_data[i][1]: constant or a function that can be
                            evaluated in a set of coordinates x (np.ndarray)
                            returing a number.
                        (c.f it is the value parameter in shapes.WrapperShape)

        Returns:
        --------
            If element is a:
                tuple, then:
                    -> replaces the element by a shapes.PiecewiseFunction class
            else:
                Does nothing to the element in question
        '''
    return __convert_to_class(input_data, convert_into='PiecewiseFunction')


def _new_value_func(new_func, shape, other_shape, operation):
    def func(x):

        values = np.zeros(x.shape[0], dtype=float)
        bool_new_func = new_func(x)

        if operation is 'sub':
            bool_self, val_self = shape.geometry(x[bool_new_func])
            values_new = np.ones(len(bool_self)) * np.nan
            values_new[bool_self] = val_self[bool_self]

        else:
            bool_other, values_other = other_shape(x[bool_new_func])
            values_new = np.ones(len(bool_other)) * np.nan
            values_new[bool_other] = values_other[bool_other]

            bool_self, val_self = shape.geometry(x[bool_new_func])
            values_new[bool_self] = val_self[bool_self]

        values[bool_new_func] = values_new
        values[np.logical_not(bool_new_func)] = np.nan

        return values
    return func


def _rotation_matrix(angles, axis=None):
    '''internal function to calculate the 3d rotation matrix
    either as product of rotation along x, y, and z (angle tx, ty, tz)
    or as rotation of tx around axis'''

    if isinstance(angles, (int, float)):
        angles = [angles, 0, 0]
    elif len(angles) == 2:
        angles = [*angles, 0]

    if axis is not None:
        cost = np.cos(angles[0])
        sint = np.sin(angles[0])
        axis = np.asarray(axis)
        if len(axis) == 3:
            ux, uy, uz = np.asarray(axis)
            rotmat = np.array([
                [cost + ux**2*(1-cost), ux*uy*(1-cost) - uz*sint,
                 ux*uz*(1-cost) + uy*sint],
                [uy*ux*(1-cost) + uz*sint, cost + uy**2*(1-cost),
                 uy*uz*(1-cost) - ux*sint]
                [uz*ux*(1-cost) - uy*sint, uz*uy*(1-cost) + ux*sint,
                 cost + uz**2*(1-cost)]])
            return rotmat
        else:
            print('Axis should be a 3d vector')
            raise ValueError

    else:
        rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])

        ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
        return rx @ ry @ rz


class Shape(ABC):
    '''
        ABC class overloading +, -, &, ^ and | operators
        TODO: when assembling two subclasses of
            type WrapperShape check for intersections  ?
    '''

    version = '0.1'
    print('Shape version: {}'.format(version))

    def __init__(self):
        super().__init__()

    @abstractmethod
    def geometry(self, x):
        '''
            Defines a:
            Function containing the information of the Shape.
            It returns True if the point is within the shape defined
            by the function or False otherwise
        '''
        return None

    def __call__(self, x=None):
        '''
            If a numpy array x is given, returns a boolean vector.
        '''
        if x is not None:
            return self.geometry(x)

    def __add__(self, other_shape):
        '''
            Adds two shapes together.
            Same as union.

            Param:
            ------
            other_shape must be a subclass of Shapes

            Returns:
            -------
            shape.General class
        '''

        if not isinstance(other_shape, Shape):
            raise ValueError('Sum operation only allowed between subclasses\
                             of Shape. ')

        def new_func(x):
            return np.any([self.geometry(x), other_shape(x)], axis=0)
        return General(func=new_func)

    def __or__(self, other_shape):
        '''
            Union of two shapes together.
            Same as sum.
            other_shape must be a subclass of Shape
       '''

        if not isinstance(other_shape, Shape):
            raise ValueError('Union operation only allowed between subclasses\
                              of Shape. ')

        def new_func(x):
            return np.logical_or(self.geometry(x), other_shape(x))
        return General(func=new_func)

    def __and__(self, other_shape):
        '''
            Intersection of two shapes together.
            other_shape must be a subclass of Shape
        '''

        if not isinstance(other_shape, Shape):
            raise ValueError('Intersection operation only allowed between \
                             subclasses of Shape. ')

        def new_func(x):
            return np.logical_and(self.geometry(x), other_shape(x))
        return General(func=new_func)

    def __sub__(self, other_shape):
        '''
            Difference of two shapes together.
            Same as xor
            other_shape must be a subclass of Shape
        '''

        if not isinstance(other_shape, Shape):
            raise ValueError('Difference operation only allowed between \
                             subclasses of Shape. ')

        def new_func(x):
            return self.geometry(x) * np.invert(other_shape(x))

        return General(func=new_func)

    def __xor__(self, other_shape):
        '''
            other_shape must be a subclass of Shape
        '''
        if not isinstance(other_shape, Shape):
            raise ValueError('XOR operation only allowed between \
                             subclasses of Shape. ')

        def new_func(x):
            return np.logical_xor(self.geometry(x), other_shape(x))

        return General(func=new_func)

    def __invert__(self):
        '''
            Not operator
        '''
        def new_func(x):
            return np.logical_not(self.geometry(x))

        return General(func=new_func)

    def translate(self, vect):
        '''
        Translate a Shapes instance

        Parameters:
        -----------
        vect: list, tuple or numpy array
            the translation vector
        '''

        vect = np.asarray(vect)
        func = self.geometry

        def newfunc(x):
            return func(translate(x, -vect))
        self.geometry = newfunc

    def rotate(self, angle, axis=None, center='self'):
        '''Rotate a Shapes instance in 2 or 3d around a center
        of an angle or a combination of angles or around an axis

        WARNING: Not tested in 3D

        Parameters:
        -----------
        angle: (tx, ty, tz) or number
            tx: number
                angle of rotation.
                if 3d, tx is either the angle around the x axis or
                the angle to rotate around the specified axis
            ty: number
                if axis is None, the angle of the rotation around y
                will be ignored if axis is not None
            tz: number (opt, defaults to None)
                if axis is None, the angle of the rotation around z
                will be ignored if axis is not None
        axis: arraylike (opt, defaults to None)
            if axis is None, rotation around main axes
            if not None, the axis of rotation
        center: None, arraylike or 'self'(opt, defaults to None)
            the center of the rotation
            if None, will be the origin according to the dimension
            if self, will be the average of the points
            (center of regular polygon for 'inplace' rotation)
        '''

        func = self.geometry
        try:
            angle = -angle
        except TypeError:
            angle = -np.asarray(angle)

        def newfunc(x):
            return func(rotate(x, angle, axis=axis, center=center))
        self.geometry = newfunc

    def reflect(self, axis):
        '''
        Parameters:
        -----------
        axis: array like
            2 (3) points that generate the line (plan)
            wrt which to reflect in 2d (3d)
        '''

        func = self.geometry

        def newfunc(x):
            return func(reflect(x, axis))
        self.geometry = newfunc


class PiecewiseFunction(object):
    '''
        When called apllies self.evaluate_coordinates to a vector of
        coordinates. self.evaluate_coordinates returns a boolean indicating
        if the coordinate is within the region defined in self.geometry as
        well as the value associated with that coordinate.
    '''

    def __init__(self, data_tuple=()):
        '''
            Params:
            -------
            data_tuple: tuple
                In data_tuple there are either:
                    -> tuples containing:
                        1. function (such as the one defined
                            for Shape.geometry(x)) or shapes.Shape class.
                        2. constant or a function that can be evaluated
                            in a set of coordinates x (np.ndarray) returing
                            a number.

                        e.g. data_tuple[i] = (func, constant)
                             data_tuple[i] = (func, func)
                             data_tuple[i] = (shapes.Shape, constant)
                             data_tuple[i] = (shapes.Shape, func)

                    -> two elements:
                        data_tuple[0]: function (such as the one defined for
                            Shape.geometry(x)) or shapes.Shape class as
                            the first element

                        data_tuple[1]: constant or a function that can be
                            evaluated in a set of coordinates x (np.ndarray)
                            returing a number.

            TODO: Add more type checks ?

        '''

        if not isinstance(data_tuple[0], (tuple, list)):
            self.data_tuple = (data_tuple, )
        else:
            self.data_tuple = data_tuple

    def evaluate_val(self, value):
        '''
            TODO: Test for int or float ?
        '''
        def eval_(x):
            if callable(value):
                return value(x)
            else:
                return value*np.ones(x.shape[0])
        return eval_

    def shape_function(self):
        '''
            Return a function giving only information about the
            geometry, i.e. concatenates (with add operation) all
            functions giving True or False for a point coordinate within
            the geometry defined by such function.
        '''
        data_tuple = self.data_tuple

        def func(x):
            return np.any(
                    [fun[0](x) for fun in data_tuple], axis=0)
        return func

    def evaluate_coordinates(self, x):
        '''
            Parameters:
            -----------
            x: np.ndarray
                1D, 2D or 3D vector containing points coordinates
        '''
        list_bools = [fun[0](x) for fun in self.data_tuple]
        bools_ = np.any(list_bools, axis=0)

        values = np.ones(x.shape[0], dtype=float) * np.nan
        values_new = np.ones(len(list_bools[0][bools_])) * np.nan

        for list_bools_el, val_fun in zip(
                reversed(list_bools),
                reversed([self.evaluate_val(f[1]) for f in self.data_tuple])):

            values_new[list_bools_el[bools_]] = val_fun(x[bools_])[
                    list_bools_el[bools_]]

        values[bools_] = values_new

        return bools_, values

    def __call__(self, x=None):
        '''
            If a numpy array x is given, returns a boolean vector.
        '''
        if x is not None:
            return self.evaluate_coordinates(x)


class General(Shape):
    '''
        Define a Shape from an input function

        Obs:  This is called directly by Shape abc.
    '''

    def __init__(self, func=None):

        super().__init__()
        self.func=func

    def geometry(self, x):
        return self.func(x)


class Rectangle(General):
    '''
        Define a rectangle
    '''

    def __init__(self, length, corner=None, center=None):
        '''
        Class that defines a rectangular shape.
        When called and a numpy array is given as parameter it returns a boolean
        numpy array of the same length of the latter.
        Returns true if inside an 2d of 3d rectangle

        Parameters:
        -----------
        corner = (x0, y0, z0) : numbers
            the lower left corner of the (if z0 si None will be 2d)
        length = (Lx, Ly, Lz): numbers
            the lengths of the sides
            if Ly (Lz) is None, Ly (Lz) will be equal to Lx
            (usefull to quickly draw cubes)
        center = (x0, y0, z0) : numbers
            overwrites the corner argument
            if center=None the corner will be used
            the lower left corner of the (if z0 si None will be 2d)

        Returns:
        --------
        func: function
            func(x) is True if x in rectangle
        '''

        super().__init__()
        self.length=length
        self.corner=corner
        self.center=center
        self.prepare_param()

    def prepare_param(self):

        if isinstance(self.length, (int, float)):
            self.length = (self.length, self.length)

        self.length = np.asanyarray(self.length)

        if self.corner is None:
            self.corner = np.zeros(self.length.shape)
        elif isinstance(self.corner, int):
            self.corner = self.corner * np.ones(self.length.shape)

        self.corner = np.asanyarray(self.corner)

        if self.center is not None:
            self.center = np.asarray(self.center)

        assert len(self.length) == len(self.corner), (
                'The corner does not have the same'
                + ' size as length / or a corner has'
                + ' been given and length is an int')

    def geometry(self, x):
        '''
            Defines a:
            Function containing the information of the Shape.
            It returns True if the point is within the shape defined
            by the function or False otherwise
        '''
        x = np.asarray(x)
        if len(x.shape) == 1:
            ndim, npt = x.shape[0], -1
            x = x[None, :]
        else:
            npt, ndim = x.shape
        assert (ndim == 2 or ndim == 3)

        if self.center is None:
            isin = np.all(x <= self.corner + self.length, axis=1) \
                 * np.all(x >= self.corner, axis=1)
        else:
            isin = np.all(x <= self.center + self.length/2, axis=1) \
                 * np.all(x >= self.center - self.length/2, axis=1)

        if npt is -1:
            return isin[0]
        else:
            return isin


class Ellipsoid(General):
    '''
        Define a rectangle
    '''

    def __init__(self, radius, center=None):
        '''
        Class that defines a ellipsoid shape.
        When called and a numpy array is given as parameter it retu(),rns a
        boolean numpy array of the same length of the latter.
        Returns true if inside an 2d of 3d ellipsoid

        Parameters:
        -----------
        center = (x0, y0, z0):
            the center of the ellipsoid (if there is no z0 it will be 2d)
        radius = (a, b, c): numbers
            the a, b, and c ellipsoid radius
            if b (c) is None, b (c) will be equal to a
            (usefull to quickly draw circles)
    '''

        super().__init__()
        self.radius = radius
        self.center = center
        self.prepare_param()

    def prepare_param(self):

        if isinstance(self.radius, (int, float)):
            self.radius = (self.radius, self.radius)

        self.radius = np.asanyarray(self.radius)

        if self.center is None:
            self.center = np.zeros(self.radius.shape)

        self.center = np.asanyarray(self.center)

        if len(self.center) != len(self.radius):
            raise AttributeError(
                    'Center has different dimensions than the radius')

    def geometry(self, x):
        '''
            Defines a:
            Function containing the information of the Shape.
            It returns True if the point is within the shape defined
            by the function or False otherwise
        '''

        x = np.asarray(x)
        axis = 0
        if len(x.shape) > 1:
            axis = 1  # vectorized

        if ((self.radius.shape[0] != x.shape[1 % len(x.shape)]) or
            (self.center.shape[0] != x.shape[1 % len(x.shape)])):

            raise AttributeError(('Wrong dimensions for radius or'
                                  + ' input points vector x'))

        return ((((x - self.center) / (self.radius)))**2).sum(axis=axis) <= 1


class InHull(Shape):
    '''
        Takes a list of points or a scipy.spatial.Delaunay
        and returns a function that returns True for a point
        within the latter or False otherwise.
        TODO: Change how the modules are correlated ?
        TODO: tests
        TODO: Docstring
    '''
    def __init__(self):
        '''
        '''
        super().__init__()
        self.delaunay = None
        self.points_coordinates = None
        self.hull = None
        self.points_coordinates = self.generate_points()
        self.points_operations = [self.generate_points]


    def get_bbox(self):
        '''
            Uses a convex hull to get the bounding box of the hull made by
            list of points.
        '''
        self.hull = qhull.ConvexHull(self.points_coordinates)
        return get_bbox(self.hull)

    def geometry(self, x):
        '''
            From stackoverflow
            Test if points in `p` are in `delaunay tesselation`
            `p` should be a `NxK` coordinates of `N` points in `K`
            dimensions
        '''
        if self.delaunay is None:
            self.prepare_param()

        if self.delaunay is not None:
            return self.delaunay.find_simplex(x) >= 0
        else:
            raise ValueError('InHull.hull has not been defined')

    def make_delaunay(self):
        '''
            Make Delaunay tessalatioon.
        '''
        if self.points_coordinates is not None:
            self.delaunay = DelaunayScipySpatial(self.points_coordinates)

    def prepare_param(self):
        '''
            Calculate the Deulaynay triangulation
            if a set of points is given.
        '''
        if self.delaunay is None:
            if self.points_coordinates is None:
                self.points_coordinates = self.generate_points()
            self.make_delaunay()

    @abstractmethod
    def generate_points(self):
        '''
            Define list of points to be passed to hull.find_simplex
            Returns a numpy array containing the coordinates of those points.
        '''
        pass

    def translate(self, vect):
        '''Translate a point or a list of points by a vector vect

            Parameters:
            -----------
            vect: list, tuple or numpy array
        '''
        if self.points_coordinates is None:
            self.points_coordinates = self.generate_points()

        self.points_coordinates = translate(self.points_coordinates, vect)
        self.points_operations.append(
                            lambda : translate(self.points_coordinates, vect))

    def rotate(self, angle, axis=None, center='self'):
        '''Rotate a point or a list of points in 2 or 3d around a center
        of an angle or a combination of angles or around an axis

        Parameters:
        -----------
        angle: (tx, ty, tz) or number
        tx: number
            angle of rotation.
            if 3d, tx is either the angle around the x axis or
            the angle to rotate around the specified axis
        ty: numbernp.repeat([rot_mat], len(x), axis=0)
            if axis is None, the angle of the rotation around y
            will be ignored if axis is not None
        tz: number (opt, defaults to None)
            if axis is None, the angle of the rotation around z
            will be ignored if axis is not None
        axis: arraylike (opt, defaults to None)
            if axis is None, rotation around main axes
            if not None, the axis of rotation
        center: None, arraylike or 'self'(opt, defaults to None)
            the center of the rotation
            if None, will be the origin according to the dimension
            if self, will be the average of the points
            (center of regular polygon for 'inplace' rotation)

        '''
        if self.points_coordinates is None:
            self.points_coordinates = self.generate_points()

        self.points_coordinates = rotate(self.points_coordinates, angle,
                                        axis=axis, center=center)
        self.points_operations.append(
                                lambda : rotate(self.points_coordinates, angle,
                                                axis=axis, center=center))

    def reflect(self, axis):
        '''returns reflections of point(s) x wrt to axis

        Parameters:
        -----------
        axis: array like
            2 (3) points that generate the line (plan)
            wrt which to reflect in 2d (3d)
        '''
        if self.points_coordinates is None:
            self.points_coordinates = self.generate_points()

        self.points_coordinates = reflect(self.points_coordinates, axis)
        self.points_operations.append(lambda : reflect(self.points_coordinates,
                                                      axis))
#####################


def box(length, corner=None, center=None):
    '''
        Returns points of a 2d of 3d box

        Parameters:
        -----------
        length = (Lx, Ly, Lz): numbers
            the lengths of the sides
            if Ly (Lz) is None, Ly (Lz) will be equal to Lx
            (usefull to quickly draw cubes)

        corner = (x0, y0, z0) : numbers
            the lower left corner of the (if z0 si None will be 2d)

        center = (x0, y0, z0) : numbers
            overwrites the corner argument
            if center=None the corner will be used
            the lower left corner of the (if z0 si None will be 2d)

        Returns:
        --------
        points: np.ndarray
            the points defining the box
    '''
    if isinstance(length, (int, float)):
        length = (length, length)

    length = np.asarray(length)
    ndim = len(length)

    if corner is None:
        corner = np.zeros(ndim)
    elif isinstance(corner, int):
        corner = corner * np.ones(ndim)

    corner = np.asarray(corner)

    if center is not None:
        center = np.asarray(center)

    assert len(length) == len(corner), (
            'The corner does not have the same'
            + ' size as length / or a corner has'
            + ' been given and length is an int')

    points = np.array([[0, 0], [0, length[1]], length[:2], [length[0], 0]])
    if ndim == 3:
        points = np.vstack((np.c_[points, np.zeros(4)],
                            np.c_[points, np.ones(4) * length[2]]))

    if center is None:
        points = points + corner
    else:
        points = points - length / 2 + center

    return points


def extrude(x, newcoord, force1d=False):
    '''return x with value newcoord in an added dimension

    Parameters:
    -----------
    x: array like (npt, ndim)
        points or list of points in 1d or 2d
        if 1d: use force1d=True
    newcoord: array like
        value of array of the values
        of the expected new coordinate
    force1d: bool (defaults to False)
        to make the difference between one point in 2d
        and a list of 1d points, force1d needs to be put
        to True in the 1d case

    Returns:
    --------
    newx: array like (npt * len(newcoord), ndim + 1)
        the new extruded array
    '''
    newcoord = np.asarray(newcoord)
    x = np.asarray(x)

    if len(x.shape) <= 1:
        if force1d:
            ndim = -1
            npt = len(x)
        else:
            ndim, npt = x.shape[0], 1
        x = x[None, :]
    else:
        npt, ndim = x.shape

    if ndim is -1:
        newx = np.tile(x, len(newcoord)).T
    else:
        newx = np.tile(x, (len(newcoord), 1))

    newcoord = np.repeat(newcoord, npt, axis=0)
    newx = np.c_[newx, newcoord]

    return newx

def inflate(x, value, direction='all', center='self'):
    '''add or substract value to list of points x
        depending depending if x[i] < or > center[i]
        for chosen i

    Notes:
    ------
    This is usefull for example
    if ones wants to create increase slightly the size of
    a polygon (polyhedron)

    Parameters:
    -----------
    x: 2d array
        the list of points to inflate
    value: number, list
        the value of which to inflate x
        in the chosen direction
        if it is a number, will inflate
        in all directions in direction param
    direction: list (opt, defaults to 'all')
        the direction in which to inflate x
        if all: will infalte all directions
        otherwhise should match shape of x
    center: list of coordinates (opt, defaults to 'self)
        the reference wrt to which to inflate.
        x[: i] < center[i] -> newx[i] = x[i] - value[i]
        if 'self', center will be mean(x)

    Returns:
    --------
    newx: 2d array
        the list of inflated coordinates
    '''
    x = np.asarray(x)
    if len(x.shape) == 2:
        npt, ndim = x.shape
    else:
        assert False, 'x should be a 2d array'
    assert (ndim == 2 or ndim == 3)

    if direction is 'all':
        direction = np.arange(ndim)

    try:
        assert len(value) == len(direction)
    except TypeError:
        tmpvalue = np.zeros(ndim)
        tmpvalue[direction] = np.ones(len(direction)) * value
        value = tmpvalue

    if center is 'self':
        center = np.mean(x, axis=0)

    newx = x.copy()
    for i in direction:
        tmp = ((x[:, i] + value[i]) * (x[:, i] > center[i])
                   + (x[:, i] - value[i]) * (x[:, i] < center[i])
                   + (x[:, i]) * (x[:, i] == center[i]))
        newx[:, i] = tmp

    return newx


def translate(x, vect):
    '''
    Translate a point or a list of points by a vector vect

    Parameters:
    -----------
    x: tuple or liste of tuple
        the points to translate
    vect: list, tuple or numpy array
        the translation vector
    '''
    x = np.asarray(x)
    vect = np.asarray(vect)

    if len(x.shape) < 2:
        # translate a point
        return x + np.asarray(vect)

    elif len(x.shape) == 2:
        return np.repeat([vect], len(x), axis=0) + x


def rotate(x, angle, axis=None, center='self'):
    '''Rotate a point or a list of points in 2 or 3d around a center
    of an angle or a combination of angles or around an axis

    Parameters:
    -----------
    angle: (tx, ty, tz) or number
        tx: number
            angle of rotation.
            if 3d, tx is either the angle around the x axis or
            the angle to rotate around the specified axis
        ty: number
            if axis is None, the angle of the rotation around y
            will be ignored if axis is not None
        tz: number (opt, defaults to None)
            if axis is None, the angle of the rotation around z
            will be ignored if axis is not None
    axis: arraylike (opt, defaults to None)
        if axis is None, rotation around main axes
        if not None, the axis of rotation
    center: None, arraylike or 'self'(opt, defaults to None)
        the center of the rotation
        if None, will be the origin according to the dimension
        if self, will be the average of the points
        (center of regular polygon for 'inplace' rotation)
    '''

    x = np.asarray(x)
    center_parent = center

    if not isinstance(angle, (list, tuple)):
        angle = (angle,)
    angle = np.asarray(angle)

    if len(x.shape) == 2:
        npt, ndim = x.shape
    elif len(x.shape) == 1:
        npt, ndim = -1, x.shape[0]
        x = x[None, :]
    else: raise ValueError

    if ndim is 2:
        rot_mat = np.array([[np.cos(angle[0]), -np.sin(angle[0])],
                            [np.sin(angle[0]), np.cos(angle[0])]])
    elif ndim is 3:
        rot_mat = _rotation_matrix(angle, axis)
    else: raise ValueError

    if center_parent is None:
        center = np.zeros(ndim)
    elif center_parent is 'self':
        center = np.mean(x, axis=0)
    elif isinstance(center_parent, (tuple, list, np.ndarray)):
        center = np.asarray(center_parent)
    else: raise ValueError

    center_mat = np.repeat([center],
                           len(x), axis=0)

    ret = (np.einsum(
            'kij, kj -> ki',
            np.repeat([rot_mat], len(x), axis=0), x - center_mat)
           + center_mat)

    if npt is -1:
        return ret[0]
    else:
        return ret


def reflect(x, axis):
    '''returns reflections of point(s) x wrt to axis

    Parameters:
    -----------
    x: array like
        point or list of points in 2 or 3d
    axis: array like
        2 (3) points that generate the line (plan)
        wrt which to reflect in 2d (3d)

    Returns:
    --------
    ret: array like (shape of x)
        the reflected point(s)
    '''

    x = np.asarray(x)
    if len(x.shape) == 1:
        ndim, npt = x.shape[0], -1
        x = x[None, :]
    else:
        npt, ndim = x.shape
    assert (ndim == 2 or ndim == 3)

    assert isinstance(axis, (list, np.ndarray))
    axis = np.asarray(axis)
    assert axis.shape[1] == ndim
    A = axis[0]
    Amat = np.repeat([A], len(x), axis=0)

    if ndim == 2:
        u = (axis[1:] - A)[0]
        v = np.array([u[1], -u[0]])
        # u, v is othogonal basis
        mirror = np.array([[1, 0],[0, -1]])
        Mat = np.array([u, v])

    elif ndim == 3:
        u, C = axis[1:] - A
        w = np.cross(u, C)
        v = np.cross(w, u)
        # u, v, w is othogonal basis
        mirror = np.array([[1, 0, 0],[0, 1, 0], [0 ,0 ,-1]])
        Mat = np.array([u, v, w])

    Matinv = np.linalg.inv(Mat)
    transf = Mat.T @ mirror @ Matinv.T
    ret = np.einsum('kij, kj -> ki', np.repeat([transf], len(x), axis=0),
                                    (x - Amat)) + Amat
    if npt is -1:
        return ret[0]
    else:
        return ret

#################

class Delaunay(InHull):
    '''
        Generate a shape from a list of points.
        From those points a Delaunay triangulation is made
        (scipy.spatial.qhull.Delaunay instance).
        From the latter a function returning True if a set of coordinates
        is within the region defined by the Delaunay triangulation is given.

        It also accepts a scipy.spatial.qhull.Delaunay instance
    '''

    def __init__(self, list_points=None, delaunay_instance=None):
        '''
            Parameters:
            -----------

                list_points: numpy array
                    Coordinates of the vertex passed to
                    scipy.spatial.qhull.Delaunay

                delaunay_instance: scipy.spatial.qhull.Delaunay instance
        '''
        super().__init__()
        if list_points is not None:
            self.points_coordinates = np.asarray(list_points)
        elif delaunay_instance is not None:
            self.delaunay = delaunay_instance
        else:
            raise ValueError((' Either list_points or delaunay_instance must'
                              + 'be defiend'))

    def generate_points(self):
        return None


class RegularPolygon(InHull):
    '''
        Define a regular polygon
    '''

    def __init__(self, npoints=3, center=(0, 0),
                 radius=0.5, clockwise=False, flat=True):
        '''
        Class that defines a polygon shape by rotation around a center.
        When called and a numpy array is given as parameter it returns a
        boolean numpy array of the same length of the latter.
        Returns true if inside an 2d of 3d polygon

        Parameters:
        -----------
        npoints: integer (opt, defaults to 3)
            the number of sides (points) of the regular polygon
        center: 2d tuple (opt, defaults to [0, 0])
            the center of the regular polygon
        radius: number (opt, defaults to 0.5)
            the half diagonal of the regular polygon
        clockwise: bool (opt, defaults to False)
            if True the polygon is constructed clockwise,
            otherwise counter-clockwise
        flat: bool (opt, defaults to True)
            it True, the polygon is rotated to have his lowest
            edge parallel to x axis
        '''

        self.radius = radius
        self.npoints = npoints
        self.center = center
        self.clockwise = clockwise
        self.flat = flat
        super().__init__()


    def generate_points(self):
        '''
            Returns a numpy array containing the coordinates of the points
            defining the shape.
        '''
        sector = (2. * np.pi) / self.npoints
        clk = 1
        offset = 0
        if self.clockwise: clk = -1
        if self.flat:
            if self.npoints % 2 == 0:
                offset = np.pi / 2. * (1 - 2. / (self.npoints
                                                 * (2 * int(self.npoints
                                                            / 4.) + 1)))
            else:
                offset = np.pi / 2. * (1 - 4 / self.npoints * int(self.npoints
                                                                  / 4.))

        poly = [[(self.center[0] + self.radius
                  * np.cos(offset + clk * sector * point_number)),
                 (self.center[1] + self.radius
                  * np.sin(offset + clk * sector * point_number))]
                for point_number in range(self.npoints)]

        return np.array(poly)


class ExtrudedRegularPolygon(InHull):
    '''
        Creates polygon by rotation around a center.
        Extruded along z.
    '''

    def __init__(self, npoints=3, center=(0, 0, 0),
                 radius=0.5, clockwise=False, flat=True, Lz=1):
        '''
        Class that defines a polygon shape by rotation around a center.
        When called and a numpy array is given as parameter it returns a
        boolean numpy array of the same length of the latter.
        Returns true if inside an 2d of 3d polygon

        Parameters:
        -----------
        n: integer (opt, defaults to 3)
            the number of sides (points) of the regular polygon
        center: tuple (opt, defaults to (0, 0, 0))
            the center of the regular polygon
        Lz: number (opt, defaults to 1)
            the length of the extrusion
            the polygon will be between center[2] and center[2] + Lz
        radius: number (opt, defaults to 0.5)
            the half diagonal of the regular polygon
        clockwise: bool (opt, defaults to False)
            if True the polygon is constructed clockwise, otherwise
            counter-clockwise
        flat: bool (opt, defaults to True)
            it True, the polygon is rotated to have his lowest edge parallel
            to x axis


        '''
        self.radius = radius
        self.npoints = npoints
        self.center = center
        self.clockwise = clockwise
        self.flat = flat
        self.Lz = Lz
        super().__init__()

    def generate_points(self):
        '''
            Defines a self.points_coordinates.
            A numpy array containing the coordinates of the points
            defining the shape.
        '''

        sector = (2. * np.pi) / self.npoints
        clk = 1
        offset = 0
        if self.clockwise: clk = -1
        if self.flat:
            if self.npoints % 2 == 0:
                offset = np.pi / 2. * (1 - 2. / (self.npoints
                                                 * (2 * int(self.npoints
                                                            / 4.) + 1)))
            else:
                offset = np.pi / 2. * (1 - 4 / self.npoints * int(self.npoints
                                                                  / 4.))

        poly = [[(self.center[0] + self.radius
                  * np.cos(offset + clk * sector * point_number)),
                 (self.center[1] + self.radius
                  * np.sin(offset + clk * sector * point_number))]
                for point_number in range(self.npoints)]

        return np.concatenate((
                np.c_[poly, np.ones(self.npoints) * self.center[2]],
                np.c_[poly, np.ones(self.npoints) * self.center[2] + self.Lz]))


##########################################################
# test plots

def __test_plot_ellipse():
    ellipse = Ellipsoid(radius=2)
    mm = continuous.grid._2d_mesh(
            *continuous.grid._points_from_2dbounding_box([-3, 3, -3, 3], 0.1))

    x, y = list(zip(*mm))
    xx, yy = list(zip(*continuous.grid._apply_stencil(mm, ellipse)))
    plt.scatter(x, y, label='bbox')
    plt.scatter(xx, yy, c='r', label='ellipse')
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.show()


def __test_plot_rectangle():
    rect1 = Rectangle(length=4)
    rect2 = Rectangle(length=2, corner=[1, 1])
    rect3 = Rectangle(length=1, corner=[1.5, 1.5])

    bbox1 = [0, 4, 0, 4]
    bbox2 = [1, 3, 1, 3]

    mesh1 = (bbox1, 0.15, rect1, 0)
    mesh2 = (bbox2, 0.1, rect2, 1)
    hole1 = (rect3)
    points = ([[2, 2]], 2)

    mesh = GridBuilder(meshs=[mesh1, mesh2], holes=[hole1],
                                  points=[points])

    x, y = list(zip(*mesh.points))
    plt.scatter(x, y, c=mesh.point_label)
    plt.show()

def __test_plot_rectangle_center():
    rect1 = Rectangle(length=4, center=[2, 2])
    rect2 = Rectangle(length=2, center=[2, 2])
    rect3 = Rectangle(length=1, center=[2, 2])

    bbox1 = [0, 4, 0, 4]
    bbox2 = [1, 3, 1, 3]

    mesh1 = (bbox1, 0.15, rect1, 0)
    mesh2 = (bbox2, 0.1, rect2, 1)
    hole1 = (rect3)
    points = ([[2, 2]], 2)

    mesh = GridBuilder(meshs=[mesh1, mesh2], holes=[hole1],
                                  points=[points])

    x, y = list(zip(*mesh.points))
    plt.scatter(x, y, c=mesh.point_label)
    plt.show()


def __test_plot_ellipse_circ():

    ellipse = Ellipsoid(radius=2)

    rlist = np.linspace(0, 5, 100)
    tlist = np.linspace(0, 2*3.14, 100)

    mm = continuous.grid._mesh_polar(rlist, tlist)
    xx, yy = list(zip(*continuous.grid._apply_stencil(mm, ellipse)))
    x, y = list(zip(*mm))
    plt.scatter(x, y, label='bbox')
    plt.scatter(xx, yy, c='r', label='ellipse')
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.show()

def __test_plot_ellipse_circ_class():

    ellipse = Ellipsoid(radius=2)

    mesh_obj = GridBuilder(build_mesh=False)

    circularbox = [0, 5, 0, 2*np.pi]
    alpha = [0.1, 0.1]

    # manually add circular mesh
    mesh_obj.add_mesh_circular(circularbox, alpha, ellipse, 0)

    x, y = list(zip(*mesh_obj.points))
    plt.scatter(x, y, label='bbox')
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.show()


def __test_plot_rotation_translation():

    poly = RegularPolygon(npoints=4, center=(0, 0))

    polyt = RegularPolygon(npoints=4, center=(0, 0))
    polyt.translate([0.2, 1])

    polyr = RegularPolygon(npoints=4, center=(0, 0))
    polyr.translate([-1, -0.2])
    polyr.rotate(np.pi/4, center='self')

    mesh = GridBuilder(meshs=[[get_bbox(poly), 0.01, poly, 0],
                         [get_bbox(polyt), 0.01, polyt, 1],
                         [get_bbox(polyr), 0.01, polyr, 2]])

    poly_2 = RegularPolygon(npoints=4, center=(0, 0))
    poly_2.translate([-1, 1])
    poly_2.rotate(np.pi/4, center='self')
    mesh2 = GridBuilder(meshs=[[get_bbox(poly_2), 0.01,
                                  poly_2, 0]]).points

    plt.scatter(*mesh.points[mesh.point_label == 0].T, c='b',
                label='original')
    plt.scatter(*mesh.points[mesh.point_label == 1].T, c='r',
                label='translated')
    plt.scatter(*mesh.points[mesh.point_label == 2].T, c='g',
                label='rotated before mesh')
    plt.scatter(*mesh2.T, c='y', label='rotated after mesh')
    plt.legend()
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

def __test_reflection():
    '''test if two reflexions is equal to identity'''
    np.random.seed(112)

    ndim = 2
    axis = np.random.rand(ndim, ndim)
    pt = np.random.rand(150, ndim)
    assert(np.allclose(pt, reflect(reflect(pt, axis), axis)))

    ndim = 3
    axis = np.random.rand(ndim, ndim)
    pt = np.random.rand(150, ndim)
    assert(np.allclose(pt, reflect(reflect(pt, axis), axis)))

    print('test reflexion passed')

def __test_plot_reflection():
    ndim = 2
    axis = np.random.rand(ndim, ndim) - 0.5
    pt = np.random.rand(10, ndim)
    newpt = reflect(pt, axis)

    plt.figure(figsize=(12,12))
    plt.plot(*pt.T, marker='o')
    plt.plot(*axis.T, lw=4)
    plt.plot(*newpt.T, marker='^')
    plt.axes().set_aspect('equal')
    plt.show()

#__test_plot_ellipse()
#__test_plot_rectangle()
#__test_plot_ellipse_circ()
#__test_plot_ellipse_circ_class()
#__test_plot_rotation_translation()
#__test_reflection()
#__test_plot_reflection()
