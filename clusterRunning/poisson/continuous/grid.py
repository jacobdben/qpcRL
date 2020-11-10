'''
 Author : Pacôme
 (minor contributions Antonio)
 TODO : Improve eval_mesher
'''

__all__ = ['GridBuilder']

import itertools

import numpy as np


def eval_vect(f, x):
    '''small function that tries to evaluate f(x) vectorialy
    and uses a map if it fails'''
    try:
        ret = f(np.asarray(x))
        if len(ret) != len(x):
            raise TypeError
        else:
            return ret
    except TypeError:
        return np.array(list(map(f, x)))


def _2d_mesh(xlist, ylist):
    '''simple wrapper on meshgrid,
    returns list of the mesh points created by xlist and ylist'''
    return np.dstack([x.flatten() for x in np.meshgrid(xlist, ylist)])[0]


def _3d_mesh(xlist, ylist, zlist):
    '''simple wrapper on meshgrid,
    returns list of the mesh points created by xlist ylist and zlist'''
    return np.dstack([x.flatten() for x in np.meshgrid(xlist, ylist, zlist)])[0]

def _mesh_line(xmin, xmax, step):
    nx = int((xmax - xmin) / step)
    xlist = (xmin + ((xmax - xmin) - nx * step)
             / 2. + np.arange(nx + 1) * step)
    return xlist

def _points_from_2dbounding_box(bbox, alpha):
    '''returns list of mesh points with step alpha.
    uses _2d_mesh

    Parameters:
    -----------
    bbox: list
        the bounding box, should have the format xmin, xmax, ymin, ymax
    alpha: number
        the mesh step. If non integer number of alpha in (xmin, xmax) or in (ymin, ymax)
        the routine will center the mesh points

    Returns:
    --------
    ret: 2d array
        the mesh points (shape Npoints * dim)
    '''

    if isinstance(alpha, (float, int)):
        alpha = [alpha, alpha]

    assert isinstance(alpha, list), ('Mesh step input is neither a list, nor a'
                                     + 'float nor an int')

    xmin, xmax, ymin, ymax = bbox
    xlist = _mesh_line(xmin, xmax, alpha[0])
    ylist = _mesh_line(ymin, ymax, alpha[1])

    return [xlist, ylist]


def _points_from_3dbounding_box(bbox, alpha):
    '''returns list of mesh points with step alpha.
    uses _2d_mesh

    Parameters:
    -----------
    bbox: list
        the bounding box, should have the format xmin, xmax, ymin, ymax, zmin, zmax
    alpha: number
        the mesh step. If non integer number of alpha in (xmin, xmax), (ymin, ymax), or (zmin, zmax)
        the routine will center the mesh points

    Returns:
    --------
    ret: 2d array
        the mesh points (shape Npoints * dim)
    '''

    if isinstance(alpha, (int, float)):
        alpha = [alpha, alpha, alpha]

    assert isinstance(alpha, list), ('Mesh step input is neither a list, nor a'
                                     + 'float nor an int')


    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    xlist = _mesh_line(xmin, xmax, alpha[0])
    ylist = _mesh_line(ymin, ymax, alpha[1])
    zlist = _mesh_line(zmin, zmax, alpha[2])

    return [xlist, ylist, zlist]


def _mesh_polar(rlist, tlist, zlist=None, center=[0, 0]):
    '''mesh according to radius and theta list
    if one element in radius is < 1.e-14,
        will consider only one point in the center
        instead of all the t.
        this is to prevent multiple definition of points
    removes values of tlist the same with 2pi period
    if zlist and philist are None -> polar
    if zlist not None: check ndim center -> cylinder
    '''
    # remove hypothetical radius to close to 0
    add_center = False
    rlist = np.asarray(rlist)
    if np.any(rlist < 1.e-14):
        rlist = np.delete(rlist, np.where(rlist < 1.e-14)[0])
        add_center = True

    # remove identical tlist modulo 2 pi
    tlist = np.unique(np.mod(tlist, 2. * np.pi))

    if zlist is None:
        # 2d polar

        assert len(center) == 2
        rs, ts = [x.flat for x in np.meshgrid(rlist, tlist)]
        ret = np.array(list(itertools.starmap(
            lambda r, t: [center[0] + r * np.cos(t), center[1] + r * np.sin(t)],
            zip(rs, ts))))
        if add_center:
            ret = np.concatenate(([center], ret))
        return ret

    if center == [0, 0]:
        # convert to 3d
        center = [0, 0, 0]

    assert len(center) == 3

    if zlist is not None:
        # cylindrical

        rs, ts, zs = [x.flat for x in np.meshgrid(rlist, tlist, zlist)]
        ret = np.array(list(itertools.starmap(
            lambda r, t, z: [center[0] + r * np.cos(t),
                             center[1] + r * np.sin(t),
                             center[1] + z],
            zip(rs, ts, zs))))
        if add_center:
            centers = (np.vstack((np.zeros(len(zs)), np.zeros(len(zs)), zs)).T
                      + np.repeat([center], len(zs), axis=0))
            ret = np.concatenate((centers, ret))
        return ret

    else:
        raise ValueError


def _mesh_spherical(rlist, tlist, philist, center=[0, 0, 0]):
    '''mesh according to radius, theta and plist list
        '''
    # remove hypothetical radius to close to 0
    # remove phi = 0 modulo np.pi
    add_center = False
    rlist = np.asarray(rlist)
    if np.any(rlist < 1.e-14):
        rlist = np.delete(rlist, np.where(rlist < 1.e-14)[0])
        add_center = True

    # remove identical tlist modulo 2 pi
    tlist = np.unique(np.mod(tlist, 2. * np.pi))
    philist = np.unique(np.mod(philist, np.pi))
    philist = np.delete(philist, np.where(philist == 0.))
    rs, ts, ps = [x.flat for x in np.meshgrid(rlist, tlist, philist)]


    ret = np.array(list(itertools.starmap(
        lambda r, t, p: [center[0] + r * np.cos(t) * np.sin(p),
                         center[1] + r * np.sin(t) * np.sin(p),
                         center[2] + r * np.cos(p)],
        zip(rs, ts, ps))))
    if add_center:
        ret = np.concatenate(([center], ret))
    return ret


def mesh_from_spherical_box(sphericalbox, alpha, center=[0, 0, 0]):
    '''Wrapper of _mesh_spherical'''

    assert len(alpha) == len(sphericalbox)/2, ('Circularbox must have same number'
                                              + 'of inputs as the number of steps')

    if len(alpha) == 3: # seems unecessary or should be merged with the previous
        if center == [0, 0]:
            center = [0, 0, 0]

        assert len(center) == 3, ('Not enough center points when defining'
                                  + ' spherical mesh')

        return _mesh_spherical(*_points_from_3dbounding_box(sphericalbox, alpha),
                           center=center)


def mesh_from_circular_box(circularbox, alpha, center=[0, 0]):
    '''Wrapper of _mesh_polar'''

    assert len(alpha) == len(circularbox)/2, ('Circularbox must have same number'
                                              + 'of inputs as the number of steps')
    if center is None:
        center = [0, 0]

    if len(alpha) == 2:

        assert len(center) == 2, ('Not enough center points when defining'
                                  + ' circular mesh')

        return _mesh_polar(*_points_from_2dbounding_box(circularbox, alpha),
                           center=center)
    if len(alpha) == 3:

        if center == [0, 0]:
            center = [0, 0, 0]

        assert len(center) == 3, ('Not enough center points when defining'
                                  + ' circular mesh')

        return _mesh_polar(*_points_from_3dbounding_box(circularbox, alpha),
                           center=center)

def mesh_from_square_box(bbox, alpha):
    '''wrapper of _points_from_(2d or 3d)bounding_box'''
    ndim = len(bbox) // 2
    if ndim is 2:
        return _2d_mesh(*_points_from_2dbounding_box(bbox, alpha))
    elif ndim is 3:
        return _3d_mesh(*_points_from_3dbounding_box(bbox, alpha))
    else:
        print('wrong bbox format', ndim)
        raise ValueError

def _apply_stencil(mesh, stencil_func, where=False):
    '''select only the mesh points where stencil_func is True
    Parameters:
    -----------
    mesh: 2d array
        the mesh points (shape Npoints * dim)
    stencil_func: function of x
        the stencil function f(x) is True inside stencil
    where: bool (opt, defaults to False)
        if True, will also return the mask of the mesh points in the stencil

    Returns:
    --------
    ret: 2d array
        the new mesh points
    msk: array (opt)
        if where is True, the mask of the mesh points in the stencil
    '''
    msk = eval_vect(stencil_func, mesh)
    if not where:
        return mesh[msk]
    else:
        return mesh[msk], msk


class GridBuilder(object):
    '''Class GridBuilder, to mesh stuffs
        When called returns self.points
        TODO: Make it lazy ?
    '''

    def __init__(self, meshs=[], holes=[], points=[], build_mesh=True):
        '''
        Parameters:
        -----------
        meshs: list of [bbox, alpha, function, label] (opt, defaults to empty)
            the infos to build meshes from function.
            the structure should be compatible with self.add_mesh_square (see  for details)
        holes: list of functions (opt, defaults to empty)
            fonctions f where f(x) = False inside the hole (will be reversed)
            will call self.remove_points
        points: list of [points, label] (opt, defaults to empty)
            the indpendants points to add to the mesh and theit label
            should be compatible with self.add_points (see for more details)

        Sets:
        -----
        self.points: 2d array (Npts * ndim)
            the list of mesh points
        self.point_label: array
            the list of the corresponding labels

        TODO: change bbox name in meshs
        '''
        self.points = None
        self.point_label = None

        if build_mesh:
            for mesh in meshs:
                self.add_mesh_square(*mesh)
            for hole in holes:
                self.remove_points(hole)
            for point in points:
                self.add_points(*point)

    def __call__(self):
        return self.points

    def remove_points(self, function):
        '''applies a negative stencil to drill hole in meshs
        Parameters:
        -----------
        function: function
            function of x (2d or 3d) that returns True inside the hole to drill
        '''
        # remove existing mesh points in the new stencil
        if self.points is not None:
            notfunction = lambda x: np.invert(function(x))
            self.points, where = _apply_stencil(self.points, notfunction, where=True)
            self.point_label = self.point_label[where]

    def add_points(self, points, label, old_points=None, old_labels=None):
        '''add points to the existing mesh with a given label
        Parameters:
        -----------
        points: list of array_like (length = ndim)
            the points to add to the mesh
        label: str or int
            the label to give to the added mesh points
        old_points: If None uses self.points. The existing list of points
        old_labels: If None uses self.point_label. The existing list of labels

        TODO: change this to make it lazy

        '''
        if old_points is None and old_labels is None:
            old_points = self.points
            old_labels = self.point_label
        else:
            raise ValueError('If points is given, then old_labels must be\
                             aswell')
        if old_points is not None:
            old_points = np.concatenate((old_points, points))
            old_labels = np.concatenate((old_labels,
                                               [label] * len(points)))
        else:
            old_points = points
            old_labels = np.array([label] * len(points))


        # Just temporary, if GridBuilder is to become lazy, self.points
        # will be removed ?
        self.points = old_points
        self.point_label = old_labels
        return old_points, old_labels

    def add_mesh_circular(self, circularbox, alpha, function, label,
                          center=None):
        '''computes a circular mesh from circularbox defining a
        circular box and extracts from it a set of points respecting the
        conditions defined by function. It then and adds those points
        to the mesh points
        Parameters:
        -----------
        circularbox: list
            the bounding box of the function in 2 or 3d.
            format is rmin, rmax, thetamin, thetamax, zmin, zmax
        alpha: list
            the mesh step for each direction.
            format rstep, thetastep, zstep
        function: function of x
            function f(x) that is True inside the shape to mesh
        label: str or int
            the label to give to the added mesh points
        '''
        # remove existing mesh points in the new stencil
        self.remove_points(function)
        # add new mesh points with their label
        self.add_points(
                points=_apply_stencil(
                        mesh_from_circular_box(circularbox, alpha),
                        function, center),
                label=label)

    def add_mesh_spherical(self, sphericalbox, alpha, function, label,
                          center=None):
        '''computes a sphericalbox mesh from circularbox defining a
        circular box and extracts from it a set of points respecting the
        conditions defined by function. It then and adds those points
        to the mesh points
        Parameters:
        -----------
        sphericalbox: list
            the bounding box of the function in 2 or 3d.
            format is rmin, rmax, thetamin, thetamax, zmin, zmax
        alpha: list
            the mesh step for each direction.
            format rstep, thetastep, zstep
        function: function of x
            function f(x) that is True inside the shape to mesh
        label: str or int
            the label to give to the added mesh points
        '''
        # remove existing mesh points in the new stencil
        self.remove_points(function)
        # add new mesh points with their label
        self.add_points(
                points=_apply_stencil(
                        mesh_from_spherical_box(sphericalbox, alpha),
                        function, center),
                label=label)

    def add_mesh_square(self, bbox, alpha, function, label):
        '''computes a mesh from bbox and functurnn 2 or 3d.
            format is xmin, xmax, ymin, ymax, zmin, zmax
        alpha: number
            the mesh step.
        function: function of x
            function f(x) that is True inside the shape to mesh
        label: str or int
            the label to give to the added mesh points
        '''
        # remove existing mesh points in the new stencil
        self.remove_points(function)
        # add new mesh points with their label
        self.add_points(
                points=_apply_stencil(
                        mesh_from_square_box(bbox, alpha), function),
                label=label)
