'''
    Defines the FiniteVolume ABC class that generates an insatnce
    with all nescesary methods and attributes for it to be accepted
    as a mesh in DiscretePoisson.
'''


__all__ = ['FiniteVolume',
           'Voronoi']


from abc import ABC, abstractmethod
import warnings
import itertools
from collections import namedtuple

from scipy.spatial import qhull

import numpy as np

from . import surface_calc as sc
from . import _finite_volume as fvcython


def check_empty_region_vertice(voronoi_obj):
    """Check for empty region vertices in  voronoi diagram
        TODO: put inside the class for visibility ?
    """

    return [index for index, region in enumerate(voronoi_obj.regions)
                    if not region]

def distance_calc_mod(list_list_point, points_coordinates, mapping=None):
    '''
        Calculates the distance between points.
        Adapted to accept :
            as list_list_points -> VoronoiMesh.ridge_points
            as mapping -> None
            as points_coordinates -> VoronoiMesh.points
            to return -> VoronoiMesh.ridgepoints_dist

            OR

            as list_list_points -> VoronoiMesh.closedridgesindex
            as mapping -> VoronoiMesh.ridge_vertices
            as points_coordinates -> VoronoiMesh.vertices
            to return -> VoronoiMesh.ridge_hypersurface[self.closedridgesindex]
    '''
    if mapping is None:
        coord = np.array([[points_coordinates[point]
                           for point in list_point]
                          for list_point in list_list_point])
    else:
        coord = np.array([[points_coordinates[point_2]
                           for point_2 in mapping[list_point]]
                          for list_point in list_list_point])

    diff_cords = coord[:, 0, :] - coord[:, 1, :]

    distance_array = np.sqrt(
            np.einsum('ij,ji->i', diff_cords, diff_cords.T))

    return distance_array


def check_missing_ridges(voronoi_obj):
    """
     Verify that all the ridges have been calculated
     c.f. http://www.qhull.org/html/qh-optf.htm#Fv2 and
     https://stackoverflow.com/questions/25754145/
        scipy-voronoi-3d-not-all-ridge-points-are-shown
     for more information on precision related problems with qhull that
     mighth cause some ridges to not be calculated.
     TODO : add param
    """

    unique_pts = np.unique(np.fromiter(
            itertools.chain.from_iterable(voronoi_obj.ridge_points), dtype=int))

    return np.delete(np.arange(voronoi_obj.npoints), unique_pts)



def correct_missing_ridges(voronoi_obj, points_id):
    """
     Verify that all the ridges have been calculated
     c.f. http://www.qhull.org/html/qh-optf.htm#Fv2 and
     https://stackoverflow.com/questions/25754145/
        scipy-voronoi-3d-not-all-ridge-points-are-shown
     for more information on precision related problems with qhull that
     mighth cause some ridges to not be calculated.

     If missing ridge detected, it calculates a voronoi diagram for a
     larger mesh sourrouding the original mesh and attemps to
     re-calculate the missing ridge. This is done by adding a few points
     around the original mesh.

     TODO : Check if the solution proposed is feasable. Check if it needs
         to be corrected.
    """
    pass


def setter_property(types__, condition=None,
                    condition_err_msg=None):
    '''
        Decorator that can be used to define a Setter descriptor for
        a class attribute.

        Parameters:
            types__ :  Tuple.
                Accepted type of the attribute value, e.g. (int, float)
            condition: function
                Returns False or True in function of the attribute value.
                True if the condition is respected.
            condition_err_msg: str.
                Error messsage displayed if condition not respected
    '''
    condition_err_msg_parent = condition_err_msg
    if condition is None:
        condition = lambda x, y:True
    class Setter:
        '''
            Adds the current value to the class obj dict by
            defining a Setter (__set__) descriptor.
        '''
        def __init__(self, func):
            self.func = func
            self.__name__ = func.__name__
            self.__doc__ = func.__doc__

        def __set__(self, obj, value):

            if value is not None:
                if not isinstance(value, types__):
                    raise TypeError('{0}.{2} must be: {1}'.format(
                            obj.__class__, ' or '.join((str(t) for t in types__)),
                            self.__name__))

                if condition_err_msg_parent is None:
                   self.condition_err_msg = 'FiniteVolume.{0} value is not \
                                         supported'.format(self.__name__)
                else:
                    self.condition_err_msg = condition_err_msg_parent

                if not condition(obj, value):
                    raise TypeError(self.condition_err_msg)

            obj.__dict__[self.__name__] = value
    return Setter


class FiniteVolume(ABC):
    '''
        TODO: Verify with Cristoph if a good idea.
        TODO: If ok to continue as it is, change the verifications
         for the ones that have a list within a list.
        TODO: Find a way to remove the def .. : pass ?

        Contains all the nescessary attributes and modules
         to preserve compatibility with the poisson.system.System class.
         One possibility to enforcing certain conditions is to use the
         property module to create a getter and setter such as :

    points = property(operator.attrgetter('_points'))
    @points.setter
    def points(self, points):
        if ((not isinstance(points, np.ndarray)) or len(points) == 0):
                raise Exception("Points cannot be empty and must be a np.array")
        self._points = points

        This is nontheless not ideal for the reasons better explained in
        these [1, 2] stackoverflow post (one of them is that the _points
        attribute can still be changed outisde of the class and
        the other is that a superfulous getter is created).

        A better option (according to [1, 2]) is to implement it how it is
        currently done here. It envolves storing the actual value in directly
        in the dict as explained by [1].

        [1] : https://stackoverflow.com/questions/9305751/
        force-ensure-python-class-attributes-to-be-of-specific-type
        [2] : https://stackoverflow.com/questions/17576009/
        python-class-property-use-setter-but-evade-getter
    '''

    def __init__(self):
        super().__init__()

    @setter_property(types__=(np.ndarray,),
                     condition=lambda obj, val: len(val) != 0,
                     condition_err_msg='Points cannot be empty')
    def points(self, value): pass

    @setter_property(types__=(int, float))
    def npoints(self, value): pass

    @setter_property(types__=(int, float))
    def nregions(self, value): pass

    @setter_property(types__=(int, float))
    def nridges(self, value): pass

    @setter_property(
            types__=(list,),
            condition=lambda obj, val: len(val) == getattr(obj, 'npoints'),
            condition_err_msg="point_ridges must have the same length as {} \
                                ".format('npoints'))
    def point_ridges(self, value): pass

    @setter_property(
            types__=(list,),
            condition=lambda obj, val: len(val) == getattr(obj, 'npoints'),
            condition_err_msg="points_firstneig must have the same length as \
                                {} ".format('npoints'))
    def points_firstneig(self, value): pass

    @setter_property(
            types__=(np.ndarray,),
            condition=lambda obj, val: len(val) == getattr(obj, 'npoints'),
            condition_err_msg="points_hypervolume must have the same length as\
                               {} ".format('npoints'))
    def points_hypervolume(self, value): pass

    @setter_property(
            types__=(np.ndarray,),
            condition=lambda obj, val: len(val) == getattr(obj, 'nregions'),
            condition_err_msg="region_point must have the same length as {} \
                                ".format('nregions'))
    def region_point(self, value): pass

    @setter_property(types__=(np.ndarray,),
                     condition=lambda obj, val: len(val) != 0,
                     condition_err_msg='closedregionindex cannot be empty')
    def closedregionindex(self, value):
        '''  Index indicating which points from the FiniteVolume.points numpy array
            form a closed volume.
        '''
        pass

    @setter_property(
            types__=(np.ndarray,),
            condition=lambda obj, val: len(val) == getattr(obj, 'nridges'),
            condition_err_msg="ridge_hypersurface must have the same length as\
                                {}".format('nridges'))
    def ridge_hypersurface(self, value): pass

    @setter_property(
            types__=(np.ndarray,),
            condition=lambda obj, val: len(val) == getattr(obj, 'nridges'),
            condition_err_msg="ridgepoints_dist must have the same length as\
                                {}".format('nridges'))
    def ridgepoints_dist(self, value): pass

    @abstractmethod
    def fneigs(self):
        '''
            TODO: describe what this function must do.
        '''
        pass


class Voronoi(FiniteVolume):
    '''
        TODO: Docstring
    '''

    version ='0.1'
    print(('VoronoiMesh version ' + version))

    def __init__(self, vor=None, check=False, grid=None, return_Qhull=True,
                 build=True, **kwargs):
        '''
            Parameters:
            -----------
                vor is a voronoi object comming from:
                    scipy.spatial.Voronoi,
                    scipy.spatial.qhull_Qhull.get_voronoi_diagram,
                    poisson.poisson.VoronoiMesh._make_voronoi,

                check: boolean. True to test each VoronoiMesh property
                    except the closedregionsindex and closedridgesindex

                grid: If vor is not given, one must give a grid so
                    a voronoi diagram can be built

                return_Qhull: if grid is not None, one can recover
                    the Qhull object used to make the voronoi diagram.

                build: bool
                    By default True. If True when Voronoi is
                    initialized the following functions are called:
                        self._closed_regions()
                        self._calc_ridges_distance()
                        self._calc_hypersurface_ridges()
                        self._calc_voronoi_hypervolume()

                **kwargs: arguments passed to
                    scipy.spatial.qhull._Qhull. c.f. doc scipy.spatial.

            Additionally:
                Contains all possible combinations between the voronoi properties.
                The name follows the rule A_B[î] is the B or list of Bs
                for element A[î] vor is either an object scipy.spatial.voronoi()
                or a namedtuple containing :
                    vor.points
                    vor.vertices
                    vor.regions
                    vor.point_region
                    vor.ridge_points
                    vor.ridge_vertices
                    vor.npoints

            Precision:
                The precision of the voronoi diagram with qhull is around
                10^-13, 10^-14.

            TODO: change the init so as to calculate any
            nescessary parameter (as requested by FiniteVolume. )
        '''
        super().__init__()
        self.grid = grid
        self.voronoi_obj = None
        self.qhull_obj = None

        if vor is not None:
            self.voronoi_obj = vor
        elif self.grid is not None:
            self._make_voronoi(return_Qhull=return_Qhull, **kwargs)
        else:
            raise AttributeError('Not enough parameters')

        # voronoi object properties
        self.points = None
        self.vertices = None
        self.point_region = None
        self.ridge_points = None
        self.ridge_vertices = None
        self.npoints = None
        self.region_vertices = None
        self.ndim = None
        self.nvertices = None
        self.nridges = None
        self.nregions = None

        # Fill voronoi properties:
        self.__fill_voronoi()

        # inverting the voronoi properties
        self.region_ridges = [[] for i in range(self.nregions)]
        # TODO: Check tests after modification of self.point_ridges
        self.point_ridges = [[] for i in range(self.npoints)]
        self.points_firstneig = [[] for i in range(self.npoints)]
        self.vertex_regions = [[] for i in range(self.nvertices)]
        self.vertex_ridges = [[] for i in range(self.nvertices)]
        self.ridge_regions = [[] for i in range(self.nridges)]
        self.closed_points_fneig = [[] for i in range(self.nregions)]
        self.region_point = np.ones(self.nregions, dtype=int)

        # Fill the lists
        self.__fill_inverse()
        # check the lists
        if check:
            self._check_inversion()

        # Closed ridges properties
        self.closedregionindex = None
        self.closed_points_fneig = None
        self.closedridgesindex = None

        # distance, hyperdistance and hypervolume properties
        self.ridgepoints_dist = None
        self.ridge_hypersurface = None
        self.points_hypervolume = None

        if build:
            print('Finding closed voronoi cells')
            self._closed_regions()
            print('Done selecting closed voronoi cells')

            print('Calculating points distance')
            self._calc_ridges_distance()
            print('Done')

            print('Calculating ridges hypersurfaces')
            self._calc_hypersurface_ridges()
            print('Done')

            print('Finding  hypervolume for closed voronoi cells' )
            self._calc_voronoi_hypervolume()
            print('Done finding hypervolume for closed voronoi cells')

    def _make_voronoi(self, return_Qhull=False, **kwargs):
        '''
            Constructs a n dimension voronoi using
            scipy.spatial.qhull._Qhull.get_voronoi_diagram()
            Also accepts a 1D grid.
        '''

        if self.grid.shape[1] > 1:
            self.voronoi_obj, self.qhull_obj = self._make_voronoi_nd(
                    grid=self.grid,
                    return_Qhull=return_Qhull,
                    **kwargs)
        else:
            self.voronoi_obj = self._make_voronoi_1d(self.grid)
            self.qhull_obj = None

    def _make_voronoi_nd(self, grid, return_Qhull=False, **kwargs):
        '''
            Make n>1 dimensions voronoi cell from a grid of points
            using scipy.spatial.qhull._Qhull.get_voronoi_diagram()
            **kwargs -> c.f scipy.spatial.qhull.
            TODO: test benchmark between this and scipt.spatial.Voronoi
                    See if it can be done in cython ( as is qhull) and
                    i think Voronoi ?
        '''

        #Parameters
        if 'furthest_site' not in kwargs:
            kwargs['furthest_site'] = False

        if 'incremental' not in kwargs:
            kwargs['incremental'] = False

        # ???: More elegant way than deleting it
        try:
            qhull_options = kwargs.pop('qhull_options')
        except KeyError:
            qhull_options = b"Qbb Qc Qx Qz"

        # Initialize Qhull object
        qhull_obj = qhull._Qhull(b"v", grid, qhull_options,
                                 **kwargs)

        # FOr incremental processing, e.g. add points use qhull._QhullUser
        voronoi = namedtuple('voronoi_prop', ['vertices', 'ridge_points',
                                                   'ridge_vertices', 'regions',
                                                   'point_region', 'points',
                                                   'npoints', 'ndim'])

        voronoi.points = qhull_obj.get_points()
        voronoi.npoints = len(voronoi.points)
        voronoi.ndim = qhull_obj.ndim

        (voronoi.vertices,
         voronoi.ridge_points,
         voronoi.ridge_vertices,
         voronoi.regions,
         voronoi.point_region) = qhull_obj.get_voronoi_diagram()


        if return_Qhull:
            return voronoi, qhull_obj
        else:
            return voronoi, None


    def _make_voronoi_1d(self, grid):
        '''
             Generates a namedtuble with the same properties of
             make_voronoi or scipy.spatial.Voronoi  but for an 1D case.
             Mesh corresponds to an 2D numpy array of shape [len, 1]
             The array must be sorted.
        '''

        grid = np.asarray(grid)
        distances = abs(grid[:-1] - grid[1:])

        voronoi = namedtuple('voronoi_1d', ['points', 'vertices', 'point_region',
                                        'ridge_points', 'ridge_vertices',
                                        'regions', 'npoints', 'ndim',
                                        'distances'])
        voronoi.points = grid
        voronoi.distances = distances

        voronoi.npoints = len(grid)
        voronoi.ndim = 1
        voronoi.vertices = (grid[:-1] + 0.5 * distances)[:, None]
        voronoi.point_region = np.arange(len(grid), dtype=np.dtype("i"))
        voronoi.ridge_points = np.array(list(zip(voronoi.point_region[:-1],
                                             voronoi.point_region[1:])))
        # it needs to be an list
        voronoi.ridge_vertices = [[i] for i in np.arange(len(voronoi.vertices),
                                                     dtype=np.dtype("i"))]
        voronoi.regions = ([(-1, 0)]
                       + list(zip(voronoi.point_region,
                                  np.roll(voronoi.point_region, -1)))[:-2]
                       + [(voronoi.npoints - 2, -1)])

        return voronoi


    def __fill_voronoi(self):
        '''
            Fill information about the voronoi diagram
        '''
        self.points = self.voronoi_obj.points
        self.vertices = self.voronoi_obj.vertices
        self.point_region = self.voronoi_obj.point_region
        self.ridge_points = self.voronoi_obj.ridge_points
        self.ridge_vertices = self.voronoi_obj.ridge_vertices
        self.npoints = self.voronoi_obj.npoints
        self.region_vertices = self.voronoi_obj.regions
        self.ndim = self.voronoi_obj.ndim
        # making room for index -1
        self.nvertices = len(self.voronoi_obj.vertices) + 1
        self.nridges = len(self.voronoi_obj.ridge_points)
        self.nregions = len(self.voronoi_obj.regions)


    def __fill_inverse(self):
        '''
            Inverse mapping of the voronoi properties
        '''

        # check for empty region_vertices
        empty_rv = check_empty_region_vertice(self.voronoi_obj)

        # remove them if nescessary
        (self.nregions, self.region_point, self.point_region,
         self.region_vertices) = self.remove_empty_region_vertices(
                 empty_rv, self.nregions, self.point_region, self.region_vertices)

        # verify
        if len(check_empty_region_vertice(self.voronoi_obj)) > 0:
            raise ValueError(('There was an attempt to remove one or '
                              + ' multiple empty region_vertices.'
                              + ' \n It was not successfull.'
                              + ' \n Please change your grid.'))

        (self.point_ridges,
         self.points_firstneig,
         self.vertex_ridges) = fvcython.inverse_ridge(self.ridge_points,
                                                      self.ridge_vertices,
                                                      self.point_ridges,
                                                      self.points_firstneig,
                                                      self.vertex_ridges)
        for i, vertices in enumerate(self.region_vertices):
            for vertex in vertices:
                self.vertex_regions[vertex].append(i)

        for i, region in enumerate(self.point_region):
            self.region_ridges[region] = self.point_ridges[i]

        for i, ridges in enumerate(self.region_ridges):
            for ridge in ridges:
                self.ridge_regions[ridge].append(i)

    def _closed_regions(self):
        '''
            Find the voronoi cells that are closed.
            Open voronoi regions are defined as regions with at least one vertex
            located in "infinity" (not exactly, for more precise information
             c.f. official doc of qhull). Those vertex are marked with a -1

        '''
        self.closedregionindex = np.delete(
            np.arange(self.nregions), self.vertex_regions[-1])


        self.closed_points_fneig = self.fneigs(
                self.region_point[self.closedregionindex])


    def _closed_ridges(self):
        '''
            Find closed ridges.
            Open voronoi ridges are defined as ridges with at least one vertex
            located in "infinity" (not exactly, for more precise information
             c.f. official doc of qhull). Those vertex are marked with a -1

        '''
        self.closedridgesindex = np.delete(
               np.arange(self.nridges), self.vertex_ridges[-1])


    def _calc_ridges_distance(self):
        '''
            Calculates the distance between the two points associated
            with one ridge.
            TODO: remove einsum for something else as it does not scales well
        '''
        if self.ndim == 1:
            self.ridgepoints_dist = self.voronoi_obj.distances
        else:

            self.ridgepoints_dist = distance_calc_mod(
                    list_list_point=self.ridge_points,
                    points_coordinates=self.points,
                    mapping=None)


    def _calc_hypersurface_ridges(self):
        '''
            Calculate the hypersurface of the voronoi cells for 2D and
            3D case.
            Uses surface_vectorized.
            TODO: Optimze
        '''
        dim = len(self.vertices[0])

        # Check if closedridges was already calculate.
        if self.closedridgesindex is None:

            self._closed_ridges()

        # Then check if it is empty
        if not bool(self.closedridgesindex.flat):

            self.ridge_hypersurface = None
            return warnings.warn('There are no closed ridges')

        self.ridge_hypersurface = np.ones(self.nridges)

        if dim == 1:
            self.ridge_hypersurface = self.ridge_hypersurface[:, None]
        if dim == 2:

            # define the 4 points of each ridge
            # Attention ; we make calculations that we don't use
            # This is because there are some ridges outside of the grid (with -1)
            # that we do calculate the surface this way
            self.ridge_hypersurface[self.closedridgesindex] = distance_calc_mod(
                    list_list_point=self.closedridgesindex,
                    points_coordinates=self.vertices,
                    mapping=self.ridge_vertices)

        elif dim == 3:

            surface_coord = [[self.vertices[vertex]
                              for vertex in self.ridge_vertices[ridge]]
                             for ridge in self.closedridgesindex]

            self.ridge_hypersurface[self.closedridgesindex] = \
                sc.surface_vectorized(surface_coord)


    def _calc_voronoi_hypervolume(self):
        '''
            Calculates the voronoi hypervolume for all the
            closed voronoi cells. Uses get_voronoi_hypervolume module.
            If it is not a closed region, returns np.NAN
        '''
        # Check if closed regions were calculated or exist
        if not bool(self.closedregionindex.flat):
            self._closed_regions()
            if not bool(self.closedregionindex.flat):
                self.closed_points_hypervolume = []
                return warnings.warn('There are no closed regions')

        self.points_hypervolume = np.ones(self.npoints) * np.NAN

        index_of_closed_points = self.region_point[self.closedregionindex]

        self.points_hypervolume[index_of_closed_points] \
            = self.voronoi_hypervolume(index_of_closed_points)

    def _check_inversion(self):

        # Does not test closed regions

        for i in range(self.npoints):
            assert i == self.region_point[self.point_region[i]
                                         ], 'point_region to region_point failled'

            # assert (i in self.ridge_points[self.point_ridges[i]]),
            # 'ridge_point to point_ridge failled'
            # in 3D There migth be points that are not associated
            # with any ridges
            if self.point_ridges[i]:
                assert (i in self.ridge_points[
                    self.point_ridges[i]]
                       ), 'ridge_point to point_ridge failled'

        for i in range(self.nridges):
            vertices = self.ridge_vertices[i]
            if -1 not in vertices:
                for vertex in vertices:
                    assert (
                        i in self.vertex_ridges[vertex]), \
                        'vertex_ridge to ridge_vertex failled'

            regions = self.ridge_regions[i]
            for region in regions:
                assert (
                    i in self.region_ridges[region]),\
                    'region_ridges to ridge_regions failled'

        for i in range(self.nregions):
            vertices = self.region_vertices[i]
            if -1 not in vertices:
                for vertex in vertices:
                    assert (
                        i in self.vertex_regions[vertex]),\
                        'vertex_regions to region_vertices failled'

        # test first neighbours
        for i in range(self.npoints):
            for ridge in self.point_ridges[i]:
                b_ridge = 0
                for neig in self.points_firstneig[i]:
                    if neig in self.ridge_points[ridge]:
                        b_ridge = 1

                assert b_ridge, 'first neighbours from point_ridges failled'

        print('Closed regions not tested')
        print('Inversion test : Passed')


    def remove_empty_region_vertices(self, empty_rv, nregions, point_region,
                                     region_vertices):
        '''
            Remove empty region_vertices and
            update region_point, point_region and nregions.
            Here for visibility and context
        '''
        if len(empty_rv) == 0:

            region_point = np.ones(nregions, dtype=int)
            region_point[point_region] = np.arange(
                    len(point_region))

        elif len(empty_rv) == 1:

            del region_vertices[empty_rv[0]]
            nregions = len(region_vertices)
            region_point = np.ones(nregions, dtype=int)

            # sort the point_region so that the index of regions can be
            # updated
            mask_sorted = point_region.argsort()

            # Find which elements in point_region must be updated
            indexes = mask_sorted[
                    np.searchsorted(point_region[mask_sorted], empty_rv[0])
                    :len(point_region)]

            point_region[indexes] = (point_region[indexes] - 1)

            region_point[point_region] = np.arange(
                    len(point_region))

        else:
            warnings.warn(('Current implementation does not accept'
                           + ' more than one empty element in'
                           + ' region_vertices'))

        return nregions, region_point, point_region, region_vertices


    def fneigs(self, index_of_points):
        '''
            Makes a copy of self.points_firstneig and eliminates
            any point not belonging to index_of_points
            TODO: add test ?
            index_of_point = np.ndarray[np.int_t, ndim=1]

            Cython version - 5x faster for large numbers (> 50K)
                for smaller number either faster (2x) or same order of mag.

            pure python version:

                points_in = np.zeros(len(self.points_firstneig),
                                     dtype=int)
                points_in[index_of_points] = np.ones(len(index_of_points),
                                                     dtype=int)

                index_of_points_fneigs = []
                for point in range(len(self.points_firstneig)):
                    index_of_points_fneigs.append([])
                    for fneig in self.points_firstneig[point]:
                        if points_in[fneig]:
                            index_of_points_fneigs[point].append(fneig)

            Here for visibility and context
        '''


        if index_of_points.dtype != np.int_:
            raise ValueError(('The element in index_of_points must have a dtype:'
                               +' np.int_t'))

        new_points_fneigs = fvcython.fneig(
                index_of_points,
                self.points_firstneig,
                self.npoints)

        return new_points_fneigs


    def voronoi_hypervolume(self, index_of_points):
        '''
        Given a voronoi mesh object and a set of index_of_points,
        it calculates the hypervolume for each voronoi cell around each
        given point.
        In 1D: distance between points
           2D: The surface of the voronoi cell. Uses poisson.surface_calc
           3D: The volume of the voronoi cell. Created a ConvexHull
               around the voronoi which then finds the volume.
        TODO: Optimize

        Here for visibility and context
        '''
        ndim = self.ndim
        index_of_regions = self.point_region[index_of_points]

        if ndim == 1:

            return np.abs(
                    np.diff(self.vertices[np.asarray(self.region_vertices)[
                            index_of_regions]][:, :, 0], axis=1)
                         )[:, 0, 0]

        elif ndim == 2:
            regs = [[self.vertices[reg]
                     for reg in self.region_vertices[i]]
                    for i in index_of_regions]
            return sc.surface_vectorized(regs)

        elif ndim == 3:

            regs = [[self.vertices[reg]
                     for reg in self.region_vertices[i]]
                    for i in index_of_regions]
            volume = np.array([qhull.ConvexHull(reg).volume for reg in regs])
            return volume





