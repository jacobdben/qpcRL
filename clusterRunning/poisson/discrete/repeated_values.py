'''
    TODO : change name
'''


__all__ = ['indices',
           'remove_from_grid']


import itertools

import numpy as np


def indices(points, index=None, return_val=False):
    '''
        Sorts points (a numpy vector) and finds the range of each one of its
        repeated elements.

        This is done by using numpy unique and making the cumulative sum
        of unique_counts.

        Since the points are sorted, if the first element is repeated n times,
        the first n elements of the array will correspond to the former.
        Therefore it can be accesed with np.arange(0, n).

        Parameters :
            points :  numpy 1D array
            index : numpy 1D array
                if None all the values in points will be used in numpy unique
                if numpy array only the indices of the numpy array points in
                the vector index will be considered.
        Attributes:
            cs : the range of each repeated element in points[index]
            index : maps the cs into the input points vector
            e.g. point[index[cs[0][0]]] corresponds to the first position of
            the first repeated element.
    '''

    if index is None:
        index = points[:].argsort()
    else:
        index = index[points[index].argsort()]

    uniqval, uniqcount = np.unique(points[index], return_counts=True)

    # Find repeated values and calculate the cumulative sum
    mask = uniqcount != 1
    cs = np.concatenate(([0], np.cumsum(uniqcount)))
    cs = np.array([cs[0:-1], cs[1:]]).T

    if return_val:
        return cs[mask], index, uniqval
    else:
        return cs[mask], index


def remove_from_grid(grid, decimals=5):
    '''
     Finds the repeated elements in the grid and then removes them from
     the latter.

     This is done by initially finding the repeated elements along the first
     axis ( c.f.indices()  function), the from those the ones
     that are also redundant in the second axis and so on. Up to three axis(3D)

     Parameters:
         grid = numpy array
         decimals = number
     Returns:
         grid : the new grid without any repeated values
         list_repeated =

     TODO : Test it in 3D
    '''

    list_repeated = []

    # The floats must be rounded so that unique can be used
    grid_points = grid.round(decimals=decimals)

    # For x axis
    cs_x, index_xx = indices(grid_points[:, 0])

    # For y axis
    list_repeated = []
    for cs_x_temp in cs_x:
        # index for the x/y cordinates with x constant for the
        # values repeating along the x axis.
        index_xy = index_xx[np.arange(cs_x_temp[0], cs_x_temp[1])]
        cs_y, index_xy = indices(grid_points[:, 1],
                                                index=index_xy)

        list_repeated_y = []
        # If 3D
        if len(grid_points[0]) == 3:
            # For all repeated y (x constant)
            for cs_2_temp in cs_y:

                index_xz = index_xy[np.arange(cs_2_temp[0], cs_2_temp[1])]
                cs_z, index_xz = indices(
                        grid_points[:, 2],
                        index=index_xz)

                list_repeated_z = []
                for cs_3mask in cs_z:

                    # -1 since we want to keep one value
                    list_repeated_z.append(index_xz[np.arange(cs_3mask[0],
                                                     cs_3mask[1] - 1)])

                list_repeated_y.append(list_repeated_z)

            list_repeated.append(itertools.chain.from_iterable(
                    list_repeated_y))

        # If only 2D
        elif len(grid_points[0]) == 2:
            # For all repeated y
            for cs_2mask in cs_y:

                # -1 since we want to keep one value
                list_repeated_y.append(index_xy[np.arange(cs_2mask[0],
                                                 cs_2mask[1] - 1)])

            list_repeated.append(list_repeated_y)

    list_repeated = list(itertools.chain.from_iterable(list_repeated))

    if len(list_repeated) > 0:

        list_repeated_conc = np.concatenate(list_repeated)
        keep = np.delete(np.arange(len(grid_points)),
                         list_repeated_conc[0:len(list_repeated_conc)])

        grid = grid[keep]

    return grid, list_repeated

def _test_remove_from_grid():

    from poisson_toolbox import shapes

    bbox2d = [0, 10.0, 0, 10.0]
    bbox3d = [0, 10.0, 0, 10.0, 0, 10.0]
    rect_2D = shapes.Rectangle(length=[10, 10])
    rect_3D = shapes.Rectangle(length=[10, 10, 10])

    grid_2d_obj = mesher.Mesher(meshs=[(bbox2d,0.1, rect_2D, 0)])
    index = np.arange(len(grid_2d_obj.mesh_points))
    index2 = np.arange(len(grid_2d_obj.mesh_points))

    np.random.shuffle(index)
    np.random.shuffle(index2)

    unique_points, list_repeated = remove_from_grid(
            np.concatenate([grid_2d_obj.mesh_points[index],
                            grid_2d_obj.mesh_points[index2],
                            grid_2d_obj.mesh_points[index2]]))

    unique_points_2, list_repeated = remove_from_grid(unique_points)

    similarity = np.where((unique_points - unique_points_2) != 0.0)
    length_diff_1 = len(unique_points) - len(grid_2d_obj.mesh_points)
    length_diff_2 = len(unique_points) - len(unique_points_2)

    print('')

    if ((len(similarity[0]) == 0) and (len(similarity[1]) == 0)
        and (length_diff_1 == 0) and (length_diff_2 == 0)
        and len(list_repeated) == 0):

        print('2D remove_from_grid function test : Passed')
    else:
        print('2D remove_from_grid function test : Not Passed')

    grid_3d_obj = mesher.Mesher(meshs=[(bbox3d, 0.1, rect_3D, 0)])
    index = np.arange(len(grid_3d_obj.mesh_points))
    index2 = np.arange(len(grid_3d_obj.mesh_points))


    unique_points, list_repeated = remove_from_grid(
            np.concatenate([grid_3d_obj.mesh_points[index],
                            grid_3d_obj.mesh_points[index2],
                            grid_3d_obj.mesh_points[index2]]))

    unique_points_2, list_repeated = remove_from_grid(unique_points)

    similarity = np.where((unique_points - unique_points_2) != 0.0)

    length_diff_1 = len(unique_points) - len(grid_3d_obj.mesh_points)
    length_diff_2 = len(unique_points) - len(unique_points_2)

    print('')

    if ((len(similarity[0]) == 0) and (len(similarity[1]) == 0)
        and (length_diff_1 == 0) and (length_diff_2 == 0) and
        len(list_repeated) == 0):

        print('3D remove_from_grid function test : Passed')
    else:
        print('3D remove_from_grid function test : Not Passed')


