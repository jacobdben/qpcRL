'''
    TODO: refactor and write Docstring
'''
from cython cimport view
import numpy as np
cimport numpy as np
import time

cpdef fneig(np.ndarray[np.int_t, ndim=1] index_keep, list index_all,
            int npoints):

    cdef list index_of_points
    cdef int point
    cdef int fneig
    cdef int id_point

    t = np.zeros(npoints, dtype=np.dtype("i"))
    t[index_keep] = np.ones(len(index_keep), dtype=np.dtype("i"))
    cdef int [:] points_in = t

    index_of_points = []
    for point in range(len(index_all)):
        index_of_points.append([])
        for fneig in index_all[point]:
            if points_in[fneig]:
                index_of_points[point].append(fneig)

    return index_of_points


cpdef inverse_ridge(np.ndarray[int, ndim=2] ridge_points, list ridge_vertices,
                    list point_ridges, list index_all,
                    list vertex_ridges):

    cdef int [:, :] cython_ridge_points = ridge_points
    cdef int i, j, point, vertex

    for i in range(len(ridge_points)):
        for j, point in enumerate(ridge_points[i]):
            point_ridges[point].append(i)
            index_all[point].append(ridge_points[i][(j+1) % 2])

        for vertex in ridge_vertices[i]:
            vertex_ridges[vertex].append(i)

    return point_ridges, index_all, vertex_ridges