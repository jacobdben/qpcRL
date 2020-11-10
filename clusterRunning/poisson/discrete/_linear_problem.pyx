from cython cimport view
import numpy as np
cimport numpy as npc

cpdef indices_from_indtpr(int[:] indtpr, int size):
    '''
        Create vector with columns index from a indtpr comming
            from slice_sparse in solver.py in order to index the data in
            the sparse matrix.
        Parameters:
            indtpr: np.ndarray of ndim=1
            size: size of the np.ndarray (ndim = 1) containing the columns
                index
        Returns:
            indices: np.ndarray[np.int, ndim=1] containing the columns
                index of the data vector from the sparse matrix returned by
                slice_sparse in solver.py
    '''
    cdef int i
    cdef int d
    cdef npc.ndarray[npc.int_t, ndim=1] indices = np.zeros((size), dtype=int)

    for i in range(indtpr.shape[0] - 1):
        for d in range(indtpr[i], indtpr[i + 1]):
            indices[d] = i

    return indices

