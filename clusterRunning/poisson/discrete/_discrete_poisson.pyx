'''
    TODO: refactor and write Docstring
'''
from cython cimport view
import numpy as np
cimport numpy as np

cpdef find_non_empty(list list_input,
                     np.ndarray[np.int_t, ndim=1] index_to_test):

    cdef int position
    cdef long indice
    cdef long [:] indices = index_to_test
    cdef int [:] mapping = np.zeros(len(indices), dtype=np.dtype("i"))

    for position, indice in enumerate(indices):
        if bool(list_input[indice]):
            mapping[position] = 1

    return np.asanyarray(mapping).astype(bool)
