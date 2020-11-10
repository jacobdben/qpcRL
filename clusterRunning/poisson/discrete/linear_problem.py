'''
    Class LinearProblem.
    Constructs the system of equations A = xB and calls solver.Solver instance.
'''


__all__ = ['LinearProblem']


import warnings
import copy
from abc import ABC, abstractmethod

from scipy import linalg
from scipy.optimize import minimize
import scipy.sparse as sps
from scipy.sparse.linalg import dsolve

import numpy as np

from matplotlib import pyplot as plt

from . import _linear_problem as sct
from poisson.tools import post_process
from poisson.tools import save
from poisson.continuous import shapes
from poisson.continuous.shapes import aspiecewise
from poisson.tools import plot as p_plt
from poisson import solver


def condition_number(sparse_matrix):

    try:

        norm_sparse = sps.linalg.norm(sparse_matrix)
        norm_inv_sparse = sps.linalg.norm(sps.linalg.inv(sparse_matrix))
        return norm_sparse*norm_inv_sparse

    except ValueError:
        return None


def minimzation_func(capa_sparse, initia_normalization,
                     pos_dirichlet, pos_neuman):

    tem_capa_sparse = copy.deepcopy(capa_sparse)
    tem_capa_sparse.data = tem_capa_sparse.data * initia_normalization

    def fun(normalization):

        tem_capa_sparse_2 = copy.deepcopy(tem_capa_sparse)

        tem_capa_sparse_2.data = tem_capa_sparse_2.data / normalization

        lhs_sparse = make_lhs(tem_capa_sparse_2.data,
                              tem_capa_sparse_2.indices,
                              tem_capa_sparse_2.indptr,
                              pos_dirichlet, pos_neuman,
                              tem_capa_sparse_2)

        return condition_number(lhs_sparse)

    return fun


def minimize_condition_number(capa_sparse, lhs_sparse, normalization,
                              pos_dirichlet, pos_neuman, tol_cond=1e-4):

    cond_num = condition_number(lhs_sparse)

    if cond_num > tol_cond:

        capa_data = capa_sparse.data
        capa_minimization_func = minimzation_func(
                capa_sparse,
                normalization,
                pos_dirichlet, pos_neuman)

        new_normalization = np.mean(np.abs(capa_data * normalization))
        new_cond_num = capa_minimization_func(new_normalization)

        if new_cond_num < cond_num:
            capa_sparse.data = (capa_data * normalization) / new_normalization
            normalization = new_normalization

    lhs_sparse = make_lhs(capa_sparse.data,
                          capa_sparse.indices,
                          capa_sparse.indptr,
                          pos_dirichlet, pos_neuman,
                          capa_sparse)

    return capa_sparse, lhs_sparse, normalization


def slice_sparce(matrix, rows, columns):
    ''' Slice sparse matrix using matrix multiplication
        Parameters :
            matrix : sparse matrix
            rows: np.ndarray of 1 dimension containing the rows to be
                selected
            rows: np.ndarray of 1 dimension containing the columns to be
                selected
    '''

    if not isinstance(matrix, sps.csc_matrix):
        print(type(matrix))
        raise

    row_sparse, col_sparse = matrix.shape
    number_rows = len(rows)
    number_cols = len(columns)
    left_matrix = sps.csr_matrix(([1]*number_rows,
                                  (range(number_rows), rows)),
                                 shape=(number_rows, row_sparse),
                                 dtype=bool)
    right_matrix = sps.csc_matrix(([1]*number_cols,
                                   (columns, range(number_cols))),
                                  shape=(col_sparse, number_cols),
                                  dtype=bool)

    return left_matrix * matrix * right_matrix


def from_capa_mat(out_row, capa_pos_row, possible_col, possible_points,
                  data_sparse, indices_sparse, indptr_sparse,
                  element_pos,
                  out_data_sparse, out_row_sparse, out_col_sparse):
    '''
        Take the data from a row in the sparse capacitance matrix
        and updates out_data_sparse, out_row_sparse and out_col_sparse.
        Parameters:
            out_row: The row of the data in the new sparse matrix
            capa_pos_row: The row in the sparse capacitance matrix from
                which the data will be taken from.
            possible_points: When checking the column elemnts in the
                selected row of the capacitance matrix, they not only need
                to be different than zero but also in the possible_points
                array
            possible_col: The index of the column where the data points taken
                from the capacitance matriw will be put in the new sparse
                matrix
            data_sparse, indices_sparse, indptr_sparse :
                c.f. scipy.sparse.csr_matrix documentation
            element_pos: indicates where to put the new data, row and column
                information into respectlvely out_data_sparse,
                out_row_sparse and out_col_sparse
            out_data_sparse, out_row_sparse, out_col_sparse: np.array
                containing information nescessary to make the new sparse matrix
        Returns:
            out_data_sparse, out_col_sparse, out_row_sparse, element_pos:
                Same format as the inputs but updated.
    '''

    data_row = data_sparse[indptr_sparse[capa_pos_row]
                           :indptr_sparse[capa_pos_row + 1]]
    col_row_all = indices_sparse[indptr_sparse[capa_pos_row]
                                  :indptr_sparse[capa_pos_row + 1]]
    col_row_mapping = np.isin(possible_points, col_row_all)
    col_row = possible_col[col_row_mapping]

    # in case some dirichlet have other dirichelt neighbours.
    if len(col_row) != len(data_row):
        mapping = np.arange(len(data_row))[
                np.isin(col_row_all,possible_points[col_row_mapping])]
        out_data_sparse[element_pos : element_pos + len(col_row)] = data_row[
            mapping]
    else:
        out_data_sparse[element_pos : element_pos + len(col_row)] = data_row

    out_col_sparse[element_pos : element_pos + len(col_row)] = col_row
    out_row_sparse[element_pos : element_pos + len(col_row)] = np.ones(
            len(col_row))*out_row

    element_pos += len(col_row)

    return out_data_sparse, out_col_sparse, out_row_sparse, element_pos


def make_points_input_vector(discrete_poisson, dirichlet_val,
                             charge_val, mixed_val):
    '''
        From the continuous function defined by the user
        build the points_input vector.
        Parameters:
            discrete_poisson: object from DiscretizePoisson
            dirichlet_val, charge_val and mixed_val: list of functions
                that return False or True if the point is in the zone
                delimited by the function and the value associated with that
                point (e.g. charge value for charge_val).
            mixed_val: c.f. LinearProblem.__init__ Docstring for more info.
        Returns:
            points_input: np.array containing the value associated at each
                point in the mesh.

    '''

    regions_names = [discrete_poisson.regions_names_dict[key] for key in [
             'dirichlet', 'neuman', 'mixed']]
    regions_values = [dirichlet_val, charge_val, mixed_val]

    # In case a points_input array does not exists yet.
    points_input = np.zeros(discrete_poisson.mesh.npoints)

    for region_numb, region_name in enumerate(regions_names):

        region_ndpt = getattr(discrete_poisson.regions, region_name)

        for sub_region_values in regions_values[region_numb]:
            if bool(sub_region_values):
                try:
                    boolean, valor = sub_region_values(
                            discrete_poisson.mesh.points[
                                    region_ndpt.index_of_mesh_points])
                    points_input[region_ndpt.index_of_mesh_points[
                        boolean]] = valor[boolean]
                except TypeError:
                    points_input[np.asarray(sub_region_values[0])] = \
                        np.asarray(sub_region_values[1])

    return points_input


def make_lhs(data_sparse, indices_sparse, indptr_sparse,
             pos_dirichlet, pos_neuman, capa_sparse):
    '''
        From the data of the sparse capacitance matrix,
        build the rigth hand side sparse matrix of the system of equations.
    '''

    # Select -C_{DN} block from C sparse matrix
    c_dn = slice_sparce(-capa_sparse, pos_dirichlet, pos_neuman)
    # Select -C_{NN} block from C sparse matrix
    c_nn = slice_sparce(-capa_sparse, pos_neuman, pos_neuman)
    # Find row index for both of them
    indices_row_dn = sct.indices_from_indtpr(c_dn.indptr,
                                             c_dn.indptr[-1])
    indices_row_nn = sct.indices_from_indtpr(c_nn.indptr,
                                             c_nn.indptr[-1])
    indices_row_nn += len(pos_dirichlet)

    lhs_data = np.concatenate((c_dn.data, c_nn.data,
                               np.ones(len(pos_dirichlet))))

    lhs_column = np.concatenate(((c_dn.indices + len(pos_dirichlet)),
                                 (c_nn.indices + len(pos_dirichlet)),
                                 np.arange(len(pos_dirichlet), dtype=int)))
    lhs_row = np.concatenate((indices_row_dn,
                              indices_row_nn,
                              np.arange(len(pos_dirichlet), dtype=int)))

    lhs_data = lhs_data.astype('complex')
    return sps.coo_matrix((lhs_data, (lhs_row, lhs_column)),
                           shape=(len(pos_dirichlet) + len(pos_neuman),
                                  len(pos_dirichlet) + len(pos_neuman)))


def make_rhs(data_sparse, indices_sparse, indptr_sparse,
             pos_dirichlet, pos_neuman, capa_sparse):
    '''
        From the data of the sparse capacitance matrix,
        build the left hand side sparse matrix of the system of equations.
    '''

    # Select C_{DD} block from C sparse matrix
    c_dd = slice_sparce(capa_sparse, pos_dirichlet, pos_dirichlet)
    # Select C_{DN} block from C sparse matrix
    c_nd = slice_sparce(capa_sparse, pos_neuman, pos_dirichlet)
    # Find row index for both of them
    index_row_c_dd = sct.indices_from_indtpr(c_dd.indptr,
                                             c_dd.indptr[-1])
    index_row_c_nd = sct.indices_from_indtpr(c_nd.indptr,
                                             c_nd.indptr[-1])
    index_row_c_nd += len(pos_dirichlet)

    rhs_data = np.concatenate((c_dd.data, c_nd.data,
                               -np.ones(len(pos_neuman))))

    rhs_column = np.concatenate((c_dd.indices,
                                 c_nd.indices,
                                 np.arange(len(pos_dirichlet),
                                           len(pos_dirichlet) + len(pos_neuman),
                                           dtype=int)))

    rhs_row = np.concatenate((index_row_c_dd,
                              index_row_c_nd,
                              np.arange(len(pos_dirichlet),
                                        len(pos_dirichlet) + len(pos_neuman),
                                        dtype=int)))
    rhs_data = rhs_data.astype('complex')

    return sps.csr_matrix((rhs_data, (rhs_row, rhs_column)),
                           shape=(len(pos_dirichlet) + len(pos_neuman),
                                  len(pos_dirichlet) + len(pos_neuman)))


def make_rhs_value_sparse(points_input, pos_dirichlet, pos_neuman,
                          discrete_poisson):
    '''
        From the input data,
        build the left hand side vector (sparse) of the system of equations.
    '''
    lhs_value = np.zeros(len(pos_dirichlet)+len(pos_neuman))

    lhs_value[0:len(pos_dirichlet)] = (points_input[pos_dirichlet]
                                       * discrete_poisson.capa_normalization)

    lhs_value[len(pos_dirichlet):len(pos_dirichlet) +
              len(pos_neuman)] = points_input[pos_neuman]


    return sps.csc_matrix(lhs_value.astype('complex'))


def check_sparse_csr(sparse_matrix):
    '''
        Check for zeros or NaN in sparse matrix of scr type.
        Important to assess the singualrity of a matrix.
        Returns:
            zeros_row_index: np.array with the indices of the rows
                which do not have any non zero value.
            nan_pos: list containing a sublist of the type [row, columns]
                where columns correspond to the values within row that
                are NaN.
    '''

    data_sparse = sparse_matrix.data
    indices_sparse = sparse_matrix.indices
    indptr_sparse = sparse_matrix.indptr

    zeros_row_index = np.arange(len(indptr_sparse) - 1)[
            (indptr_sparse == np.roll(indptr_sparse, -1))[:-1]]

    nan_pos = list()

    for i in range(len(indptr_sparse) - 1):
        data = data_sparse[indptr_sparse[i]:indptr_sparse[i + 1]]
        if np.isnan(data).any():
            nan_col_index = indices_sparse[
                    indptr_sparse[i]:indptr_sparse[i + 1]][np.isnan(data)]
            nan_pos.append([i, nan_col_index])

    return zeros_row_index, nan_pos


def check_capacitance(capacitance_sparse, pos_neuman, pos_dirichlet,
                      pos_mixed):

    '''
        Check capacitance sparse matrix for NaN and zeros.
        return a warning if any is found
    '''
    zeros_row_index, nan_pos_list = check_sparse_csr(capacitance_sparse)

    if (len(zeros_row_index) != 0.0) or (len(nan_pos_list) != 0.0):
        print('\n')
        print('#'*80)
        print('#'*80)
        print('\n')
        print((' The following capacitance matrix rows'
               + ' are zero : '))
        print('\n')
        print(zeros_row_index)
        print('\n')
        print(('The ones that belong to either pos_neuman, pos_dirichlet or'
               + ' pos_mixte are:'))
        print('pos_neuman : ')
        print(pos_neuman[np.isin(pos_neuman, zeros_row_index)])
        print('pos_dirichlet : ')
        print(pos_dirichlet[np.isin(pos_dirichlet, zeros_row_index)])
        print('pos_mixed : ')
        print(pos_mixed[np.isin(pos_mixed, zeros_row_index)])

        print('\n')
        print((' The folowing capacitance matrix elements '
               + ' are NaN ( c.f. [row, columns]): '))
        print('\n')
        print(nan_pos_list)
        print('\n')

        print('\n')
        warnings.warn('The capacitance matrix is singular. This migth cause'
                      + ' the rigth hand side matrix to be singular.')
        print('\n')
        print('\n')
        print('#'*80)
        print('#'*80)
        print('\n')


def check_rhs_lhs(sparse_matrix, which_matrix, error=True):
    '''
        Check rigth hand side (or lhs) sparse matrix for NaN and zeros.
        return an error if any is found
        Parameters:
                sparse_matrix: csr sparse matrix
                which_matrix = string with the name of
                    the matrix to be tested.
    '''
    zeros_row_index, nan_pos_list = check_sparse_csr(sparse_matrix)

    if (len(zeros_row_index) != 0.0) or (len(nan_pos_list) != 0.0):
        print('\n')
        print('#'*80)
        print('#'*80)
        print('\n')
        print((' The following ' + which_matrix + ' matrix rows'
               + ' are zero : '))
        print('\n')
        print(zeros_row_index)
        print('\n')

        print('\n')
        print((' The folowing ' + which_matrix + ' matrix elements '
               + ' are NaN ( c.f. [row, columns]): '))
        print('\n')
        print(nan_pos_list)
        if error:

            print('\n')
            print('The matrix for the system of equations is singular')
            print('\n')
            raise (AssertionError(
                    'The matrix for the system of equations is be singular'))
        else:

            print('\n')
            print('The matrix for the system of equations migth singular')
            print('\n')
            warnings.warn(
                    'The matrix for the system of equations migth be singular')
            print('\n')
            print('\n')
            print('#'*80)
            print('#'*80)
            print('\n')


def to_piecewise(data_input):
    '''
        From data_input check each element and verifies if it can
        be sent to shapes.aspieceiwise().

        If data_input[i][0] != callable or if there is a j so that
        data_input[i][j] != callable that particular element i
        is not sent to aspieceiwise and is thus not modified.

        Parameters:
        -----------
            data_input: list, tuple or dict.

        Returns:
        --------
            data_input; list, tuple or dict after it has
                gone through shapes.aspiecewise.

        TODO: Change how this is done. More elegant way of input selection ?
    '''

    tem_data_input = []
    removed_data_input = np.zeros(len(data_input), dtype=bool)

    if isinstance(data_input, dict):
        data_input.update({key: val for key, val in zip(
                data_input.keys(),
                to_piecewise(list(data_input.values())))})

    elif isinstance(data_input, (list, tuple)):
        for pos, el in enumerate(data_input):
            if not isinstance(el, shapes.PiecewiseFunction):
                if callable(el[0]):
                    tem_data_input.append(el)
                    removed_data_input[pos] = 1
            elif (isinstance(el, (tuple, list))
                  and np.all([isinstance(el_, (tuple, list)) for el_ in el])):
                if np.all([callable(el_2[0]) for el_2 in el]):
                    tem_data_input.append(el)
                    removed_data_input[pos] = 1


    tem_data_input = aspiecewise(tem_data_input)
    for num, pos in enumerate(np.arange(len(data_input))[removed_data_input]):
        data_input[pos] = tem_data_input[num]

    return data_input


class LinearProblem(object):
    '''
        Defnies a system of equations from a given poisson problem
        and input values ( voltage and charge) which can be solved or
        updated as needed.
        version 0.2 -> 0.3: Uses WrapperShape in input instead
            of functions. Unified solve and solve_mumps into solve and
            added parameters solver to chose with solver to use.
            This was done to allow for the use of the Solver abc class
            and its subclasses.
    '''

    version = '0.3'
    print('SystEquations version: {0}'.format(version))

    def __init__(self, discrete_poisson, voltage_val=None, charge_val=None,
                 mixed_val=None, pos_voltage_mixed=None,
                 pos_charge_mixed=None, is_charge_density=True,
                 check_rhs=False, check_lhs=False, check_capa_mat=False,
                 build_equations=True, solve_problem=True, verbose=False,
                 solver='mumps'):
        '''
            Parameters:
                discrete_poisson: Object from Discretize Poisson

                voltage_val: shapes.PieceiwiseFunction class or
                    tuple/list of tuples:
                    e.g. ((func, val), ((func, val), (func, val)))
                    An element of dielectric can thus be a tuple that has
                    as elements:
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
                            list_input[i][0]: function (such as the one defined
                                for Shape.geometry(x)) or shapes.Shape class as
                                the first element

                            list_input[i][1]: constant or a function that can
                                be evaluated in a set of coordinates
                                x (np.ndarray) returing a number.
                            (c.f it is the value parameter in shapes.WrapperShape)

                      3 -> two elements:
                           list_input[i][0]: np.ndarray with, as element,
                               the index of the point in the mesh.

                           list_input[i][1]: np.ndarary with the values of
                               voltage at the corresponding point.

                charge_val: idem as voltage val. If it is not given the
                    default value is 0.
                    The value is an adimensional quantity (
                    normalized by scipy.constants.elementary_charge). It can
                    be a density (default) or the number of "charge carriers"
                    inside a volumic cell.
                    Concerning how the value is defined in 2D and 1D
                    go to the report.

                mixed_val: list where each element is either :
                        1- a function such as for the charge_val
                            and voltage_val
                        2- a list where the first element is a numpy array of
                            points indexes and the second element is a
                            numpy array containing the values associated
                            with each point. The point index must be the one
                            used to index the point in the mesh.

                        If not defined but there is a region of
                        mixed type, the default value of 0 is given.

                pos_voltage_mixed: numpy array.
                    Specified among the points of mixed type which ones
                    are now of voltage type.
                    The elements in the array are a list of the points index
                    where the indexes are the same as the ones used in
                    the mesh.

                pos_charge_mixed: Idem as pos_voltage_mixed but not obligatory.
                    Every point not defined in pos_voltage_mixed is by
                    default of charge type.

                is_charge_density: boolean.
                    If True the input value in charge_val is treated as a
                    density of carriers. The default value is True.
                    If True the solver output is also in charge density.

                check_rhs: boolean
                    If True tests the right hand side matrix B (Ax = B)

                check_lhs: boolean
                    If True tests the integrity of the left hand side
                    matrix A (Ax=B)

                check_capa_mat: boolean
                    If True tests the integrity of the capacitance matrix
                    given by discrete_poisson

                build_equations: boolean
                    If True automatically builds the system of equations
                    when initializing the class. BY default True

                solve_problem: boolean.
                    If True automatically solve the system of equations
                    when initializing the class. By default True.
                    Default solver -> solve_mumps()

                verbose: bool
                    if True, print additionnal info when
                    define_mixed_points_type is called

                solver: str
                    Can be:'mumps' -> uses kwant.linlag.mumps
                           'spsolve' -> uses scipy.sparse.linlag.dsolve.spsolve

            TODO: Change how the solver calculated the charge at each cell
            when a charge density is given.

        '''

        self.discrete_poisson = discrete_poisson

        if voltage_val is None:
            voltage_val = ()
        self.voltage_val = to_piecewise(voltage_val)

        if charge_val is None:
            charge_val = ()
        self.charge_val = to_piecewise(charge_val)

        if mixed_val is None:
            mixed_val = (())
        self.mixed_val = to_piecewise(mixed_val)

        if pos_voltage_mixed is None:
            pos_voltage_mixed = np.array([])
        self.pos_voltage_mixed = pos_voltage_mixed

        if pos_charge_mixed is None:
            pos_charge_mixed = np.array([])
        self.pos_charge_mixed = pos_charge_mixed

        self.is_charge_density = is_charge_density
        self.check_rhs = check_rhs
        self.check_lhs = check_lhs
        self.check_capa_mat = check_capa_mat

        self.pos_voltage = self.discrete_poisson.points_system_voltage
        self.pos_charge = self.discrete_poisson.points_system_charge

        self.rhs_sparse = None
        self.lhs_sparse = None
        self.rhs_sparse_mat = None
        self.rhs_vector_sparse = None

        self.points_input = None
        self.points_output = None

        self.points_charge = None
        self.points_voltage = None

        self.solver = None
        self.solver_instance = None

        self.verbose = verbose

        if build_equations:
            self.build()

        if solve_problem and build_equations:
            self.solve(solver_type=solver)

    def build(self, normalize_capa=True):
        '''
            Build the rhigh hand side and left hand side
            matrices of  the system of equations Ax=B
        '''

        self.points_input = make_points_input_vector(
                self.discrete_poisson,
                self.voltage_val,
                self.charge_val,
                self.mixed_val)

        # Add mixed points index to self.pos_voltage and self.pos_charge
        self.define_mixed_points_type()

        if self.is_charge_density:
            self.points_input[self.pos_charge] = (
                    (self.points_input[self.pos_charge]
                     * self.discrete_poisson.mesh.points_hypervolume[
                             self.pos_charge]))

        if normalize_capa:
            self._normalized_capacitance()

        capa_sparse = self.discrete_poisson.points_capacitance_sparse
        data_sparse = capa_sparse.data
        indices_sparse = capa_sparse.indices
        indptr_sparse = capa_sparse.indptr

        pos_mixed = self.discrete_poisson.points_system_mixed

        if self.check_capa_mat:
            check_capacitance(capa_sparse, self.pos_charge,
                              self.pos_voltage,
                              pos_mixed)

        self.rhs_vector_sparse = make_rhs_value_sparse(
                self.points_input, self.pos_voltage, self.pos_charge,
                self.discrete_poisson)

        self.lhs_sparse = make_lhs(
                data_sparse, indices_sparse, indptr_sparse,
                self.pos_voltage, self.pos_charge, capa_sparse)

        if self.check_lhs:
            check_rhs_lhs(
                    self.lhs_sparse, which_matrix='Left hand side',
                    error=True)

        self.rhs_sparse_mat = make_rhs(
                data_sparse, indices_sparse, indptr_sparse,
                self.pos_voltage, self.pos_charge, capa_sparse)

        self.rhs_sparse = self.rhs_sparse_mat * self.rhs_vector_sparse.T

        if self.check_rhs:
            check_rhs_lhs(self.rhs_sparse, which_matrix='Rigth hand side',
                  error=False)

    def _normalized_capacitance(self):
        '''
            Normalizes the capacintace matrix so the average over the
            absolute value of all elements is 1.
        '''

        capa_sparse = self.discrete_poisson.points_capacitance_sparse
        data_sparse = capa_sparse.data

        # To minimize the condition number
        new_normalization = np.mean(np.abs(data_sparse))
        capa_sparse.data = (data_sparse / new_normalization)
        self.discrete_poisson.capa_normalization = (
                self.discrete_poisson.capa_normalization
                                              * new_normalization)
        data_sparse = capa_sparse.data

    def define_mixed_points_type(self, pos_charge_mixed=None,
                                  pos_voltage_mixed=None):
        '''
            Adds the points index in pos_charge_mixed
            to self.pos_neuman and the index in
            pos_voltage_mixed in self.pos_voltage
        '''

        if pos_voltage_mixed is not None:
            self.pos_voltage_mixed = pos_voltage_mixed
        if pos_charge_mixed is not None:
            self.pos_charge_mixed = pos_charge_mixed

        if self.pos_voltage_mixed is None:
            raise AttributeError((
                    'self.pos_voltage_mixed is None.'))

        if self.pos_charge_mixed is None:
            raise AttributeError((
                    'self.pos_charge_mixed is None.'))

        intersected_mixed_map = np.in1d(
                self.discrete_poisson.points_system_mixed,
                np.concatenate((self.pos_voltage_mixed,
                                self.pos_charge_mixed)))
        #TODO: Change this so it uses np.logical_not
        pos_default_mixed = np.delete(
                self.discrete_poisson.points_system_mixed,
                np.arange(len(self.discrete_poisson.points_system_mixed))[
                        intersected_mixed_map])
        if self.verbose:
            if len(pos_default_mixed) > 1:
                print('\n')
                print('#'*48)
                print('\n')
                warnings.warn((
                        '\n \n There are points belonging to mixed regions that'
                        + ' are not defined in pos_dirichlet_mixed nor'
                        + ' pos_neuman_mixed. \n  To those points'
                        + ' the default charge value will be associated'
                        + ', i.e. 0 \n'),
                        UserWarning)
                print('\n')
                print('#'*48)

        self.pos_charge_mixed = np.concatenate((
                self.pos_charge_mixed,
                pos_default_mixed))

        self.pos_voltage = np.concatenate((
                self.pos_voltage,
                self.pos_voltage_mixed)).astype(int)

        self.pos_charge = np.concatenate((
                self.pos_charge,
                self.pos_charge_mixed)).astype(int)

    def solve(self, solver_type='mumps',
              new_instance=False, **kwargs):
        '''
            Solve the linear system of equations using:
                i)  scipy.sparse.dsolve.spsolve
                ii) kwant.linalg.mumps

            solver_type: string (default value -> 'mumps')

                'spsolve': Uses scipy.sparse.dsolve.spsolve
                    kwargs: None

                'mumps': Uses kwant.linalg.mumps
                    kwargs: mums_solver_obj: Instance of
                                kwant.linalg.mumps.MUMPSContext

                            factorize: bool (default value -> True)
                                If True LU decomposition is made
                                If not the previous LU decomposition is used

                            verbose: bool (default value -> False)
                                Parameter sent to kwant.linalg.mumps.\
                                MUMPSContext
            new_instance: bool.
                By default True, which initializes a new instance of
                self.solver_instance. If one wants to use a previous
                instance of solver to, e.g. avoid re-decomposition,
                one can set new_instance to False.

            TODO: discuss with Christoph the use of new_instance
        '''

        solver_options = {'spsolve': solver.SpsolveScipy,
                          'mumps': solver.Mumps}
        if (new_instance
            or (self.solver_instance  is None)
            or (self.solver is not solver_type)):
            #print('New instance created') #TODO: remove this once test done
            self.solver_instance = solver_options[solver_type]()
        elif ((not new_instance) and
              (self.solver_instance is None)):
            raise ValueError('There are no instances of solver')

        self.solver = solver_type
        return self.solver_instance(self, **kwargs)

    def update(self, voltage_val=None, charge_val=None, mixed_val=None,
               pos_voltage_mixed=None, pos_charge_mixed=None):
        '''
        Update the system of equations without recalculating the
        sparse matrices.
        Parameters:
        -----------
            Sames as for __init__().
        Returns:
        --------
            None
        This update does not require re-factorization
        '''

        if (pos_voltage_mixed is not None or pos_charge_mixed is not None):
            # Matrix order will have to change. Not implemented yet.
            print('Not implemented')

        # TODO: See how to bookkeep
        if voltage_val is not None:
            self.voltage_val = to_piecewise(voltage_val)

        if charge_val is not None:
            self.charge_val = to_piecewise(charge_val)

        if mixed_val is not None:
            self.mixed_val = to_piecewise(mixed_val)


        self.points_input = make_points_input_vector(
                self.discrete_poisson,
                dirichlet_val=self.voltage_val,
                charge_val=self.charge_val,
                mixed_val=self.mixed_val)

        if self.is_charge_density:
            self.points_input[self.pos_charge] = (
                    (self.points_input[self.pos_charge]
                     * self.discrete_poisson.mesh.points_hypervolume[
                             self.pos_charge]))

        self.rhs_vector_sparse = make_rhs_value_sparse(
                self.points_input, self.pos_voltage, self.pos_charge,
                self.discrete_poisson)

        self.rhs_sparse = self.rhs_sparse_mat * self.rhs_vector_sparse.T

    def save_to_vtk(self, filename, encoding='ascii', vtk_version='1.0',
                    xml_declaration=False):
        '''
            Saves data into a .vtu file format aceepted in ParaView
        '''

        if self.points_charge is None or self.points_voltage is None:
            raise warnings.Warning('No voltage or charge values \
                                    have been calculated. \
                                    Solve the system before saving it. ')

        save.points_vtk(
                filepath=filename,
                coordinates=self.discrete_poisson.mesh.points[
                        self.discrete_poisson.points_system],
                points_data={
                        'Charge_density': (
                                self.points_charge[
                                        self.discrete_poisson.points_system],
                                'Scalar'),
                        'Voltage': (
                                self.points_voltage[
                                        self.discrete_poisson.points_system],
                                'Scalar')},
                vtk_data_format=encoding, vkt_version=vtk_version,
                xml_declaration=xml_declaration)

    def plot_3d(self, plot_='both', titles=None,
                scale_factor=1, **kwargs):
        '''
            Wrapper around poisson.plot_linear_problem_3d
            Parameters:
            ------------
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
        return p_plt.plot_linear_problem_3d(
                self, plot_=plot_, titles=titles,
                scale_factor=scale_factor, **kwargs)

    def plot_cut_1d(self, directions, plot_type='both', bbox=None, npoints=200,
                    figsize=(10, 10), interpolation_data=0, decimals=5):
        '''
            Wrapper around poisson.plot_linear_problem_1d
            Parameters:
            -----------
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
        return p_plt.plot_linear_problem_1d(
                self, directions=directions, plot_type=plot_type,
                bbox=bbox, npoints=npoints, figsize=figsize,
                interpolation_data=interpolation_data, decimals=decimals)


    def plot_cut_2d(self, plot_type='both', direction=2,
                    xlabel=None, ylabel=None, npoints=(100, 100),
                    bbox=None, figsize=(11, 11)):
        '''
            Wrapper around poisson.plot_toolbox.sliced_values_2d.
            Plot a 2D imshow plot of the charge and voltage dispersion.

            Parameters:
            ------------
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
        return p_plt.plot_linear_problem_2d(
                self, plot_type=plot_type, direction=direction,
                xlabel=xlabel, ylabel=ylabel, npoints=npoints, bbox=bbox,
                figsize=figsize)
