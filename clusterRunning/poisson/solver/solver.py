'''
    ABC class Solver that is passed to linear_problem.
    When called, if a LinearProblem isntance is given,
    it returns two numpy arrays points_charge and points_voltage.
'''


__all__ = ['Solver',
           'SpsolveScipy',
           'Mumps']


from abc import ABC, abstractmethod

from scipy.sparse.linalg import dsolve

import numpy as np

from poisson.tools import post_process
from poisson.discrete import LinearProblem

class Solver(ABC):
    '''
        ABC Solver class.
        Defines the structure of the class that solves the system of equations
        defined in SysEquations.
    '''

    version = '0.1'
    print('Solver version {}'.format(version))

    def __init__(self):
        super().__init__()
        self.points_out = None
        self.points_input = None

        self.points_charge = None
        self.points_voltage = None

    @abstractmethod
    def solve(self, linear_problem,  **kwargs):
        '''
            Function returning the output of the solver.
            I.e. for a equation of type A=xB, where x are
            unknows, returns x.
            x is a np.ndarray

            This function is called when a subclass of Solver is called.

        '''
        pass

    def transform_out(self, output, linear_problem):
        '''
            Takes an output and self.input in order to merge input and ouput
            into points_voltage and points_charge. Also do any nescessary
            treatemlent, i.e. normalization, of the output and input.

            output: numpy array.
        '''

        self.points_input = linear_problem.points_input
        self.points_out = np.zeros(
                len(linear_problem.discrete_poisson.mesh.points)) * np.NAN

        self.points_out[linear_problem.pos_charge] = (
                output[len(linear_problem.pos_voltage): len(
                        linear_problem.pos_voltage)
                        + len(linear_problem.pos_charge)]
                / linear_problem.discrete_poisson.capa_normalization)

        if linear_problem.is_charge_density:
            self.points_out[linear_problem.pos_voltage] = (
                    output[0:len(linear_problem.pos_voltage)]
                    / linear_problem.discrete_poisson.mesh.points_hypervolume[
                                                linear_problem.pos_voltage])

            self.points_input[linear_problem.pos_charge] = (
                    (self.points_input[linear_problem.pos_charge]
                     / linear_problem.discrete_poisson.mesh.points_hypervolume[
                             linear_problem.pos_charge]))

        else:
            self.points_out[linear_problem.pos_voltage] = output[
                    0:len(linear_problem.pos_voltage)]

        linear_problem.points_output = self.points_out

        self.points_charge = post_process.charge_from_all(
            linear_problem_inst=linear_problem,
            discrete_poisson_inst=linear_problem.discrete_poisson)

        self.points_voltage = post_process.voltage_from_all(
            linear_problem_inst=linear_problem,
            discrete_poisson_inst=linear_problem.discrete_poisson)

        linear_problem.points_charge = self.points_charge
        linear_problem.points_voltage = self.points_voltage

        return self.points_charge, self.points_voltage

    def __call__(self, linear_problem=None, **kwargs):
        if linear_problem is None:
            return linear_problem
        elif isinstance(linear_problem, LinearProblem):

            out = self.solve(linear_problem, **kwargs)

            if not isinstance(out, np.ndarray):
                raise TypeError(('Solve function not well defined.'
                                 + 'It returns {0} when it should return'
                                 + 'a np.ndarray'.format(type(out))))

            if (out.shape[0]
                == linear_problem.discrete_poisson.points_system.shape[0]):
                return self.transform_out(out, linear_problem)
            else:
                raise ValueError(('Solve function not well defined.'
                                  + 'The returned np.ndarray.shape[0] is'
                                  + ' different than the number of points in'
                                  + 'the system.'))
        else:
            raise TypeError(('linear_problem must be an instance of '
                            + 'poisson.LinearProblem'))

class SpsolveScipy(Solver):
    '''
        Wrapper around scipy.sparse.linalg.dsolve.spsolve
    '''

    def solve(self, linear_problem):
        '''
            Parameters:
            -----------
                linear_problem: Instance of poisson.LinearProblem

            Returns:
            --------
                out: np.ndarray
        '''
        print('Solver : scipy.sparse.dsolve.spsolve')
        if linear_problem.lhs_sparse.getformat() is 'coo':
            linear_problem.lhs_sparse = linear_problem.lhs_sparse.tocsr()
        if linear_problem.rhs_sparse.getformat() is 'coo':
            linear_problem.rhs_sparse = linear_problem.rhs_sparse.tocsr()

        return dsolve.spsolve(linear_problem.lhs_sparse,
                              linear_problem.rhs_sparse,
                              use_umfpack=True)


class Mumps(Solver):
    '''
        Wrapper around kwant.linalg.mumps
    '''

    def __init__(self):
        super().__init__()
        self.mumps_solver_obj = None

    def solve(self, linear_problem, mumps_solver_obj=None,
                    factorize=True, verbose=False):
        '''
            Uses kwant interface of the MUMPS parse solver library

            Parameters:
            -----------
                linear_problem: Instance of poisson.LinearProblem

                mums_solver_obj: Instance of kwant.linalg.mumps.MUMPSContext

                facpos_voltage_mixedtorize: bool
                    If True LU decomposition is made (default)
                    If not the previous LU decomposition is used

                verbose: bool
                    Parameter sent to kwant.linalg.mumps.MUMPSContext.
                    Default False.

            Returns:
            --------
                out: np.ndarray
        '''

        #print('Solver : MUMPS')
        if ((mumps_solver_obj is None)
            and (self.mumps_solver_obj is None)):
            try:
                self.mumps_solver_obj = mumps.MUMPSContext(
                        verbose=verbose)
            except NameError:
                from kwant.linalg import mumps
                self.mumps_solver_obj = mumps.MUMPSContext(
                        verbose=verbose)

        elif mumps_solver_obj is not None:
            self.mumps_solver_obj = mumps_solver_obj

        if linear_problem.lhs_sparse.getformat() is not 'coo':
            linear_problem.lhs_sparse = linear_problem.lhs_sparse.tocoo()
        if linear_problem.rhs_sparse.getformat() is not 'coo':
            linear_problem.rhs_sparse = linear_problem.rhs_sparse.tocoo()


        if factorize:
            self.mumps_solver_obj.factor(linear_problem.lhs_sparse)

        out = self.mumps_solver_obj.solve(linear_problem.rhs_sparse)

        return out.real[:, 0]



