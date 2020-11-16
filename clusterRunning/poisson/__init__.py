'''
    poisson package module
'''

from .solver import Solver
from .discrete import DiscretePoisson, LinearProblem, FiniteVolume
from .continuous import Shape, ContinuousGeometry, GridBuilder
from .tools import plot

__all__ = (['Solver',
           'DiscretePoisson',
           'LinearProblem',
           'FiniteVolume',
           'Shape',
           'ContinuousGeometry',
           'GridBuilder'])
