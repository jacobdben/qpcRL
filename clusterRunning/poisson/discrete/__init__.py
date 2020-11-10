'''
    discrete module
'''


# each submodule imported here must have an __all__ variable
from .discrete_poisson import *
from .finite_volume import *
from .linear_problem import *


__all__ = (discrete_poisson.__all__
           + finite_volume.__all__
           + linear_problem.__all__)






