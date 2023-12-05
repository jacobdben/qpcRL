import cvxpy as cp
from cvxpy.atoms.affine.vec import vec
import numpy as np



def new_point_array(x,bounds,offset=0):
    y=cp.Variable(x.shape)
    y.value=x.copy()

    objective=cp.Minimize(cp.sum_squares(x-y))
    constraints=[cp.sum(y,axis=1)/x.shape[1]-offset==np.zeros(x.shape[0]), #SUM TO 0
                    vec(y) <=bounds[1], #upper bound
                    -vec(y) <=abs(bounds[0])] #lower bound

    prob=cp.Problem(objective,constraints)
    prob.solve()

    new_point=y.value

    return new_point, cp.sum_squares(x-new_point).value
