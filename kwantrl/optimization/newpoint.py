import cvxpy as cp
from cvxpy.atoms.affine.vec import vec
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def new_point(x,bounds=(-1.5,1.5),offset=0):
    y=cp.Variable(len(x))
    y.value=x.copy()
    
    objective=cp.Minimize(cp.sum_squares(x-y))
    constraints=[cp.sum(y)/len(x)-offset==0, #SUM TO 0
                  np.eye(len(x)) @ y <=bounds[1], #upper bound
                  -np.eye(len(x)) @ y <=abs(bounds[0])] #lower bound
    
    prob=cp.Problem(objective,constraints)
    prob.solve()
    # print(result)
    new_point=y.value
    # print(new_point)
    return new_point, cp.sum_squares(x-new_point).value

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
    # print(new_point)
    return new_point, cp.sum_squares(x-new_point).value


def check_bound(bound,x):
    penalty=0
    if not bound[0]<=x:
        penalty+=(bound[0]-x)**2
        x=bound[0]
    if not x<=bound[1]:
        penalty+=(bound[1]-x)**2
        x=bound[1]
    return x,penalty


def simple_new_point(x,bounds):
    penalty=0
    for i in range(len(x)):
        x[i],p_temp=check_bound(bounds,x[i])
        penalty+=p_temp
        
    return x, penalty

if __name__=="__main__":
    repeats=10
    c=1
    plot=False
    for i in range(repeats):
        point=np.random.uniform(-1,1,size=9)
        # print('start={} '.format(point))
        print('start with sum={}'.format(np.sum(point)))
        # x,dist=new_point2(point)
        result1,pen=new_point(point)
        # print("end: {}".format(result1))
        print("end with penalty={}".format(c*pen))
        print("and sum ={}".format(np.sum(result1)))

        # if plot:
        #     plt.scatter(point[0],point[1],c='r')
        #     plt.scatter(x[0],x[1],c='g')
        #     plt.plot([point[0],x[0]],[point[1],x[1]],'b-.')
