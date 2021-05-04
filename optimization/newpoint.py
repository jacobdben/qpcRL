
import cvxpy as cp
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
        x[i],p_temp=check_bound(bounds[i],x[i])
        penalty+=p_temp
        
    return x, penalty

# def fourier_modes(p):
#     # p are the V_kxky parameters, so the first one is set to the average gate voltage?
#     # the rest are optimized by cma-es
#     V=[]
#     V2=[]
#     i=0
#     kxs=np.arange(3)
#     kys=np.arange(3)
#     for y in range(3):
#         for x in range(3):
#             V.append(np.sum(np.array([np.exp(1j*2*np.pi/3*(kx*x+ky*y)) for kx in kxs for ky in kys if kx!=0 or ky!=0])*p[1:]).real)
#             V2.append(np.sum(np.array([np.exp(1j*2*np.pi/3*(kx*x+ky*y)) for kx in kxs for ky in kys])*p).real)
#             i+=1
#     return np.abs(V)**2,np.abs(V2)**2
"""
def fourier_modes(p):
    V=np.zeros((3,3), dtype=np.complex_)
    for x in range(3):
        for y in range(3):
            V[x,y] = np.sum([np.exp(1j*2*np.pi/3*kx*x) * np.exp(1j*2*np.pi/3*ky*y) * p[kx * 3 + ky] for kx in range(3) for ky in range(3)])
    return V

def fourier_modes_cos(p):
    V=np.zeros((3,3))
    for x in range(3):
        for y in range(3):
            V[x,y] = np.sum([np.cos(2*np.pi/3*kx*x) * np.cos(2*np.pi/3*ky*y) * p[kx * 3 + ky] for kx in range(3) for ky in range(3)])
    return V



def get_mode(kx,ky):
    L = 2
    V = np.zeros((3,3))
    for x in range(3):
        for y in range(3):
            Vt = np.exp(1j*2*np.pi/L*kx*x)*np.exp(1j*2*np.pi/L*ky*y)
            V[x,y] = np.real(Vt)
            
    return V

fig, ax = plt.subplots(3,3)
for kx in range(3):
    for ky in range(3):
        ax[kx,ky].matshow(get_mode(kx,ky))
        
        
"""        
        

# result1,pen=new_point(np.array([3,2,0,-7,0,0,1,0,1]))

# using scipy.optimize.minimize()
# def new_point2(point):
#     result=minimize(lambda x: np.sum((x-point)**2),x0=np.zeros(len(point)),constraints=[{"type":"eq","fun":(lambda x:np.sum(x))}])
#     print("distance={}".format(result.fun))
#     print("point={}".format(result.x))
#     print("mean={}".format(np.mean(result.x)))
#     return result.x,result.fun

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
