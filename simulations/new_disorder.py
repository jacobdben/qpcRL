
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def make_disorder(L,W,length_scale):
    np.random.seed(2)
    xs=np.arange(start=np.floor(length_scale/2),stop=L,step=length_scale)
    ys=np.arange(start=np.floor(length_scale/2),stop=W,step=length_scale)
    XS,YS=np.meshgrid(xs,ys)
    points=np.array([XS.flatten(),YS.flatten()]).T
    disorder=np.random.uniform(-1,1,size=(len(xs),len(ys)))

            
    grid_x,grid_y=np.meshgrid(np.arange(L),np.arange(W))
    
    
    disorder_int=griddata(points,disorder.flatten(),(grid_x,grid_y),method='cubic')


    return np.nan_to_num(disorder_int)

if __name__=="__main__":
    W=70
    L=60
    dis=make_disorder(L,W,5)*0.1
    plt.figure()
    plt.imshow(dis,origin='lower')
    # plt.colorbar()