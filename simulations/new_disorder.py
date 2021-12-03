#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def make_disorder(L,W,length_scale,random_seed=2):
    rng=np.random.RandomState(random_seed)
    #np.random.seed(random_seed)
    xs=np.arange(start=np.floor(length_scale/2),stop=L,step=length_scale)
    ys=np.arange(start=np.floor(length_scale/2),stop=W,step=length_scale)
    XS,YS=np.meshgrid(xs,ys,indexing='ij')
    points=np.array([XS.flatten(),YS.flatten()]).T
    #disorder=np.random.uniform(-1,1,size=(len(xs),len(ys)))
    disorder=rng.uniform(-1,1,size=(len(xs),len(ys)))
            
    grid_x,grid_y=np.meshgrid(np.arange(L),np.arange(W),indexing='ij')
    
    
    disorder_int=griddata(points,disorder.flatten(),(grid_x,grid_y),method='cubic')

    #reset the random seed
    #np.random.seed()

    return np.nan_to_num(disorder_int)

def make_pixel_disorder(L,W,pixel_locations,random_seed=2):
    rng=np.random.RandomState(random_seed)
    vals=rng.uniform(-1,1,size=[3,3])
    disorder=np.zeros([L,W])
    for i in range(3):
        for j in range(3):
            val=vals[j,i]
            dims=pixel_locations[j+3*i][1:]
            disorder[int(np.round(dims[0])):int(np.round(dims[0]))+5,int(np.round(dims[2])):int(np.round(dims[2]))+5]=val
            
    return disorder

if __name__=="__main__":
    W=70
    L=120
    dis=make_disorder(L,W,5)*0.1
    print(dis.shape)
    # dis2=make_pixel_disorder(L,W)*0.1
    plt.figure()
    plt.imshow(dis,origin='lower')
    # plt.colorbar()
# %%
