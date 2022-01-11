import numpy as np

# general implementation for modes up to ...

L_x=3
L_y=3
n,m=(2,2)
gate_locs=((-1,0,1),(-1,0,1))

#returns a list with all modes as lambda functions of x,y
def general_modes(n_max,m_max):
    A_list=[]
    B_list=[]
    C_list=[]
    D_list=[]
    
    for n in range(n_max+1):
        for m in range(m_max+1):
            A_list.append(cos_cos(n,m))
            B_list.append(cos_sin(n,m))
            C_list.append(sin_cos(n,m))
            D_list.append(sin_sin(n,m))
    return A_list+B_list+C_list+D_list
    
#function for each type of mode, return a lambda function that is specific to the n,m pair
def cos_cos(n,m):
    return lambda x,y: np.cos(2*np.pi*n*x/L_x)*np.cos(2*np.pi*m*y/L_y)
    
def cos_sin(n,m):
    return lambda x,y: np.cos(2*np.pi*n*x/L_x)*np.sin(2*np.pi*m*y/L_y)

def sin_cos(n,m):
    return lambda x,y: np.sin(2*np.pi*n*x/L_x)*np.cos(2*np.pi*m*y/L_y)

def sin_sin(n,m):
    return lambda x,y: np.sin(2*np.pi*n*x/L_x)*np.sin(2*np.pi*m*y/L_y)



modes_list=general_modes(n,m)

#for fourier_modes (parameters) and modes calculates the values at gate_locs
def fourier_to_potential_general(fourier_modes):
    xs=gate_locs[0]
    ys=gate_locs[1]
    XX,YY=np.meshgrid(xs,ys)
    
    modes=np.array([fourier_modes[i]*modes_list[i](XX,YY) for i in range(len(modes_list))])
    
    return modes, np.sum(modes,axis=0)


if __name__=="__main__":
    import matplotlib.pyplot as plt
    def title_gen(n,m):
        A_=[]
        B_=[]
        C_=[]
        D_=[]
        for i in range(n+1):
            for j in range(m+1):
                A_.append(r'$\alpha_{%i,%i}$'%(i,j))
                B_.append(r'$\beta_{%i,%i}$'%(i,j))
                C_.append(r'$\gamma_{%i,%i}$'%(i,j))
                D_.append(r'$\delta_{%i,%i}$'%(i,j))
    
        return A_+B_+C_+D_
    #length / period in x and y
    
    
    #generate modes up to n,m=2
    
    
    #generate random parameters,
    fourier_modes=np.random.uniform(-1,1,len(modes_list))
    # fourier_modes=np.ones(len(modes_list))
    
    #let the average be 0
    fourier_modes[0]=0
    
    gen_modes,gen_sum=fourier_to_potential_general(fourier_modes)
    
    plt.imshow(gen_sum,origin='lower')
    print(np.mean(gen_sum))
    
    #all plotting
    title=title_gen(n,m)
    
    
    im=plt.imshow(np.unique(gen_modes)[:,np.newaxis])
    shape=int(np.sqrt(gen_modes.shape[0]))
    fig,axes=plt.subplots(shape,shape)
    for i in range(shape):
        for j in range(shape):
            # axes[i,j].imshow(gen_modes[i,:,:],origin='lower')
            axes[i,j].imshow(gen_modes[i*shape+j,:,:],origin='lower',vmin=np.min(np.unique(gen_modes)),vmax=np.max(np.unique(gen_modes)))   
            axes[i,j].set_title(title[i*shape+j],y=0.6,x=0.30,color='red',fontsize=15)
            axes[i,j].xaxis.set_ticks([])
            axes[i,j].xaxis.set_ticklabels([])
            
            axes[i,j].yaxis.set_ticks([])
            axes[i,j].yaxis.set_ticklabels([])
            # axes[i,j].set_xticks([0,1,21])
            # axes[i,j].tick_params(axis='both',which='both',color='white',labelcolor='white')
    
    fig.colorbar(im,ax=axes.ravel().tolist())
    
    plt.figure()
    plt.imshow(gen_sum)
    plt.colorbar()