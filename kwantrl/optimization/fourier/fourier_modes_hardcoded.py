
import numpy as np
import matplotlib.pyplot as plt
#specific to 3x3 gates up to n,m=1
var=2*np.pi/3 #3=L_x
gate_locs=((-1,0,1),(-1,0,1))
#hardcoded modes
modes_list=[lambda x,y : np.ones(shape=x.shape),      #a0,0
            lambda x,y : np.cos(var*y),               #a0,1
            lambda x,y : np.cos(var*x),               #a1,0
            lambda x,y : np.cos(var*x)*np.cos(var*y), #a1,1
            lambda x,y : np.sin(var*y),               #b0,1
            lambda x,y : np.cos(var*x)*np.sin(var*y), #b1,1
            lambda x,y : np.sin(var*x),               #c1,0
            lambda x,y : np.sin(var*x)*np.cos(var*y), #c1,1
            lambda x,y : np.sin(var*x)*np.sin(var*y)] #d1,1


# calculate the gate voltages based on fourier modes.
def fourier_to_potential(fourier_modes,gate_locs=gate_locs):
    xs=gate_locs[0] #gate locations
    ys=gate_locs[1]
    XX,YY=np.meshgrid(xs,ys) # make grid with x,y values
    
    #calculate the individual modes, when using the modes_list of lambda functions cant be done without the loop i think
    modes=np.array([fourier_modes[i]*modes_list[i](XX,YY) for i in range(len(modes_list))])
    return modes,np.sum(modes,axis=0)


def plot_fourier_modes(fourier_modes):
    if len(fourier_modes)==8:
        x=[0]
        x.extend[fourier_modes]

    modes,V_g=fourier_to_potential(fourier_modes)
    
    title=[r'$\alpha_{0,0}$',
           r'$\alpha_{0,1}$',
           r'$\alpha_{1,0}$',
           r'$\alpha_{1,1}$',
           r'$\beta_{0,1}$',
           r'$\beta_{1,1}$',
           r'$\gamma_{1,0}$',
           r'$\gamma_{1,1}$',
           r'$\delta_{1,1}$']
    #im is just for making a collected colorbar of all the plots.
    imfig=plt.figure()
    print(np.unique(modes))
    im=plt.imshow(np.unique(modes)[:,np.newaxis])
    # imfig.set_visible(False)
    
    fig,axes=plt.subplots(3,3)
    for i in range(3):
        for j in range(3):      
            axes[i,j].imshow(modes[i*3+j,:,:],origin='lower',vmin=np.min(np.unique(modes)),vmax=np.max(np.unique(modes)))   
            axes[i,j].set_title(title[i*3+j],y=0.7,x=0.25,color='red',fontsize=15)
            axes[i,j].set_xticks([0,1,2])
            axes[i,j].tick_params(axis='both',which='both',color='white',labelcolor='white')
    
    fig.colorbar(im,ax=axes.ravel().tolist())
    # fig.savefig(r'C:\Users\Torbjørn\Google Drev\UNI\MastersProject\Thesis\Figures\QPC chapter\algorithm/fourier.pdf',format='pdf')
    return fig,axes


def potential_to_fourier(Vg,gate_locs=gate_locs):
    xs=gate_locs[0] #gate locations
    ys=gate_locs[1]
    XX,YY=np.meshgrid(xs,ys) # make grid with x,y values
    
    potential_to_fourier_modes=[lambda x,y: np.mean(Vg), #a0,0
                                lambda x,y: 2/9*np.sum(Vg*np.cos(var*y)), #a0,1
                                lambda x,y: 2/9*np.sum(Vg*np.cos(var*x)), #a1,0
                                lambda x,y: 4/9*np.sum(Vg*np.cos(var*x)*np.cos(var*y)), #a1,1
                                lambda x,y: 2/9*np.sum(Vg*np.sin(var*y)), #b0,1
                                lambda x,y: 4/9*np.sum(Vg*np.cos(var*x)*np.sin(var*y)), #b1,1
                                lambda x,y: 2/9*np.sum(Vg*np.sin(var*x)), #c1,0
                                lambda x,y: 4/9*np.sum(Vg*np.sin(var*x)*np.cos(var*y)), #c1,1
                                lambda x,y: 4/9*np.sum(Vg*np.sin(var*x)*np.sin(var*y))]
    parameters=[]
    for i in range(9):
        parameters.append(potential_to_fourier_modes[i](XX,YY))
    return parameters
# fourier_modes=np.ones(9)
if __name__=="__main__":
    import matplotlib
    matplotlib.rcParams['figure.dpi']=300

    # fourier_modes=np.random.uniform(-1,1,9)
    
    fourier_modes=np.ones(9)
    # fourier_modes[1]=0
    # fourier_modes[7]=1
    # fourier_modes[0]=0
    modes,V_g=fourier_to_potential(fourier_modes,gate_locs)
    
    modes2=potential_to_fourier(V_g)
    # print(fourier_modes-modes2)
    plot=True
    if plot:
        plot_fourier_modes(fourier_modes)
        
        # plot the combined gate voltages
        plt.figure()
        plt.imshow(V_g,origin='lower')
        print(np.mean(V_g)-fourier_modes[0])
        plt.colorbar(label='[V]')
        
        plt.xticks(ticks=[0,1,2],labels=['-1','0','1'])
        plt.yticks(ticks=[0,1,2],labels=['-1','0','1'])
    
    

