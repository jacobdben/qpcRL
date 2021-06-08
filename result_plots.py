from datahandling.datahandling import load_cma_data, unpack_cma_data
import numpy as np
import matplotlib.pyplot as plt

from optimization.fourier.fourier_modes_hardcoded import fourier_to_potential, plot_fourier_modes
from lossfunctions.staircasiness import staircasiness
from optimization.newpoint import new_point
from simulations.pixel_array_sim_2 import pixelarrayQPC
    
stairs=staircasiness()

QPC=pixelarrayQPC()
QPC.U0=0.1
QPC.V1=-4
QPC.v11=-4

def plot_losses(losses,losslabels,iter_loss=False):
    fig,ax=plt.subplots()
    if iter_loss:
        for i in range(len(losses)):
            iter_loss=[np.min(losses[i][j:j+10]) for j in np.arange(0,len(losses[i])-10,10)]
            mini=iter_loss[0]
            for k in range(len(iter_loss)):
                if iter_loss[k]<=mini:
                    mini=iter_loss[k]
                else:
                    iter_loss[k]=mini
            # iter_loss=[np.min(losses[i][j:j+10]) for j in np.arange(0,len(losses[i])-10,10)]
            ax.plot(range(len(iter_loss)),iter_loss,label=losslabels[i])
    else:
        for i in range(len(losses)):
            ax.plot(losses[i],label=losslabels[i])
    ax.set_xlabel('iteration',fontsize=18)
    ax.set_ylabel('loss',fontsize=18)
    ax.legend()
    return fig,ax

def plot_staircases(staircases,labels,offset=0):
    fig,ax=plt.subplots()
    for i in range(len(staircases)):
        ax.plot(common_voltages,offset*i+np.array(staircases[i]),label=labels[i])
        
    maximum_value=np.max(np.array(staircases))-1
    if offset!=0:
        maximum_value+=len(staircases)*offset
        
    y_ticks=np.arange(0,int(maximum_value),1)
    plt.yticks(y_ticks)
    plt.xlabel('avg voltage',fontsize=18)
    plt.ylabel('conductance',fontsize=18)
    plt.legend()
    plt.grid()
    return fig,ax

def get_best_run(data):
    for i in range(data.shape[0]):
        loss,staircases,voltages,xs=data
        
        best_loss=np.argmin(xs)
    
        return_list=[]
        if xs!=None:
            return_list.append([loss[best_loss],staircases[best_loss],voltages[best_loss],xs[best_loss]])
        else:
            return_list.append([ loss[best_loss],staircases[best_loss],voltages[best_loss],None])
    return return_list

def plot_current_and_wavefunc(voltages,titles):
    voltages.append(np.zeros(len(voltages[0])))
    titles.append('no voltages')
    for i in range(len(voltages)):
        # print(voltages)
        QPC.set_all_pixels(voltages[i])
        fig,ax=QPC.plot_current(1000)
        ax.set_title(titles[i])
        
        fig,ax=QPC.wave_func(lead=0)
        ax.set_title(titles[i])
        
def plot_wavefunc_steps(voltages,common_voltages,location):
    import imageio
    transmissions=[]
    frames=[]
    for i in range(len(common_voltages)):
        fig,axes=plt.subplots(1,2,gridspec_kw={'width_ratios': [1, 2.5]})
        QPC.set_all_pixels(common_voltages[i]+voltages)
        transmission=QPC.transmission()
        transmissions.append(transmission)
        if i>0:
            axes[0].plot(common_voltages[:i+1],transmissions)
        axes[1].set_title('Wave Function')
        axes[0].scatter(common_voltages[i],transmission,c='r')
        _,h=QPC.wave_func(lead=0,ax=axes[1],plot_both=True)
        axes[0].set_title("Conductance:{:.2f}".format(transmission))
        fig.colorbar(h)
        fig.savefig(location+str(i))
        
        frames.append(imageio.imread(location+str(i)+".png"))
    
    imageio.mimsave(location+"fingif.gif",frames,duration=0.2)

              

def plot_voltages(voltages,title):
    fig,ax=plt.subplots()
    image=ax.imshow(np.array(voltages).reshape((3,3)),origin='lower')
    plt.colorbar(image,label='V')
    ax.set_title(title)
    
def plot_potentials(voltages,titles):
    voltages.append(np.zeros(len(voltages[0])))
    titles.append('no voltages')
    for i in range(len(voltages)):
        # print(voltages)
        QPC.set_all_pixels(voltages[i])
        fig,ax,vals=QPC.plot_potential(bounds=((20,40),(25,45)))
        ax.set_title(titles[i])
        
        print(titles[i]+":{:.5f}".format(np.std(vals)))


def plot_everything(datadicts,labels,iter_loss=False,offset=0,
                    loss=False,
                    potentials=False,
                    current=False,
                    ):
    
    data=np.array([unpack_cma_data(datadict) for datadict in datadicts])
    starting_points=[unpack_cma_data(datadict,starting_point=True) for datadict in datadicts]
    
    # check starting points are all the same
    plt.figure()
    plt.title('starting staircases')
    for i in range(len(starting_points)):
        plt.plot(starting_points[i][1],label=labels[i])
    plt.legend()
    
    losses=data[:,0].tolist()
    new_losses=recalculate_loss(data[:,1])
    
    # best_runs=[np.argmin(losses[i]) for i in range(len(losses))]
    best_runs2=[np.argmin(new_losses[i]) for i in range(len(new_losses))]
    
    # best_loss=[losses[i][np.argmin(losses[i])] for i in range(len(losses))]
    # best_loss2=[new_losses[i][np.argmin(new_losses[i])] for i in range(len(new_losses))]
    starting_staircase=[starting_points[0][1]]
    staircases=[data[i,1][best_run] for i,best_run in enumerate(best_runs2)] 
    staircases=staircases+starting_staircase
    
    voltages=[data[i,2][best_run] for i,best_run in enumerate(best_runs2)] 
    xs=[(data[i,3][best_run],i) for i,best_run in enumerate(best_runs2) if data[i,3]!=None] 
    
    
    
    fig1,ax1=plot_losses(losses,labels,iter_loss=iter_loss)
    ax1.set_title('optimization loss function')
    fig2,ax2=plot_losses(new_losses,labels,iter_loss=iter_loss)
    ax2.set_title('recalculated loss function')
    
    starting_loss_label=['default : {:.3f}'.format(starting_points[0][0])]
    loss_labels=[labels[i]+' : {:.3f}'.format(new_losses[i][best_run]) for i,best_run in enumerate(best_runs2)]
    loss_labels=loss_labels+starting_loss_label
    plot_staircases(staircases,loss_labels,offset)
    
    for i in range(len(voltages)):
        plot_voltages(voltages[i],title=labels[i])
        
    plot_potentials(voltages, labels)
    plot_current_and_wavefunc(voltages, labels)
    print('checking that fourier modes give stated voltages:')
    for x,i in xs:
        fig,ax=plot_fourier_modes(x)
        fig.suptitle(labels[i],fontsize=18)
        
        # check modes give correct voltages
        # plt.figure()
        # plt.title(labels[i])
        bounds=(-1,1)
        check_voltage=fourier_to_potential(x)[1]
        
        print(new_point(check_voltage.ravel(),bounds)[0]-np.array(voltages[i]))
        # plt.colorbar()
    
def recalculate_loss(staircases):
    return_list=[]
    for i in range(len(staircases)):
        return_list.append([stairs.deriv_metric_zeros1(staircases[i][j]) for j in range(len(staircases[i]))])
    return return_list

if __name__=="__main__":    
    # things to plot
    plot_loss=False
    plot_staircase=False
    plot_voltage=False
    plot_potential=False
    plot_current=False
    plot_modes=False
    plot_wave_func_gif=True
    
    #cluster start_point runs
    common_voltages=np.linspace(-3,0,100)
    cluster_data_path="C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/saved_data_cluster"
    run_ids=np.arange(14,18)
    data_dicts=[load_cma_data(run_id,cluster_data_path) for run_id in run_ids]
    labels=[str(run_id) for run_id in run_ids]
    # plot_everything(data_dicts,data_labels,iter_loss=True,offset=1)

    
    data=np.array([unpack_cma_data(datadict) for datadict in data_dicts])
    starting_points=[unpack_cma_data(datadict,starting_point=True) for datadict in data_dicts]
    
    # check starting points are all the same
    # plt.figure()
    # plt.title('starting staircases')
    # for i in range(len(starting_points)):
    #     plt.plot(starting_points[i][1],label=labels[i])
    # plt.legend()
    
    losses=data[:,0].tolist()
    new_losses=recalculate_loss(data[:,1])
    

    best_runs2=[np.argmin(new_losses[i]) for i in range(len(new_losses))]
    
    starting_staircase=[starting_points[0][1]]
    staircases=[data[i,1][best_run] for i,best_run in enumerate(best_runs2)] 
    staircases=staircases+starting_staircase
    
    voltages=[data[i,2][best_run] for i,best_run in enumerate(best_runs2)] 
    xs=[(data[i,3][best_run],i) for i,best_run in enumerate(best_runs2) if data[i,3]!=None] 
    
    
    if plot_loss:
        fig1,ax1=plot_losses(losses,labels,iter_loss=True)
        ax1.set_title('optimization loss function')
        fig2,ax2=plot_losses(new_losses,labels,iter_loss=True)
        ax2.set_title('recalculated loss function')
    
    starting_loss_label=['default : {:.3f}'.format(starting_points[0][0])]
    loss_labels=[labels[i]+' : {:.3f}'.format(new_losses[i][best_run]) for i,best_run in enumerate(best_runs2)]
    loss_labels=loss_labels+starting_loss_label
    if plot_staircase:
        plot_staircases(staircases,loss_labels,offset=1)
    
    if plot_voltage:
        for i in range(len(voltages)):
            plot_voltages(voltages[i],title=labels[i])
    if plot_potential:   
        plot_potentials(voltages, labels)
    if plot_current:
        plot_current_and_wavefunc(voltages, labels)
    if plot_modes:
        print('checking that fourier modes give stated voltages:')
        for x,i in xs:
            fig,ax=plot_fourier_modes(x)
            fig.suptitle(labels[i],fontsize=18)
            
            # check modes give correct voltages
            # plt.figure()
            # plt.title(labels[i])
            bounds=(-1,1)
            check_voltage=fourier_to_potential(x)[1]
            
            print(new_point(check_voltage.ravel(),bounds)[0]-np.array(voltages[i]))


    if plot_wave_func_gif:
        
        plot_wavefunc_steps(voltages[-1],common_voltages,
                            "C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/figures/wave_func_gif"+"/17_3/")

