from kwantrl.datahandling.datahandling import datahandler
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


#print(plt.style.available) for available styles

class plotter_class():
    def __init__(self,run_id) -> None:
        self.data,self.starting_point=datahandler().load_transformed_data(run_id)
        self.run_id=run_id

        self.best_call=np.argmin(self.data['loss'])
        self.best_loss=self.data['loss'][self.best_call]
        # mpl.rcParams['figure.dpi']=300
        mpl.rcParams['figure.dpi']=200
        mpl.rcParams["figure.facecolor"]='white'
        mpl.rcParams["axes.facecolor"]='white'
        mpl.rcParams["savefig.facecolor"]='white'
        # plt.style.use('classic')

    def __plot_1d(self,ax,y_data,x_data=None,**kwargs):
        if x_data==None:
            ax.plot(y_data,**kwargs)
        else:
            ax.plot(x_data,y_data,**kwargs)

    def plot_loss(self,**kwargs):
        fig,ax=plt.subplots()
        self.__plot_1d(ax,self.data['loss'])
        ax.set_xlabel('function call')
        ax.set_ylabel('loss')
        ax.set_title(f'CMA-id:{self.run_id}, loss')
        return fig,ax

    def plot_iter_loss(self,):
        print('unimplemented')

    def plot_result(self,):
        fig,ax=plt.subplots()
        if 'xaxis' in self.data:
            self.__plot_1d(ax,self.data['staircase'][self.best_call],self.data['xaxis'][self.best_call],label=f'B-loss:{self.best_loss:.3f}')
            self.__plot_1d(ax,self.starting_point['staircase'],self.starting_point['xaxis'],label=f"S-loss:{self.starting_point['loss']:.3f}")
        else:
            self.__plot_1d(ax,self.data['staircase'][self.best_call],label=f'loss:{self.best_loss:.3f}')
            self.__plot_1d(ax,self.starting_point['staircase'],label=f"loss:{self.starting_point['loss']:.3f}")
        ax.legend(loc='upper left')
        ax.set_xlabel(r'$V_{avg}$ [V]')
        ax.set_ylabel('Conductance')
        ax.set_title(f'CMA-id:{self.run_id}, result')
        ax.grid(axis='y')
        return fig,ax

    def plot_wave_func(self,axis_point,QPC=None,starting_run=False,run_num=None):
        if starting_run:
            temp_data=self.data
        else:
            temp_data=self.data

        if run_num==None:
            run_num=self.best_call

        if QPC==None:
            try:
                QPC=datahandler().load_qpc(self.run_id)
            except:
                print(f'No QPC instance provided and instance is not saved for run_id:{self.run_id}')
        
        if isinstance(axis_point,(list,np.ndarray)):
            return [self.plot_wave_func(axis_p) for axis_p in axis_point]
        
        which_point=np.argmin(np.abs(np.array(temp_data['xaxis'][run_num])-axis_point))
        point_value=temp_data["xaxis"][run_num][which_point]
        print(f'plotting wavefunction at x-axis value: {point_value:.2f}')
        voltages=temp_data['voltages'][run_num][which_point]

        QPC.set_all_pixels(voltages)
        fig,ax=QPC.wave_func()
        ax.set_title(f'CMA-id:{self.run_id}, WF at xaxis: {point_value:.2f}, with conductance: {temp_data["staircase"][run_num][which_point]:.2f}')
        return fig,ax
    
