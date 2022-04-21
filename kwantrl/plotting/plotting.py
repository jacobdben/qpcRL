from kwantrl.datahandling.datahandling import datahandler
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np


#print(plt.style.available) for available styles

class plotter_class():
    def __init__(self,run_id,data_path=None,legacy_data=False) -> None:
        self.datahandler_ins=datahandler(data_path)
        self.data,self.starting_point=self.datahandler_ins.load_transformed_data(run_id,legacy=legacy_data)
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

    def loss(self,**kwargs):
        fig,ax=plt.subplots()
        self.__plot_1d(ax,self.data['loss'])
        ax.set_xlabel('function call')
        ax.set_ylabel('loss')
        ax.set_title(f'CMA-id:{self.run_id}, loss')
        return fig,ax

    def iter_loss(self,):
        print('unimplemented')

    def result(self,):
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

    def wave_func(self,axis_point,QPC=None,starting_run=False,run_num=None):
        if starting_run:
            temp_data=self.data
        else:
            temp_data=self.data

        if run_num==None:
            run_num=self.best_call

        if QPC==None:
            try:
                QPC=self.datahandler_ins.load_qpc(self.run_id)
            except:
                print(f'No QPC instance provided and instance is not saved for run_id:{self.run_id}')
        
        if isinstance(axis_point,(list,np.ndarray)):
            return [self.wave_func(axis_p) for axis_p in axis_point]
        
        which_point=np.argmin(np.abs(np.array(temp_data['xaxis'][run_num])-axis_point))
        point_value=temp_data["xaxis"][run_num][which_point]
        print(f'plotting wavefunction at x-axis value: {point_value:.2f}')
        voltages=temp_data['voltages'][run_num][which_point]

        QPC.set_all_pixels(voltages)
        ax=QPC.wave_func()
        ax.set_title(f'CMA-id:{self.run_id}, WF at xaxis: {point_value:.2f}, with conductance: {temp_data["staircase"][run_num][which_point]:.2f}')
        return ax
    
    def voltages(self,):
        best_voltages=np.array(self.data['voltages'][np.argmin(self.data['loss'])])
        labels=[str(i+1) for i in range(9)]
        colors=[]
        fig,ax=plt.subplots()
        for i in range(9):
            p=ax.plot(self.data['xaxis'][0],best_voltages[:,i],label=labels[i])
            colors.append(p[0].get_color())
        ax.set_xlabel(r"$V_{avg} \ [V]$")
        ax.set_ylabel(r"$V_{pix} \ [V]$")
        ax.grid()

        #make the inset axis plot
        axin=inset_axes(ax,width=1,height=1,loc=2)
        axin.set_xlim(0,3.25)
        axin.set_ylim(0,3.25)
        for i in range(3):
            for j in range(3):
                rect=Rectangle((j+0.25,i+0.25),0.75,0.75)
                pc=PatchCollection([rect],facecolor=colors[i*3+j])
                axin.text(x=j+0.26,y=i+0.3,s=str(1+i*3+j))
                axin.add_collection(pc)
        axin.axes.xaxis.set_visible(False)
        axin.axes.yaxis.set_visible(False)