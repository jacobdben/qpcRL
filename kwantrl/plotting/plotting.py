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
        
        
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(cov,mean, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    
    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    
    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics
    
    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor='r',
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
