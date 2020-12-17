import sys
sys.path.append('../')
from simulations.pixel_array_sim_2 import pixelarrayQPC
from staircasiness import staircasiness
from datahandling import datahandler,save_optimization_dict,load_optimization_dict
from optimization.small_pixel_array_stepoptimization import step_optimization


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# 0.5 disorder experiment
"""
exp1=datahandler("small_pixel_array_0.5disorder")
start,stop,steps=exp1.dict['sweep']
disorder=0.5
outer_gates=-2

step_optimization(start=start,stop=stop,steps=steps,experiment_name='small_pixel_array_0.5disorder',outer_gates=outer_gates,disorder=disorder)

disordered_optimization_results=load_optimization_dict('small_pixel_array_0.5disorder')

QPC=pixelarrayQPC()
QPC.V1=outer_gates
QPC.V11=outer_gates
QPC.U0=disorder

for i in range(50):
    QPC.V2=disordered_optimization_results['x'][i][0]
    QPC.V3=disordered_optimization_results['x'][i][1]
    QPC.V4=disordered_optimization_results['x'][i][2]
    QPC.V5=disordered_optimization_results['x'][i][3]
    QPC.V6=disordered_optimization_results['x'][i][4]
    QPC.V7=disordered_optimization_results['x'][i][5]
    QPC.V8=disordered_optimization_results['x'][i][6]
    QPC.V9=disordered_optimization_results['x'][i][7]
    QPC.V10=disordered_optimization_results['x'][i][8]
    
    QPC.plot_potential()
"""

# higher energy 
experiment_name='no_disorder+high_energy'
disorder=0
outer_gates=-3
start,stop,steps=(-2.5,0,50)

step_optimization(start=start,stop=stop,steps=steps,experiment_name=experiment_name,outer_gates=outer_gates,disorder=disorder)

disordered_optimization_results=load_optimization_dict(experiment_name)

def complete_plot():
    sweep=np.linspace(start,stop,steps)
    
    plt.figure()
    exp1="no_disorder+high_energy"
    exp1dict=load_optimization_dict(exp1)
    plt.plot(sweep,exp1dict['trans'],label='Optimized, No Disorder')
    
    
    exp2="disorder+high_energy"
    exp2dict=load_optimization_dict(exp2)
    plt.plot(sweep,exp2dict['trans'],label='Optimized, With Disorder')
    
    
    QPC=pixelarrayQPC()
    QPC.V1=outer_gates
    QPC.V11=outer_gates
    QPC.U0=0
    
    results=[]
    for V in sweep:
        QPC.set_all_pixels(V)
        results.append(QPC.transmission())
    
    plt.plot(sweep,results,label='Not Optimized, No Disorder')
    
    QPC.U0=0.5
    
    results=[]
    for V in sweep:
        QPC.set_all_pixels(V)
        results.append(QPC.transmission())
    
    plt.plot(sweep,results,label='Not Optimized, With Disorder')
    
    plt.ylabel("Conductance")
    plt.xlabel("Avg Pixel Voltage [V]")
    plt.grid('on')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def setstep(step,QPC):
    QPC.V2=step[0]
    QPC.V3=step[1]
    QPC.V4=step[2]
    QPC.V5=step[3]
    QPC.V6=step[4]
    QPC.V7=step[5]
    QPC.V8=step[6]
    QPC.V9=step[7]
    QPC.V10=step[8]


def plot_potentials():
    
    QPC=pixelarrayQPC()
    QPC.V1=-3
    QPC.V11=-3

    exp1="no_disorder+high_energy"
    exp1dict=load_optimization_dict(exp1)

    exp2="disorder+high_energy"
    exp2dict=load_optimization_dict(exp2)
    
    common_voltages=np.linspace(-2.5,0,50)
    
    for i in range(50):
        fig,axes=plt.subplots(1,2)
        ax1=axes[0]
        QPC.U0=0

        setstep(exp1dict['x'][i],QPC)
        QPC.plot_potential(ax1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        # ax1.set_title('without')
        
        
        ax2=axes[1]
        QPC.U0=0.5
        setstep(exp2dict['x'][i],QPC)
        QPC.plot_potential(ax2)
        # ax2.set_title('with')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.tight_layout()
        plt.suptitle("Optimized Step %i"%i)   
        
        
        norm=mpl.colors.Normalize(vmin=0,vmax=1)
        sm=plt.cm.ScalarMappable(cmap='seismic',norm=norm)
        sm.set_array([])
        
        fig.subplots_adjust(right=0.85)
        cbar_ax=fig.add_axes([0.90,0.16,0.025,0.7])
        h=fig.colorbar(sm,cax=cbar_ax)
        cbar_ax.set_xlabel('')
        cbar_ax.set_ylabel('Arbitrary Energy Units')
        h.set_ticks([])
        plt.savefig("C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/figures/Disorder_comparison/{}".format(i))
    
        
        
        # 
        # 
        # 
        # 
        # 
        # 
        
        fig,axes=plt.subplots(1,2)
        ax1=axes[0]
        QPC.U0=0

        setstep(np.ones(9)*common_voltages[i],QPC)
        QPC.plot_potential(ax1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        # ax1.set_title('without')
        
        
        ax2=axes[1]
        QPC.U0=0.5
        setstep(np.ones(9)*common_voltages[i],QPC)
        QPC.plot_potential(ax2)
        # ax2.set_title('with')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.tight_layout()
        plt.suptitle("Non Optimized Step %i"%i)   
        
        
        norm=mpl.colors.Normalize(vmin=0,vmax=1)
        sm=plt.cm.ScalarMappable(cmap='seismic',norm=norm)
        sm.set_array([])
        
        fig.subplots_adjust(right=0.85)
        cbar_ax=fig.add_axes([0.90,0.16,0.025,0.7])
        h=fig.colorbar(sm,cax=cbar_ax)
        cbar_ax.set_xlabel('')
        cbar_ax.set_ylabel('Arbitrary Energy Units')
        h.set_ticks([])
        plt.savefig("C:/Users/Torbjørn/Google Drev/UNI/MastersProject/EverythingkwantRL/figures/Disorder_comparison_non_optimized/{}".format(i))
    
        
    


