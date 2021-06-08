import matplotlib.pyplot as plt
import numpy as np


from datahandling.datahandling import load_cma_output
from optimization.newpoint import new_point
 
def plot_run(QPC,data_path,run_number,common_voltages,bounds,staircasiness,pfactor):
    fitness,recentbestxs,xbest=load_cma_output(data_path,run_number)
    xbest,penalty=new_point(xbest,bounds=bounds)
    print("penalty from projection: %.4f" %(penalty*pfactor))
    
    result=[]
    baseline=[]
    for avg_voltage in common_voltages:
        QPC.set_all_pixels(xbest+avg_voltage)
        result.append(QPC.transmission())
        
        QPC.set_all_pixels(avg_voltage)
        baseline.append(QPC.transmission())
        
    plt.figure()
    plt.plot(common_voltages,result,label="Optimized: {:.4f}".format(staircasiness.histogram(result)+pfactor*penalty))
    plt.plot(common_voltages,baseline,label="Not Optimized: {:.4f}".format(staircasiness.histogram(baseline)))
    plt.xlabel("Avg voltage")
    plt.ylabel("Conductance")
    plt.grid('on')
    plt.legend()
    

def plot_potentials(QPC,data_path,run_number,common_voltages,bounds,staircasiness,pfactor,section=((19,41),(25,45))):
    fitness,recentbestxs,xbest=load_cma_output(data_path,run_number)
    xbest,penalty=new_point(xbest,bounds=bounds)
    if section==None:
        section=((0,60),(0,70))
    fig,ax=plt.subplots() 
    ax.set_title("Optimized at avg_voltage: {:.2f}".format(common_voltages[14])) 
    QPC.set_all_pixels(xbest+common_voltages[14])
    optimized_mid,p1=QPC.plot_potential_section(bounds=section,ax=ax)
    plt.colorbar(p1)
    
    fig,ax=plt.subplots() 
    ax.set_title("Optimized at avg_voltage: {:.2f}".format(common_voltages[0])) 
    QPC.set_all_pixels(xbest+common_voltages[0])
    optimized_start,p1=QPC.plot_potential_section(bounds=section,ax=ax)
    plt.colorbar(p1)
    
    fig,ax=plt.subplots() 
    ax.set_title("Not Optimized at avg_voltage: {:.2f}".format(common_voltages[14])) 
    QPC.set_all_pixels(common_voltages[14])
    not_optimized_mid,p1=QPC.plot_potential_section(bounds=section,ax=ax)
    plt.colorbar(p1)
    
    fig,ax=plt.subplots() 
    ax.set_title("Not Optimized at avg_voltage: {:.2f}".format(common_voltages[0])) 
    QPC.set_all_pixels(common_voltages[0])
    not_optimized_start,p1=QPC.plot_potential_section(bounds=section,ax=ax)
    plt.colorbar(p1)
    
    
    # fig,ax=plt.subplots()
    # # ax.set_title("Not Optimized at avg_voltage: {:.2f}".format(common_voltages[0])) 
    # plt.imshow((optimized_mid-not_optimized_start).T,origin='lower',extent=(section[0][0],section[0][1],section[1][0],section[1][1]))
    # plt.colorbar()
    
    
    
# plt.savefig(figurepath+"2")

# fig,ax=plt.subplots()
# ax.set_title("Non Optimized")
# QPC.U0=disorder
# QPC.set_all_pixels(common_voltages[14])
# t2,p2=QPC.plot_potential_section(bounds=bounds,ax=ax)
# plt.colorbar(p2)
# # np.where(t1==t2)
# plt.savefig(figurepath+"3")

# fig,ax=plt.subplots()
# ax.set_title("Non Optimized,no disorder")
# QPC.U0=0
# QPC.set_all_pixels(common_voltages[14])
# t3,p3=QPC.plot_potential_section(bounds=bounds,ax=ax)
# plt.colorbar(p3)
# plt.savefig(figurepath+"4")
# # np.where(t1==t2)

# fig,ax=plt.subplots()
# ax.set_title('Non Optimized - NonOpt No Disorder')
# p4=ax.imshow(t2.T-t3.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
# plt.colorbar(p4)
# plt.savefig(figurepath+"5")

# fig,ax=plt.subplots()
# ax.set_title('Optimized - NonOpt No Disorder')
# p5=ax.imshow(t1.T-t3.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
# plt.colorbar(p5)
# plt.savefig(figurepath+"6")


# fig,ax=plt.subplots()
# ax.set_title('Optimized - NonOpt')
# p6=ax.imshow(t1.T-t2.T,origin='lower',extent=(bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
# plt.colorbar(p6)
# plt.savefig(figurepath+"7")