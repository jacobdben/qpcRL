import matplotlib.pyplot as plt
import numpy as np

from datahandling.datahandling import datahandler, load_optimization_dict
from lossfunctions.staircasiness import staircasiness
from optimization.fourier.fourier_modes_hardcoded import fourier_to_potential

stairs=staircasiness()

#import the fourier optimization

fourier_dict=load_optimization_dict("fourier_3")
fourier_vals=[]
fourier_results=[]

fourier_new_vals=[]

for key in fourier_dict.keys():
    fourier_vals.append(fourier_dict[key]['val'])
    fourier_results.append(fourier_dict[key]['result'])   
    fourier_new_vals.append(stairs.deriv_metric_zeros1(fourier_results[-1]))


#import the normal optimization
normal_dict=load_optimization_dict("normal_1")
normal_vals=[]
normal_results=[]

for key in normal_dict.keys():
    normal_results.append(normal_dict[key]['result'])   

for result in normal_results:
    normal_vals.append(stairs.deriv_metric_zeros1(result))

plt.figure()
plt.plot(fourier_new_vals,label='fourier optimization')
# plt.plot(fourier_new_vals,label='fourier new vals')
plt.plot(normal_vals,label='normal optimization')
plt.legend()
plt.ylabel('loss',fontsize=18)
plt.xlabel('function call',fontsize=18)


normal_best_i=np.argmin(normal_vals)
fourier_best_i=np.argmin(fourier_vals)

normal_best_key=list(normal_dict.keys())[normal_best_i]
fourier_best_key=list(fourier_dict.keys())[fourier_best_i]


normal_best_x=normal_dict[normal_best_key]['x']
fourier_best_x=fourier_dict[fourier_best_key]['x']

normal_best_staircase=normal_dict[normal_best_key]['result']
fourier_best_staircase=fourier_dict[fourier_best_key]['result']

# fourier_alt_val=stairs.histogram(fourier_best_staircase)
# normal_alt_val=stairs.histogram(normal_best_staircase)

# fourier_rev_val=stairs.deriv_metric(fourier_best_staircase)
# normal_rev_val=stairs.deriv_metric(normal_best_staircase)

# print(fourier_vals[fourier_best_i])
# print(fourier_rev_val)
# print(fourier_alt_val)

# print(normal_vals[normal_best_i])
# print(normal_rev_val)
# print(normal_alt_val)



plt.figure()
plt.title('normal voltages')
plt.imshow(normal_best_x.reshape((3,3)),origin='lower')
plt.xticks([0,1,2],labels=['-1','0','1'])
plt.yticks([0,1,2],labels=['-1','0','1'])
plt.colorbar(label='V')

plt.figure()
plt.title('fourier voltages')
plt.xticks([0,1,2],labels=['-1','0','1'])
plt.yticks([0,1,2],labels=['-1','0','1'])
plt.imshow(fourier_to_potential(fourier_best_x)[1],origin='lower')
plt.colorbar(label='V')

plt.figure()
plt.grid()
plt.plot(np.linspace(-3,0,30),fourier_best_staircase,label='fourier staircase: {:.2f}'.format(fourier_vals[fourier_best_i]))
plt.plot(np.linspace(-3,0,30),normal_best_staircase,label='normal staircase: {:.2f}'.format(normal_vals[normal_best_i]))
plt.xlabel('avg voltage',fontsize=18)
plt.ylabel('conductance',fontsize=18)
plt.legend()

# AAAAA=fourier_to_potential(fourier_best_x)[1]