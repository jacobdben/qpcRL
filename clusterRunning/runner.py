from Poisson_integration import *
# from scipy.optimize import minimize
import time
# import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def sweep_common_mode_and_measure(V2,tilt,qpc, common_voltages):
    #global global_results
    result=[]
    qpc.phi(0)
    qpc.U0(0)
    
    # Set the middle gate
    qpc.V2(V2)
    
    for i in range(len(common_voltages)):
        V1, V3 = common_voltages[i], common_voltages[i] + tilt
        qpc.V1(V1)
        qpc.V3(V3)
        result.append(qpc.transmission())
    #global_results.append(result)
    return np.array(result)


if __name__=="__main__":
    qpc = PoissonQPC(plot=False)
    common_mode_voltages = np.linspace(-5, -0.5, 30,dtype=float)
    import sys
    
    V2=float(sys.argv[1])
    tilt=float(sys.argv[2])
    print("V2: {}".format(V2))
    print("tilt: {}".format(tilt))
    
    start_time=time.perf_counter()
    global_results=[]
    result=sweep_common_mode_and_measure(V2,tilt,qpc,common_mode_voltages)
    stop_time=time.perf_counter()
    plt.plot(result)
    
    #saving all relevant data
    import os

    prefix="results/"
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    
    np.save(prefix+"V2_{}_tilt_{}".format(V2,tilt),result)

