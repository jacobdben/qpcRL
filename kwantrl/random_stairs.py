import numpy as np
import matplotlib.pyplot as plt
from os import cpu_count
import sys
from lossfunctions.staircasiness import staircasiness
from concurrent.futures import ProcessPoolExecutor 
from optimization.cma import parallel_cma
from helper_functions import generate_polynomials, fourier_polynomials_to_voltages, initialise_device



common_mode=np.linspace(-1.75, 0, 100)
order = 1


def get_stairs(X):
    """
    X: An array of shape (8*n, ) containing the parameters to tune using cmaes. 
       This corresponds to 8 Fourier coefficients as n-order polynomials of the
       ninth Fourier coefficient a0 (pixel average).
    
    common_mode: array of Fourier coefficient to sweep
    
    qpca: KwantChip object
    
    order: polynomial order
    
    bounds: constraints on the pixel voltages
    
    loss_func: loss function 
    
    """
    
    print(X, flush=True)
    X = X.reshape(order, -1)
    
    polynomials = generate_polynomials(np.linspace(0,1,len(common_mode)), X) 

    voltages = np.array(fourier_polynomials_to_voltages(polynomials,vals=common_mode))

    
    Gs = []
    
    

    for i in range(len(common_mode)):
        
        qpca.set_gate('Vp1', voltages[i, 0])
        qpca.set_gate('Vp2', voltages[i, 1])
        qpca.set_gate('Vp3', voltages[i, 2])
        qpca.set_gate('Vp4', voltages[i, 3])
        qpca.set_gate('Vp5', voltages[i, 4])
        qpca.set_gate('Vp6', voltages[i, 5])
        qpca.set_gate('Vp7', voltages[i, 6])
        qpca.set_gate('Vp8', voltages[i, 7])
        qpca.set_gate('Vp9', voltages[i, 8])
        Gs.append(qpca.transmission())
    

    
    return np.array(Gs), voltages
    


disorder = None

if len(sys.argv)>1:
    disorder=(0.0005, int(sys.argv[1])) # Disorder size 50 is reference value.


print("CPUs available:", cpu_count(), flush=True)

qpca = initialise_device(L=500, W=300, dis=disorder)

configs = np.random.normal(0,0.5,size=8*10).reshape(10,8)

G = []
V = []

with ProcessPoolExecutor() as pool:
    G, V = zip(*pool.map(get_stairs, configs))

print(G)

np.save("random_stairs.npy", np.array([{'G':G, 'V':V}]))



