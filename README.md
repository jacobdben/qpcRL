## Code Workflow

Optimisation of simulated QPC is done by running the `optimise_kwant_device.py` script. It calls code from `helper_functions.py` to set up the device geometry (which again calls `simulations/kwant_sim.py` to make the KWANT object). From `lossfunctions/staircasiness.py` it calls the loss function to be used to judge how staicase-like the conductance is, and then lastly calls `optimization/cma-py` to perform the CMA algorithm loop. 

Note that simulations where performed on a cluster and may prove time-consuming on a regular computer.
