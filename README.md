## Optimizing experiments
This repository contains all of the code that we developed for automated optimization of experiments at QDev.
In particular, we have:

- Simulations of the (non-interacting) experiments using Kwant
    - A simple QPC with 3 gates
    - A cool QPC with a 3x3 pixel array of gates
    - ...
- Custom loss functions
    - 'Staircasiness' for QPC transmission plots
    - ...
- Different optimization routines
    - Gradient Descent (Conjugate)
    - Gradient Free optimization through CMA-ES
    - ...
- Direct interfacing with the experiment using QCodes

Planned additions for the future:

- Generative modelling of the experiments to speed up optimization (through fast measurement evaluation)
- ...

## For Jacob

In simulations several different scripts are available, the one named pixel_array contains the one we should be using. It defines a QPC class with two big outer gates and 3x3 pixel gates between the outer gates, it has functions for setting voltages on the different gates and for acquiring a measurement of the conductance through the simulated QPC. 

In lossfunctions.staircasiness all the lossfunctions we have tried are located, i would use staircasiness.window_loss as a starting point. This looks at the derivative of conductance within a predefined conductance window and scores it such that flat plateaus and steep ascents are favored.

The optimisation itself is done in optimisation.cma which contains optimise_cma and cma_p for parrallel optimisation. This should in combination with the datahandler from datahandling.datahandling also take care of saving a bunch of stuff regarding the optimisation runs, there you also need to define a location for the output of everything.

Lastly running an optimisation run is a matter of writing a function_to_minimize, you can decide on many different things when writing this function such as: which gates makes up the x-axis, if and how pixels are combined (e.g. through fourier modes or in rows or columns), the range of the x-axis or it can optimise over the x-axis aswell, it's only a matter of containing it in the function_to_minimize. 


