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
- Different pptimization routines
    - Gradient Descent (Conjugate)
    - Gradient Free optimization through CMA-ES
    - ...
- Direct interfacing with the experiment using QCodes

Planned additions for the future:

- Generative modelling of the experiments to speed up optimization (through fast measurement evaluation)
- ...
