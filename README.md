### Overview ###

This repository is associated with the manuscript "The physical chemistry of interphase loop extrusion" and houses notebooks demonstrating the lattice and polymer implementations of the five-state model of interphase cohesin.

Preprint available here: https://doi.org/10.1101/2024.08.23.609419

Code in this repository relies on:
- https://github.com/open2c/polychrom-hoomd/ for 3D polymer simulations implemented with [HooMD-Blue](https://github.com/glotzerlab/hoomd-blue)
- https://github.com/Fudenberg-Research-Group/discrete-time-extrusion for 1D lattice models of extrusion implemented in Python
- [SymPy](https://www.sympy.org/en/index.html) for fitting ODE models

To simulate genome conformations, we couple a 1D lattice model describing cohesin dynamics with a 3D polymer model describing the spatial position of chromatin:
![Model](./figures/five-state-polymer.png?raw=true)



This movie illustrates a typical trajectory of a single extruder for this model, sped up by a factor of 10 relative to real time.
The extruder is loaded into an active NIPBL-bound (blue) state, goes through multiple rounds of PDS5 exchange (green states), and finally unloads after WAPL binding (purple). Note that we stretch the chromatin on either side of this extruder for visualization purposes.

[!Watch the video](https://github.com/user-attachments/assets/2d13ef30-2c15-4322-bf2d-8f2c3dd2a510)






