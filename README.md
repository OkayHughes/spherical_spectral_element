The purpose of this project is to create the a highly readable, well documented, well tested atmospheric dynamical core with support for variable resolution meshes.
This project prioritizes code readability and maintainability over performance. 
We want to minimize external dependencies and, insofar as it is possible, create a codebase that is almost entirely written in python.
Given the constraints of these design decisions, it is unlikely that the resulting dynamical core will scale to hundreds or thousands of 
nodes on an HPC computing system. This is acceptable to us.

An eventual central part of this project will be a tutorial that allows non-experts to understand (in broad terms) the design decisions
that must be made to accomodate the unique design requirements of atmospherical dynamical cores in the 21st century.
This project will provide one or two alternative choices that perform well in practice, and we will strive to 
demystify the tacit knowledge that leads dycore developers to converge on these one-or-two choices.

We are currently hopeful that aiming for simplicity and world-class documentation will make this a prime candidate
for porting to python auto-differentiation libraries, especially those that like Jax and Pytorch. 
