ITERATIVE TOMOGRAPHIC RECONSTRUCTION ALGORITHMS
===============================================



##  Brief description
This repository contains different iterative reconstruction algorithms for 
parallel beam tomography.

The following algorithms are included:

* the **Simultaneous Iterative Reconstruction Technique (SIRT)**;

* the **Maximum-Likelihood Expectation-Maximization (MLEM)**;

* the **Separable Paraboloidal Surrogate (SPS)**;

* the **Alternate Directions Method of Multipliers (ADMM)**.



##  Installation
Basic compilers like gcc and g++ and the FFTW library are required.
The simplest way to use the code is with an Anaconda environment equipped with
python-2.7, scipy, scikit-image and Cython.

Procedure:

1. Create the Anaconda environment (if not created yet): `conda create -n iter-rec python=2.7 anaconda`.

2. Install required Python packages: `conda install -n iter-rec scipy scikit-image Cython`.

3. Activate the environment: `source activate iter-rec`.

4. `git clone git@github.com:arcaduf/iterative_tomographic_reconstruction_algorithms.git`.
 
5. Install routines in C: `python setup.py`.

If `setup.py` runs without giving any error all subroutines in C have been installed and
your python version meets all dependencies.

If you run `python setup.py 1` (you can use any other character than 1), the 
all executables, temporary and build folders are deleted, the test data are 
placed in .zip files. In this way, the repository is restored to its original
status, right after the download.



##  Test the package
Go inside the folder "scripts/" and run: `python run_all_tests.py`

