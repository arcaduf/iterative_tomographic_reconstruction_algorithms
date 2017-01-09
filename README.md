###################################################################
###################################################################
####                                                           ####
####                   TOMOGRAPHIC PROJECTORS                  ####
####                                                           ####
###################################################################
###################################################################



##  Brief description
This repository contains different implementations in 2D of the X-ray transform or Radon
transform and its adjoint operator, the backprojector, for parallel beam geometry. 

More informations on the algorithm implementations of the X-ray transform are given in 
the ipython notebook file contained in this repository.



##  Installation
Basic compilers like gcc and g++ are required.
The simplest way to install all the code is to use Anaconda with python-2.7 and to 
add the installation of the python package scikit-image.

On a terminal, just type:
	1) conda create -n test-repo python=2.7 anaconda
	2) conda install -n test-repo Cython scikit-image
	3) source activate test-repo
	4) download the repo and type: python setup.py

If setup.py runs without giving any error all subroutines in C have been installed and
your python version meets all dependencies.



##  Test the package
Go inside the folder "scripts/" and run: python run_all_tests.py
Every time this script creates an image, the script is halted. To run the successive tests
just close the image.

