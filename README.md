# efc_repo
This repository provides the python class useful to handle and to evaluate the *Economic Fitness and Complexity* (EFC) *framework*. 

The repository is organized as follows:
- a class file collecting the two classes of efc_matrix and efc_matrix_dataset
- routine files
- a simple tutorial Jupyter notebook showing the use of the framework on a simple set of data


## efc_matrix class
This class provides the routines on a single snapshot, for example of the international trade export on a single year.

Main features are:
- it handles  dense (numpy, pandas) and sparse (scipy.sparse) matrix representations
- it evaluates the comparative advantages and its binarizations
- it evaluates the main economic complexity indices: Fitness, Diversification, ECI
- it evaluates simple projections and relatedness both validated and not validated



## efc_matrix_dataset class
