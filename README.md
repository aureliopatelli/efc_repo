# ECONOMIC FITNESS AND COMPLEXITY PACKAGE 
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
This class provides the routines to handle a collection of data with many snapshots, for example when the data is available on many years.


# TUTORIAL
The package provides a tutorial jupyter notebook in the *tutorial* folder


# USAGE
```python
from efc_repo import efc_matrix

# load the stored csv data in matrix format
data = efc_matrix.load_matrix('folder/stored_data.csv')

# get the Fitness
data.get_fitness(aspandas=True)

```


# EFC references

[1] Tacchella A., Cristelli M., Caldarelli G., Gabrielli A., Pietronero L., [A New Metrics for Countries' Fitness and Products' Complexity](https://doi.org/10.1038/srep00723), Sci Rep, 2, 723 (2012) 

[2] Hidalgo C., Hausmann R.,  [The building blocks of economic complexity](https://www.pnas.org/cg/doi/10.1073/pnas.0900943106), PNAS, 106 (26), 10570-10575, (2009)

[3] Tacchella A., Mazzilli D., Pietronero L. [A dynamical systems approach to gross domestic product forecasting](https://doi.org/10.1038/s41567-018-0204-y). Nature Phys, 14, 861–865 (2018)

[4] Mazzilli D., Mariani M.S., Morone F., Patelli A., [Equivalence between the Fitness-Complexity and the Sinkhorn-Knopp algorithms](https://iopscience.iop.org/article/10.1088/2632-072X/ad2697/meta), J. Phys. Complex., 5, 015010, (2024)

[5] Hidalgo C., Klinger B., Barabási A.L., Hausmann R., [The product space conditions the development of nations](https://www.science.org/doi/abs/10.1126/science.1144581), Science, 317 (5837), 482-487, (2007)

[6] Zaccaria A., Cristelli M., Tacchella A., Pietronero L, [How the Taxonomy of Products Drives the Economic Development of Countries](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0113770), PloS one, 9(12), e113770, (2014)
