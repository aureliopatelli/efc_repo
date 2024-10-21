####################################################################################
############ TODO
## - check everything work with the scipy sparse module (needs -> NODF, projections)
## - implement the smoothing of the temporal series (at least moving averages)
## - projections (?)
##
####################################################################################

import numpy as np
import pandas as pd
import scipy, bicm, copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from . import fitness_complexity as fc
from . import nestedness as ned
from . import eci as eci_index
from . import projection as pj

from bicm.graph_classes import *
from bicm.network_functions import *


class efc_matrix:
    """
    this class implement the single matrix object that can be used in EFC.
    
    """
    
    def __init__(self, matrix, hardcopy=True, label_rows=None, label_columns=None):
        """
        :param matrix: raw/binary matrix to be considered in the analysis
        :type matrix: numpy.ndarray, scipy.sparse, pandas.DataFrame
        
        :param hardcopy: make an hard copy of the matrix (thus any transformation does not affect the original matrix)
        :type hardcopy: bool (True)
        
        :param label_rows: if present, the labels of the rows
        :type label_rows: list (None)

        :param label_columns: if present, the labels of the columns
        :type label_columns: list (None)
            
        """
        self.matrix = None
        self.label_rows = label_rows
        self.label_columns = label_columns
        self.fitness = None
        self.complexity = None
        self.ubiquity = None
        self.diversification = None
        self.density = None
        self.eci = None
        self.pci = None 
        self.nested_temperature = None
        self.nodf = None
        self.shape = None
        self._initialize_matrix(matrix, hardcopy)
    
    
    def _initialize_matrix(self, matrix, hardcopy): # ok with sparse
        """
        Initialize the class

        :param matrix: raw/binary matrix to be considered in the analysis
        :type matrix: numpy.ndarray, scipy.sparse, pandas.DataFrame

        :param hardcopy: make an hard copy of the matrix (thus any transformation does not affect the original matrix)
        :type hardcopy: bool (True)
        """
        if isinstance(matrix, np.ndarray) or isinstance(matrix, np.matrix):
            if hardcopy:
                self.matrix = matrix.copy()
            else:
                self.matrix = matrix

        elif scipy.sparse.isspmatrix(matrix):
            if hardcopy:
                self.matrix = matrix.copy()
            else:
                self.matrix = matrix
        
        elif isinstance(matrix, pd.DataFrame):
            self.label_rows = matrix.index
            self.label_columns = matrix.columns
            if hardcopy:
                self.matrix = matrix.to_numpy().copy()
            else:
                self.matrix = matrix.to_numpy()

        self.shape = np.array(matrix.shape)
    
    def print_matrix(self):
        """
        print information of the efc matrix
        """
        if scipy.sparse.issparse(self.matrix):
            print(self.matrix.info())
        else:
            print(self.matrix)
            
    def get_pvalue(self, inplace=False): # ok with sparse
        myGraph = BipartiteGraph()
        myGraph.set_biadjacency_matrix(np.array(matrix))
        myGraph.solve_tool(verbose=False, linsearch=True, print_error=False)
        pvalue = myGraph.get_weighted_pvals_mat()
        if inplace:
            del self.matrix
            self.matrix = pvalue
            return self
        else:
            return pvalue

    def get_binary(self, threshold=1, inplace=False): # ok with sparse
        """
        Get the binary efc matrix

        :param threshold: threshold of the efc matrix to get a binary matrix
        :type threshold: integer (1)

        :param inplace: transform the efc matrix to the binary version
        :type inplace: bool (False)

        :return: the class
        """
        bin_mat = self.matrix
        bin_mat[bin_mat < threshold] = 0
        bin_mat[bin_mat >= threshold] = 1
        if scipy.sparse.issparse(self.matrix):
            bin_mat.eliminate_zeros()
        if inplace:
            del self.matrix
            self.matrix = bin_mat
            return self
        return efc_matrix(bin_mat, label_rows=self.label_rows, label_columns=self.label_columns)
    
    def get_rca(self, inplace=False): # ok with sparse
        """
        Get the Revealed Comparative Advantage, aka the Balassa Index, of the efc matrix

        :param inplace: transform the efc matrix to the RCA version
        :type inplace: bool (False)

        :return: the class
        """
        if scipy.sparse.issparse(self.matrix):
            mat = self.matrix.copy()
            val = np.sqrt(mat.sum().sum())
            s0 = np.nan_to_num(val/mat.sum(0))
            s1 = np.nan_to_num(val/mat.sum(1))
            mat = mat.multiply(s0).transpose().multiply(s1).transpose()
            if inplace:
                del self.matrix
                self.matrix = mat.tocsr()
                return self
            return efc_matrix(mat.tocsr(), label_rows=self.label_rows, label_columns=self.label_columns)
        else:
            s0 = np.nan_to_num(self.matrix).sum(1)
            s1 = np.nan_to_num(self.matrix).sum(0)
            inv_average = np.dot(np.reshape(s0,(self.shape[0],1)), np.reshape(s1,(1,self.shape[1])))/s0.sum()
            inv_average = 1./inv_average
            inv_average[inv_average == np.inf] = 0
            if inplace:
                self.matrix = np.nan_to_num(self.matrix*inv_average)
                return self
            return efc_matrix(np.nan_to_num(self.matrix*inv_average), label_rows=self.label_rows, label_columns=self.label_columns)

    def get_binarize(self, method='rca', full_return=False, threshold=1): # ok with sparse
        """
        Get the binarization of the efc matrix (inplace) selecting the methodology

        :param method: indicates the methodology od binarization
        :type method: 'rca', 'biwcm', 'topological', 'significance', 'threshold' ('rca')

        :param full_return: is 'biwcm' is selected, the routing returns also the BipartiteGraph() class
        :type full_return: bool (False)

        :return: the class
        """
        if method == 'rca':
            self.get_rca(inplace=True)
            self.get_binary(1, inplace=True)
        elif method == 'topological':
            self.get_binary(1e-10, inplace=True)
        elif method == 'threshold':
            self.get_binary(threshold=threshold, inplace=True)
        elif method == 'significance':
            if scipy.sparse.issparse(self.matrix):
                bin_mat = self.matrix
                bin_mat.data = (bin_mat.data <= threshold).astype(int)
                bin_mat.eliminate_zeros()
                del self.matrix
                self.matrix = bin_mat
            else:
                self.matrix = 1. - self.matrix
                self.get_binary(self, threshold=1, inplace=False)
        elif method == 'mu biwcm' or method == 'biwcm':
            if scipy.sparse.issparse(self.matrix):
                matrix = self.matrix
            else:
                matrix = np.array(np.nan_to_num(self.matrix))
            myGraph = BipartiteGraph()
            myGraph.set_biadjacency_matrix(matrix)
            myGraph.solve_tool(linsearch=True, verbose=False, print_error=False, model='biwcm_c')
            avg_mat = 1./myGraph.avg_mat
            avg_mat[avg_mat == np.inf] = 0
            self.matrix = np.nan_to_num(matrix * avg_mat)
            self.get_binary(1, inplace=True)
            if full_return:
                return self, myGraph
        return self

    def get_distribution_values(self, dx=0.001, maxx=1000, minx=0, log=False, scale=0, N=np.nan): # ok with sparse
        if scipy.sparse.issparse(self.matrix):
            data = self.matrix.data
        else:
            data = self.matrix.data
        if log:
            if N == np.nan:
                N = int( ((maxx-minx)/dx ))
            if scale>0:
                y = np.linspace(np.log10( (minx+scale*dx)), np.log10(maxx+scale*dx), num=N, endpoint=True)
                space = np.power(10.0, y ).astype(np.double)-scale*dx
            else:
                space = np.geomspace(minx, maxx, num=N, endpoint=True)
            y, x = np.histogram(data, bins=space)
            ratio = dx/np.diff(space)
            space = ((space + np.roll(space,-1))/2)[:-1]
            return pd.DataFrame(y*ratio,index=space)
        else:
            if N != np.nan:
                dx = (maxx-minx)/N
            y, x = np.histogram(data , bins=np.arange(minx,maxx,dx))
            return pd.DataFrame(y,index=x[:-1]+dx*0.5)

    def _eci(self, ztransform=True): # ok with sparse
        if scipy.sparse.issparse(self.matrix):
            self.eci = eci_index.eci_sparse(self.matrix, ztransform=ztransform)
        else:
            self.eci = eci_index.eci_dense(self.matrix, ztransform=ztransform)
            
    def _pci(self, ztransform=True): # ok with sparse
        if scipy.sparse.issparse(self.matrix):
            self.pci = eci_index.pci_sparse(self.matrix, ztransform=ztransform)
        else:
            self.pci = eci_index.pci_dense(self.matrix, ztransform=ztransform)
    
    def _NODF(self, removezero=False):
        if scipy.sparse.issparse(self.matrix):
            pass
        else:
            self.nodf = ned.NODF(self.matrix, removezero=removezero)

    def _fitness_complexity(self, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, consider_dummy=False): # ok with sparse
        fit, com = fc.fitness_complexity(self.matrix.copy(), max_iteration = max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, consider_dummy=consider_dummy)
        self.fitness = fit.to_numpy()
        self.complexity = com.to_numpy()
        
    def _fitness_complexity_servedio(self, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, delta=1.0): # ok with sparse
        fit, com = fc.fitness_complexity_servedio(self.matrix.copy(), max_iteration = max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, delta=delta)
        self.fitness = fit.to_numpy()
        self.complexity = com.to_numpy()

    def _fitness_complexity_mazzolini(self, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, gamma=gamma): # ok with sparse
        fit, com = fc.fitness_complexity_mazzolini(self.matrix.copy(), max_iteration = max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, gamma=gamma)
        self.fitness = fit.to_numpy()
        self.complexity = com.to_numpy()

    def _density(self): # ok with sparse
        self.density = np.sum(self.matrix) / (self.shape[0]*self.shape[1])

    def _diversification_ubiquity(self): # ok with sparse
        self.diversification = np.nan_to_num(self.matrix).sum(1)
        self.ubiquity  = np.nan_to_num(self.matrix).sum(0)

    def remove_low_degrees(self, threshold=0, inplace=False):
        """
        This routine returns a efc_matrix element where the matrix has cutted all the rows and colums with a degree originally below a given threshold
        :param threshold:
        :type threshold: float
        :return:
        """
        if inplace:
            mat = self
        else:
            mat = self.copy()

        wherer = mat.matrix.sum(1) > threshold
        wherec = mat.matrix.sum(0) > threshold

        if hasattr(mat.matrix, 'getformat'):
            mat.matrix[wherer,:] = np.array([0 for j in range(len(wherer))])
            mat.matrix[:, wherec] = np.array([0 for j in range(len(wherec))])
            mat.matrix.eliminate_zeros()

        elif isinstance(mat.matrix, np.ndarray):
            mat.matrix = mat.matrix[wherer]
            mat.matrix = mat.matrix[:, wherec]

        mat.label_rows = mat.label_rows[wherer]
        mat.label_columns = mat.label_columns[wherec]

        if mat.fitness is not None:
            mat.fitness = mat.fitness[wherer]
        if mat.eci is not None:
            mat.eci = mat.eci[wherer]
        if mat.diversification is not None:
            mat.diversification = mat.diversification[wherer]
        if mat.complexity is not None:
            mat.complexity = mat.complexity[wherec]
        if mat.pci is not None:
            mat.pci = mat.pci[wherec]
        if mat.ubiquity is not None:
            mat.ubiquity = mat.ubiquity[wherec]

        mat.shape = np.array(mat.matrix.shape)

        return mat

        
    def set_label_rows(self, label): # ok with sparse
        """
        Associates the labels of the rows

        :param label: if not None, it collects the labels of the rows
        :type label: list
        """
        if self.label_rows is None and len(label) == self.shape[0]:
            self.label_rows = label
        elif len(label) != self.shape[0]:
            print('WARNING: the size of the label {} is different w.r.t. the size of the matrix {}'.format(len(label), self.shape[0]))
            
    def set_label_columns(self, label): # ok with sparse
        """
        Associates the labels of the columns

        :param label: if not None, it collects the labels of the columns
        :type label: list
        """
        if self.label_columns is None and len(label) == self.shape[1]:
            self.label_columns = label
        elif len(label) != self.shape[1]:
            print('WARNING: the size of the label {} is different w.r.t. the size of the matrix {}'.format(len(label), self.shape[1]))
                
    def get_matrix(self, aspandas=False): # ok with sparse
        """
        Returns the efc matrix
        :param aspandas: True if the desired output is a pandas DataFrame
        :type aspandas: bool (False)
        :return: the numpy.ndarray of the pandas.DataFrame of the efc matrix
        """
        if aspandas:
            if self.label_rows is None:
                self.label_rows = range(self.shape[0])
            if self.label_columns is None:
                self.label_columns = range(self.shape[1])
            if scipy.sparse.issparse(self.matrix):
                return pd.DataFrame(self.matrix.todense(), index=self.label_rows, columns=self.label_columns)
            return pd.DataFrame(self.matrix, index=self.label_rows, columns=self.label_columns)
        return self.matrix

    def get_density(self): # ok with sparse
        """
        Returns the density of the efc matrix
        :return: float
        """
        if self.density is None:
            self._density()
        return self.density

    def get_diversification(self, aspandas=False): # ok with sparse
        """
        Returns the diversification vector (row degree)
        :param aspandas: True if the desired output is a pandas DataFrame
        :type aspandas: bool (False)
        :return: the numpy.ndarray of the pandas.DataFrame of the diversification
        """
        if self.diversification is None:
            self._diversification_ubiquity()
        if aspandas:
            if self.label_rows is None:
                self.label_rows = range(self.shape[0])
            return pd.DataFrame(self.diversification, index=self.label_rows, columns=['diversification'])
        return self.diversification

    def get_ubiquity(self, aspandas=False): # ok with sparse
        """
        Returns the ubiquity vector (column degree)
        :param aspandas: True if the desired output is a pandas DataFrame
        :type aspandas: bool (False)
        :return: the numpy.ndarray of the pandas.DataFrame of the ubiquity
        """
        if self.ubiquity is None:
            self._diversification_ubiquity()
        if aspandas:
            if self.label_columns is None:
                self.label_columns = range(self.shape[1])
            return pd.DataFrame(self.ubiquity, index=self.label_columns, columns=['ubiquity'])
        return self.ubiquity

    def get_fitness(self, aspandas=False, force=False, method=None, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, consider_dummy=False, delta=1.0, gamma=-1.0): # ok with sparse
        """
        Return the Fitness vector
        :param aspandas: True if the desired output is a pandas DataFrame
        :param force: force the computation, otherwise use the stored result
        :param method: decide if the algorithm is Tacchella-2012 or Servedio-2018
        :type method: 'servedio', 'mazzolini', None (standard method)
        :param max_iteration: maximum number of iterations
        :param check_stop: set the stopping condition
        :param min_distance: L1 distance of successive iterations below which the loop end
        :param normalization: normalization condition
        :param fit_ic: vector of the initial condition of the fitness
        :param com_ic: vector of the initial condition of the complexity
        :param removelowdegrees: remove from the matrices columns with degrees below the specified one
        :param verbose: being verbose
        :param consider_dummy: consider the algorithm as if the dummy row is present
        :param delta: parameter of the Servedio algorithm
        :return: numpy.ndarray or pandas.DataFrame of the Fitness
        """
        if self.fitness is None or force:
            if method=='servedio':
                self._fitness_complexity_servedio(max_iteration=max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, delta=delta)
            elif method=='mazzolini':
                self._fitness_complexity_mazzolini(max_iteration=max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, gamma=gamma)
            else:
                self._fitness_complexity(max_iteration=max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, consider_dummy=consider_dummy)
        if aspandas:
            if self.label_rows is None:
                self.label_rows = range(self.shape[0])
            return pd.DataFrame(self.fitness, index=self.label_rows, columns=['fitness'])
        return self.fitness
    
    def get_complexity(self, aspandas=False, force=False, method=None, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, consider_dummy=False, delta=1.0, gamma=-1.0): # ok with sparse
        """
        Return the Complexity vector
        :param aspandas: True if the desired output is a pandas DataFrame
        :param force: force the computation, otherwise use the stored result
        :param method: decide if the algorithm is Tacchella-2012 or Servedio-2018
        :type method: 'servedio', None (standard method)
        :param max_iteration: maximum number of iterations
        :param check_stop: set the stopping condition
        :param min_distance: L1 distance of successive iterations below which the loop end
        :param normalization: normalization condition
        :param fit_ic: vector of the initial condition of the fitness
        :param com_ic: vector of the initial condition of the complexity
        :param removelowdegrees: remove from the matrices columns with degrees below the specified one
        :param verbose: being verbose
        :param consider_dummy: consider the algorithm as if the dummy row is present
        :param delta: parameter of the Servedio algorithm
        :return: numpy.ndarray or pandas.DataFrame of the Complexity
        """
        if self.complexity is None or force:
            if method=='servedio':
                self._fitness_complexity_servedio(max_iteration=max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, delta=delta)
            elif method=='mazzolini':
                self._fitness_complexity_mazzolini(max_iteration=max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, gamma=gamma)
            else:
                self._fitness_complexity(max_iteration=max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, consider_dummy=consider_dummy)
        if aspandas:
            if self.label_columns is None:
                self.label_columns = range(self.shape[1])
            return pd.DataFrame(self.complexity, index=self.label_columns, columns=['complexity'])
        return self.complexity

    def get_nodf(self): # ok with sparse
        """
        Return the NODF value
        :return: float of NODF
        """
        if self.nodf is None:
            self._NODF()
        return self.nodf
    
    def get_eci(self, aspandas=False, force=False): # ok with sparse
        """
        Return the ECI vector
        :param aspandas: True if the desired output is a pandas DataFrame
        :param force: force the computation, otherwise use the stored result
        :return: numpy.ndarray or pandas.DataFrame of ECI
        """
        if self.eci is None or force:
            self._eci()
        if aspandas:
            if self.label_rows is None:
                self.label_rows = range(self.shape[0])
            return pd.DataFrame(self.eci, index=self.label_rows, columns=['eci'])
        return self.eci
    
    def get_pci(self, aspandas=False, force=False): # ok with sparse
        """
        Return the PCI vector
        :param aspandas: True if the desired output is a pandas DataFrame
        :param force: force the computation, otherwise use the stored result
        :return: numpy.ndarray or pandas.DataFrame of PCI
        """
        if self.pci is None or force:
            self._pci()
        if aspandas:
            if self.label_columns is None:
                self.label_columns = range(self.shape[1])
            return pd.DataFrame(self.pci, index=self.label_columns, columns=['pci'])
        return self.pci
    
    def copy(self): # ok with sparse
        """
        Copy routine
        :return: return the hard copy of the efc class
        """
        return copy.deepcopy(self)

    def add_dummy(self, dummy_row=True, dummy_col=False, inplace=False):
        """
        Create a new class where the efc matrix has a new row and/or column with all entries 1
        :param dummy_row: If True a dummy row is added
        :param dummy_col: If True a dummy column is added
        :param inplace: If True the efc matrix is changed
        :return: the efc class with the added dummy
        """
        if inplace:
            if self.label_rows is None:
                self.label_rows = range(self.shape[0])
            if self.label_columns is None:
                self.label_columns = range(self.shape[1])
            mat_dummy = pd.DataFrame(self.matrix, index=self.label_rows, columns=self.label_columns)
            if dummy_row:
                mat_dummy.loc['dummy row'] = np.ones(mat_dummy.shape[1]).astype(int)
            if dummy_col:
                mat_dummy['dummy column'] = np.ones(mat_dummy.shape[0]).astype(int)
            self.matrix = mat_dummy.to_numpy()
            self.label_rows = mat_dummy.index
            self.label_columns = mat_dummy.columns
            self.fitness = None
            self.complexity = None
            self.ubiquity = None
            self.diversification = None
            self.density = None
            self.eci = None
            self.pci = None 
            self.nested_temperature = None
            self.nodf = None
            self.shape = np.array(mat_dummy.shape)
            return self
        else:
            copymat = self.copy()
            return copymat.add_dummy(dummy_row=dummy_row, dummy_col=dummy_col, inplace=True)
    
    def get_validated_cooccurrence(self, rows=True, alpha=0.05, method='poisson'):
        """
        Evaluate the co-occurrences validated through the BiCM method
        :param rows: select the layer of projection
        :param alpha: statistical significance (es. 0.5 correspond to a 50% threshold)
        :param method: approximated method of evaluation of the significance
        :return: list of elements in the validated projection
        """
        matrix = np.array(np.nan_to_num(self.matrix))
        myGraph = BipartiteGraph()
        myGraph.set_biadjacency_matrix(matrix)
        myGraph.compute_projection(rows=rows, alpha=alpha, method=method, threads_num=4, progress_bar=False)
        if rows:
            label = self.label_rows
            proj = myGraph.projected_rows_adj_list
            list_proj = []
            for key in proj.keys():
                list_proj += [(label[key],label[i]) for i in proj[key]]
            return list_proj
        else:
            label = self.label_columns
            proj = myGraph.projected_cols_adj_list
            list_proj = []
            for key in proj.keys():
                list_proj += [(label[key],label[i]) for i in proj[key]]
            return list_proj
        
    def get_projection(self, method='cooccurrence', rows=False, top=2, alpha=None, verbose=False, aspandas=False): # add product space
        """
        Routine that compute the projection of the matrix using the main formulas
        :param method: select the method of projection
        :param rows: select the layer of projection
        :param top: number of top edges in the projection values if no validation is considered
        :param alpha: statistical significance (es. 0.5 correspond to a 50% threshold)
        :param verbose: being verbose
        :param aspandas: return a pandas.DataFrame
        :return: numpy.ndarray with the projected matrix (validated or not)
        """
        proj = None

        if alpha is None:
            if method == 'cooccurrence':
                proj = pj.occurrence_matrix(self.matrix, row_proj=rows)

            elif method == 'taxonomy':
                tax = pj.taxonomy_matrix(self.matrix, row_proj=rows)
                proj = pj.net_top(tax, top)

            elif method == 'assist matrix':
                proj = pj.assist_matrix(self.matrix, self.matrix, row_proj=rows)

            elif method == 'product space':
                proj = pj.proximity_matrix(self.matrix, row_proj=rows)

            else:
                print('not found {}', method)

        else:
            myGraph = BipartiteGraph()
            myGraph.set_biadjacency_matrix(self.matrix)
            myGraph.solve_tool(verbose=False, linsearch=True, print_error=False)
            pval = myGraph.get_bicm_matrix()
            size_random_ensemble = int(5./alpha)
            random_matrices = {}
            for r in range(size_random_ensemble):
                random_matrices[r] = pj.random_binary_matrix(pval)
                
            if verbose:
                print('method {}, number random {}'.format(method,size_random_ensemble))
                
            if method == 'cooccurrence':
                proj = pj.occurrence_matrix(self.matrix, row_proj=rows)
                adj = np.zeros(proj.shape)
                for r in range(size_random_ensemble):
                    random_proj = pj.occurrence_matrix(random_matrices[r], row_proj=rows)
                    adj += (random_proj > proj).astype(int)
                adj /= size_random_ensemble
                proj = (adj<=alpha).astype(int)
            
            elif method == 'taxonomy':
                proj = pj.taxonomy_matrix(self.matrix, row_proj=rows)
                adj = np.zeros(proj.shape)
                for r in range(size_random_ensemble):
                    random_proj = pj.taxonomy_matrix(random_matrices[r], row_proj=rows)
                    adj += (random_proj > proj).astype(int)
                adj /= size_random_ensemble
                proj = (adj<=alpha).astype(int)
                
            elif method == 'assist matrix':
                proj = pj.assist_matrix(self.matrix, self.matrix, row_proj=rows)
                adj = np.zeros(proj.shape)
                for r in range(size_random_ensemble):
                    random_proj = pj.assist_matrix(random_matrices[r], random_matrices[r], row_proj=rows)
                    adj += (random_proj > proj).astype(int)
                adj /= size_random_ensemble
                proj = (adj<=alpha).astype(int)

            elif method == 'product space':
                proj = pj.proximity_matrix(self.matrix, row_proj=rows)
                adj = np.zeros(proj.shape)
                for r in range(size_random_ensemble):
                    random_proj = pj.proximity_matrix(random_matrices[r], row_proj=rows)
                    adj += (random_proj > proj).astype(int)
                adj /= size_random_ensemble
                proj = (adj<=alpha).astype(int)

            else:
                print('not found {}', method)

        if aspandas:
            if rows:
                return pd.DataFrame(proj, index=self.label_columns, columns=self.label_columns)
            else:
                return pd.DataFrame(proj, index=self.label_rows, columns=self.label_rows)
        return proj

    def plot_matrix(self, index='fitness', cmap='Blues', label_rows='Countries', label_columns='Products', fontsize=20):
        """
        Plot the matrix ordered using
        :param index:
        :param cmap:
        :param label_rows:
        :param label_columns:
        :param fontsize:
        :return:
        """
        if index == 'eci':
            set0 = self.get_eci()
            set1 = self.get_pci()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        if index == 'degree':
            set0 = self.get_diversification()
            set1 = self.get_ubiquity()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        else:
            set0 = self.get_fitness()
            set1 = self.get_complexity()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::]
        matrix = self.matrix[set0][:,set1]
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.matshow(matrix, aspect='auto', cmap=cmap)
        if index == 'eci':
            ax.set_xlabel('{} (ordered by increasing PCI)'.format(label_columns), fontsize=fontsize, color='black', loc='center')
            ax.set_ylabel('{} (ordered by decreasing ECI)'.format(label_rows), fontsize=fontsize, color='black', loc='center')
        if index == 'degree':
            ax.set_xlabel('{} (ordered by increasing degree)'.format(label_columns), fontsize=fontsize, color='black', loc='center')
            ax.set_ylabel('{} (ordered by decreasing degree)'.format(label_rows), fontsize=fontsize, color='black', loc='center')
        else:
            ax.set_xlabel('{} (ordered by increasing Q)'.format(label_columns), fontsize=fontsize, color='black', loc='center')
            ax.set_ylabel('{} (ordered by decreasing F)'.format(label_rows), fontsize=fontsize, color='black', loc='center')
            
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([0, self.shape[1]-0.5])
        if index == 'eci' or index == 'degree':
            ax.set_xticklabels(['{}°'.format(self.shape[1]), '1°'], minor=False, fontdict={'fontsize': fontsize})
        else:
            ax.set_xticklabels(['{}°'.format(self.shape[1]), '1°'], minor=False, fontdict={'fontsize': fontsize})
        ax.set_xlim(-0.5, self.shape[1]-0.5)
        ax.set_yticks([0,self.shape[0]-1])
        ax.set_yticklabels(['1°', '{}°'.format(self.shape[0])], minor=False, fontdict={'fontsize': fontsize})
        ax.set_ylim(self.shape[0]-0.5, -0.5)

        return fig, ax
    
    def plot_large_matrix(self, index='fitness', cmap='Blues', label_rows='Actors', label_columns='Activities', fontsize=20):
        """
        Plot the matrix ordered using a scatter plot
        :param index:
        :param cmap:
        :param label_rows:
        :param label_columns:
        :param fontsize:
        :return:
        """        
        if index == 'eci':
            set0 = self.get_eci()
            set1 = self.get_pci()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        if index == 'degree':
            set0 = self.get_diversification()
            set1 = self.get_ubiquity()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        else:
            set0 = self.get_fitness()
            set1 = self.get_complexity()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::]
            
        matrix = self.matrix.copy()
        if hasattr(matrix, 'getformat'):
            matrix = matrix.tocoo()
        else:
            matrix = scipy.sparse.coo_matrix(matrix)
        
        set1pos = pd.Series(np.arange(self.shape[0]), index=set0).sort_index().to_numpy()[matrix.row]
        set2pos = pd.Series(np.arange(self.shape[1]), index=set1).sort_index().to_numpy()[matrix.col]
                
        fig, ax = plt.subplots(figsize=(40,40*set0.shape[0]/set1.shape[0]))
        if index == 'eci':
            ax.set_xlabel('{} (ordered by increasing PCI)'.format(label_columns), fontsize=fontsize, color='black', loc='center')
            ax.set_ylabel('{} (ordered by decreasing ECI)'.format(label_rows), fontsize=fontsize, color='black', loc='center')
        if index == 'degree':
            ax.set_xlabel('{} (ordered by increasing degree)'.format(label_columns), fontsize=fontsize, color='black', loc='center')
            ax.set_ylabel('{} (ordered by decreasing degree)'.format(label_rows), fontsize=fontsize, color='black', loc='center')
        else:
            ax.set_xlabel('{} (ordered by increasing Q)'.format(label_columns), fontsize=fontsize, color='black', loc='center')
            ax.set_ylabel('{} (ordered by decreasing F)'.format(label_rows), fontsize=fontsize, color='black', loc='center')
            
            
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([0, self.shape[1]-0.5])
        if index == 'eci' or index == 'degree':
            ax.set_xticklabels(['{}°'.format(self.shape[1]), '1°'], minor=False, fontdict={'fontsize': fontsize})
        else:
            ax.set_xticklabels(['{}°'.format(self.shape[1]), '1°'], minor=False, fontdict={'fontsize': fontsize})
        ax.set_xlim(-0.5, self.shape[1]-0.5)
        ax.set_yticks([0,self.shape[0]-1])
        ax.set_yticklabels(['1°', '{}°'.format(self.shape[0])], minor=False, fontdict={'fontsize': fontsize})
        ax.set_ylim(self.shape[0]-0.5, -0.5)
        ax.scatter(set2pos, set1pos, s=0.1, color='dimgray')
        
        return fig, ax
    
    def get_masked_from_indices(self, list_rows=None, list_columns=None):
        if list_rows is None and list_columns is None:
            return self
        if list_rows is not None:
            absent_items = 0
            for e in list_rows:
                if e not in self.label_rows:
                    absent_items += 1
            if absent_items:
                print('ERROR: there are elements of list_rows not present in the matrix row labels')
                return None
            mask_rows = np.isin(self.label_rows,list_rows)
        else:
            list_rows = self.label_rows
            mask_rows = np.ones((self.shape[0])).astype(bool)
            
        if list_columns is not None:
            absent_items = 0
            for e in list_columns:
                if e not in self.label_rows:
                    absent_items += 1
            if absent_items:
                print('ERROR: there are elements of list_columns not present in the matrix column labels')
                return None
            mask_columns = np.isin(self.label_columns,list_columns)
        else:
            list_columns = self.label_columns
            mask_columns = np.ones((self.shape[1])).astype(bool)
            
        
        if scipy.sparse.issparse(self.matrix):
            matrix = self.matrix[mask_rows,mask_columns]
            return efc_matrix(matrix, label_rows=self.label_rows[mask_rows], label_columns=self.label_columns[mask_columns])
        else:
            matrix = self.matrix[mask_rows,:][:,mask_columns]
            return efc_matrix(matrix, label_rows=self.label_rows[mask_rows], label_columns=self.label_columns[mask_columns])
        return None
    
    
    def get_energy_matrix(self, aspandas=None):
        matrix = self.matrix
        fit = self.get_fitness().copy()
        np.divide(np.ones(self.shape[0]), fit, out=fit, where=fit>0)
        com = self.get_complexity()
        outer = np.outer(fit, com)
        matrix = outer*matrix
        matrix /= max(matrix.sum(1))
        if aspandas:
            if self.label_rows is None:
                self.label_rows = range(self.shape[0])
            if self.label_columns is None:
                self.label_columns = range(self.shape[1])
            return pd.DataFrame(matrix, index=self.label_rows, columns=self.label_columns)
        return matrix



    @classmethod
    def load_matrix(cls, path):
        supp = None
        if path[-4:] == '.csv':
            supp = cls(pd.read_csv(path, index_col=0))
        elif path[-4:] == '.pkl' or path[-3:] == '.pk':
            supp = cls(pd.read_pickle(path))
        elif path[-4:] == '.npz':
            supp = cls(scipy.sparse.load_npz(path))
        else:
            print('ERROR loading: ',path)

        return supp
    
    def get_exogenous_fitness(cls, complexity, aspandas=False):
        if len(complexity) != cls.shape[1]:
            print('ERROR: dimensions are different {} != {}'.format(len(complexity), cls.shape[1]))
            return None
        efitness = np.dot(cls.matrix,complexity).flatten()
        efitness /= np.max(efitness)
        if aspandas:
            if cls.label_rows is None:
                return pd.DataFrame(efitness, index=range(cls.shape[0]), columns=['exogenous fitness'])
            return pd.DataFrame(efitness, index=cls.label_rows, columns=['exogenous fitness'])
        return efitness
    

    
    
    
class efc_matrix_dataset:
    """
    definition of a compilation of efc matrices, assumed uniform (same dimensions)
    """
    
    def __init__(self, matrices={}, label_rows=None, label_columns=None):
        self.rangeyear = None
        self.matrices = None
        self.label_rows = label_rows
        self.label_columns = label_columns
        self.fitness = None
        self.complexity = None
        self.ubiquity = None
        self.diversification = None
        self.density = None
        self.eci = None
        self.pci = None
        self.leave_tqdm = False
        
        rangeyear = list(matrices.keys())
        self.rangeyear = rangeyear
        self.matrices = {}
        for year in rangeyear:
            self.matrices[year] = efc_matrix(matrices[year])
            
        # here we assume the size of the matrices is uniform
        self.label_rows = self.matrices[rangeyear[0]].label_rows
        self.label_columns = self.matrices[rangeyear[0]].label_columns

        dim = self.matrices[rangeyear[0]].shape
        self.shape = np.array((dim[0],dim[1],len(rangeyear)))
        
    def set_label_rows(self, label, force=False):
        if self.label_rows is None and len(label) == self.shape[0]:
            self.label_rows = label
        elif force and len(label) == self.shape[0]:
            self.label_rows = label
#        else:
#            print('Warning: there are some problem with the size: {} -> {}'.format(len(label), self.shape[0]))
            
    def set_label_columns(self, label, force=False):
        if self.label_columns is None and len(label) == self.shape[1]:
            self.label_columns = label
        elif force and len(label) == self.shape[1]:
            self.label_columns = label
#        else:
#            print('Warning: there are some problem with the size: {} -> {}'.format(len(label), self.shape[1]))
            
    def add_dummy(self, dummy_row=True, dummy_col=False, inplace=False):
        if inplace:
            if dummy_row:
                self.shape[0] += 1
            if dummy_col:
                self.shape[1] += 1
            for year in self.rangeyear:
                self.matrices[year].add_dummy(dummy_row=dummy_row, dummy_col=dummy_col, inplace=True)
            self.label_rows = self.matrices[self.rangeyear[0]].label_rows
            self.label_columns = self.matrices[self.rangeyear[0]].label_columns
            return self
        else:
            supp = self.copy()            
            return supp.add_dummy(dummy_row=dummy_row, dummy_col=dummy_col, inplace=True)
        
    def get_fitness(self, aspandas=False, force=False, method=None, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, consider_dummy=False, delta=1.0):
        self.fitness = np.zeros((self.shape[0],self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            fit = self.matrices[year].get_fitness(aspandas=False, force=force, method=method, max_iteration = max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, consider_dummy=consider_dummy, delta=delta)
            self.fitness[:,i] = fit[:]
        if self.label_rows is not None:
            if 'dummy row' in self.label_rows:
                pos_dumy = np.where(self.label_rows == 'dummy row')[0]
                self.fitness[:,:] /= self.fitness[pos_dumy,:]
            elif 'dummy col' in self.label_columns:
                self.fitness /= np.sum(self.fitness, axis=0)
        if aspandas:
            self.set_label_rows(range(self.shape[0]))
            return pd.DataFrame(self.fitness, index=self.label_rows, columns=self.rangeyear)
        return self.fitness

    def get_complexity(self, aspandas=False, force=False, method=None, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, consider_dummy=False, delta=1.0):
        self.complexity = np.zeros((self.shape[1],self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            com = self.matrices[year].get_complexity(aspandas=False, force=force, method=method, max_iteration = max_iteration, check_stop=check_stop, min_distance=min_distance, normalization=normalization, fit_ic=fit_ic, com_ic=com_ic, removelowdegrees=removelowdegrees, verbose=verbose, consider_dummy=consider_dummy, delta=delta)
            self.complexity[:,i] = com[:]
        if 'dummy col' in self.label_columns:
            pos_dumy = np.where(self.label_columns == 'dummy col')[0]
            self.complexity[:,:] /= self.complexity[pos_dumy,:]
        elif 'dummy row' in self.label_rows:
            self.complexity /= np.sum(self.complexity, axis=0)
        if aspandas:
            self.set_label_columns(range(self.shape[1]))
            return pd.DataFrame(self.complexity, index=self.label_columns, columns=self.rangeyear)
        return self.complexity

    def get_diversification(self, aspandas=False):
        self.diversification = np.zeros((self.shape[0],self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            div = self.matrices[year].get_diversification(aspandas=False)
            self.diversification[:,i] = div[:]
        if aspandas:
            self.set_label_rows(range(self.shape[0]))
            return pd.DataFrame(self.diversification, index=self.label_rows, columns=self.rangeyear)
        return self.diversification
    
    def get_ubiquity(self, aspandas=False):
        self.ubiquity = np.zeros((self.shape[1],self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            ubi = self.matrices[year].get_ubiquity(aspandas=False)
            self.ubiquity[:,i] = ubi[:]
        if aspandas:
            self.set_label_columns(range(self.shape[1]))
            return pd.DataFrame(self.ubiquity, index=self.label_columns, columns=self.rangeyear)
        return self.ubiquity
    
    def get_eci(self, aspandas=False):
        self.eci = np.zeros((self.shape[0],self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            div = self.matrices[year].get_eci(aspandas=False)
            self.eci[:,i] = div[:]
        if aspandas:
            self.set_label_rows(range(self.shape[0]))
            return pd.DataFrame(self.eci, index=self.label_rows, columns=self.rangeyear)
        return self.eci

    def get_pci(self, aspandas=False):
        self.pci = np.zeros((self.shape[1],self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            ubi = self.matrices[year].get_pci(aspandas=False)
            self.pci[:,i] = ubi[:]
        if aspandas:
            self.set_label_columns(range(self.shape[1]))
            return pd.DataFrame(self.pci, index=self.label_columns, columns=self.rangeyear)
        return self.pci

    def get_density(self, aspandas=False):
        self.density = np.zeros((self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            den = self.matrices[year].get_density()
            self.density[i] = den
        if aspandas:
            self.set_label_columns(range(self.shape[1]))
            return pd.DataFrame(self.density, index=self.rangeyear, columns=['density'])
        return self.density

    def get_nodf(self, aspandas=False):
        nodf = np.zeros((self.shape[2]))
        for i in tqdm(range(len(self.rangeyear)), leave=self.leave_tqdm):
            year = self.rangeyear[i]
            nod = self.matrices[year].get_nodf()
            nodf[i] = nod
        if aspandas:
            self.set_label_columns(range(self.shape[1]))
            return pd.DataFrame(nodf, index=self.rangeyear, columns=['nodf'])
        return nodf

    def copy(self):
        return copy.deepcopy(self)
    
    def get_binarize(self, method='rca', threshold=1):
        for year in tqdm(self.rangeyear, leave=self.leave_tqdm):
            self.matrices[year].get_binarize(method=method, full_return=False, threshold=threshold)
        return self
    
    def get_projection(self, method='cooccurrence', average=False, dt=5, rows=False, top=2, alpha=0.05, verbose=False):
        if average:
            if rows:
                proj = np.zeros((self.shape[0],self.shape[0]))
            else:
                proj = np.zeros((self.shape[1],self.shape[1]))
            
            size = self.shape[2]
            if dt==0:
                for year in self.rangeyear:
                    proj += self.matrices[year].get_projection(method=method, rows=rows, top=top, alpha=alpha, verbose=verbose)
            
            else:
                size = 0
                for dt_ in (range(dt)+1):
                    size += len(range(dt)+1)
                                        
            proj[proj < size] = 0
            proj[proj == size] = 1    
            return proj
            
        else:

            if rows:
                proj = np.zeros((self.shape[0],self.shape[0]))
            else:
                proj = np.zeros((self.shape[1],self.shape[1]))

            if dt==0:
                for year in self.rangeyear:
                    proj += self.matrices[year].get_projection(method=method, rows=rows, top=top, alpha=alpha, verbose=verbose)
            
            else:
                for year in self.rangeyear[dt:]:
                    y0 = year - dt
                    proj += cooccurrence_matrix(self.matrices[year], mat2, row_proj=False)        
        
            return proj
        return None

    
    def get_smoothing_exponential_average(self, alpha=0.95, inplace=False):
        one_minus_alpha = 1.0-alpha
        matrices = {}
        rolling_matrix = self.matrices[self.rangeyear[0]].matrix.copy()
        matrices[self.rangeyear[0]] = rolling_matrix.copy()
        for count in range(1,len(self.rangeyear)):
            rolling_matrix = alpha*self.matrices[self.rangeyear[count]].matrix + one_minus_alpha*rolling_matrix
            matrices[self.rangeyear[count]] = rolling_matrix.copy()
        
        if inplace:
            del self.matrices
            self.matrices = matrices
            return self
        
        return efc_matrix_dataset(matrices, label_rows=self.label_rows, label_columns=self.label_columns)

    def get_smoothing_moving_average(self, k=3, inplace=False):
        matrices = {}
        for year in self.rangeyear:
            
            kleft = max(year-k,0)
            kright = min(year+k,len(self.rangeyear)-1)
            mat = self.matrices[year].matrix.copy()
            for k in range(kleft,year):
                mat += self.matrices[k].matrix
            for k in range(year+1,kright):
                mat += self.matrices[k].matrix 
            matrices[year] = mat.copy() / len(range(kleft,kright))
            
        if inplace:
            del self.matrices
            self.matrices = matrices
            return self
        
        return efc_matrix_dataset(matrices, label_rows=self.label_rows, label_columns=self.label_columns)
            

    def store_matrices(self, folder, name, store_type='csv'):
        for year in tqdm(self.rangeyear, leave=self.leave_tqdm):
            string_year = str(year)
            string_year = string_year.replace(' ', '_')
            name_file = folder + name.replace('{}',string_year) + '.' + store_type
            if store_type == 'csv':
                df = self.matrices[year].get_matrix(aspandas=True)
                df.to_csv(name_file)
            elif store_type == 'pkl' or store_type == 'pk':
                df = self.matrices[year].get_matrix(aspandas=True)
                df.to_pickle(name_file)
            
    def get_matrix(self, year, aspandas=False): # ok with sparse
        if year in self.rangeyear:
            return self.matrices[year].get_matrix(aspandas=aspandas)
        return None

        
    @classmethod
    def load_matrices(cls, path, rangeyear, verbose=False, leave_tqdm=True):
        supp = {}
        for year in tqdm(rangeyear, leave=leave_tqdm):
            path2 = path.replace('{}','{}'.format(year))
            mat = None
            if path[-4:] == '.csv':
                mat = pd.read_csv(path2, index_col=0)
            elif path[-4:] == '.pkl' or path[-3:] == '.pk':
                mat = pd.read_pickle(path2)
            elif path[-4:] == '.npz':
                mat = scipy.sparse.load_npz(path2)
            else:
                print('ERROR loading: ',path2)
            supp[year] = mat

        return cls(supp)
