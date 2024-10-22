import numpy as np
import pandas as pd


def sNODF(bin_rca):
    '''
    Routine to evaluate the sNODF (Symetric NODF)
    
    :param bin_rca: raw/binary matrix to be considered in the analysis
    :type bin_rca: numpy.ndarray
        
    '''
    deg0 = bin_rca.sum(1)
    deg1 = bin_rca.sum(0)
    dim = bin_rca.shape
    kmat0 = np.array([deg0 for i in range(dim[0])])
    with np.errstate(divide='ignore'):
        kmat0 = 1.0/np.minimum(kmat0,kmat0.transpose())
        kmat0[kmat0 == np.inf] = 0
    kmat1 = np.array([deg1 for i in range(dim[1])])
    with np.errstate(divide='ignore'):
        kmat1 = 1.0/np.minimum(kmat1,kmat1.transpose())
        kmat1[kmat1 == np.inf] = 0
    cooc0 = np.dot(bin_rca,bin_rca.transpose())
    cooc1 = np.dot(bin_rca.transpose(),bin_rca)
    cooc0 = np.multiply(cooc0,kmat0)
    cooc1 = np.multiply(cooc1,kmat1)
    norm = 0.01*( dim[0]*(dim[0]-1) + dim[1]*(dim[1]-1) )
    return ( (cooc0.sum().sum()-cooc0.diagonal().sum()) + (cooc1.sum().sum()-cooc1.diagonal().sum())) / norm


def NODF(bin_rca, rowcol=False, removezero=False):
    '''
    Routine to evaluate the sNODF (Symetric NODF)
    
    :param bin_rca: raw/binary matrix to be considered in the analysis
    :type bin_rca: numpy.ndarray
        
    :param rowcol: return the row and column components
    :type rowcol: bool (False)
        
    :param removezero: remove the row and columns ith zero degree
    :type removezero: bool (False)
        
    '''
    if removezero:
        bin_rca.drop(columns=bin_rca.columns[bin_rca.sum(0)==0], index=bin_rca.index[bin_rca.sum(1)==0], inplace=True)
    deg0 = bin_rca.sum(1)
    deg1 = bin_rca.sum(0)
    dim = bin_rca.shape
    kmat0 = np.array([deg0 for i in range(dim[0])]).astype(np.float64)
    fill = ( kmat0.transpose() - kmat0 )
    kmat0[fill < 0] = 0
    np.divide(np.ones(kmat0.shape), kmat0, out=kmat0, where=kmat0 != 0)    
    kmat1 = np.array([deg1 for i in range(dim[1])]).astype(np.float64)
    fill = ( kmat1.transpose() - kmat1 )
    kmat1[fill < 0] = 0
    np.divide(np.ones(kmat1.shape), kmat1, out=kmat1, where=kmat1 != 0)
    cooc0 = np.dot(bin_rca,bin_rca.transpose())
    cooc1 = np.dot(bin_rca.transpose(),bin_rca)
    cooc0 = np.multiply(cooc0,kmat0)
    cooc1 = np.multiply(cooc1,kmat1)
    if rowcol:
        Nrows = (cooc0.sum().sum()-cooc0.diagonal().sum())
        Ncols = (cooc1.sum().sum()-cooc1.diagonal().sum())
        N = ( Nrows + Ncols )/(0.01*( dim[0]*(dim[0]-1) + dim[1]*(dim[1]-1) ))
        Nrows /= 0.01*dim[0]*(dim[0]-1)
        Ncols /= 0.01*dim[1]*(dim[1]-1)
        return (N,Nrows,Ncols)
    norm = 0.01*( dim[0]*(dim[0]-1) + dim[1]*(dim[1]-1) )
    return ( (cooc0.sum().sum()-cooc0.diagonal().sum()) + (cooc1.sum().sum()-cooc1.diagonal().sum())) / norm














