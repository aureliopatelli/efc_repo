import pandas as pd
import numpy as np

def eci_sparse(mat, witheigenvalue=False, ztransform = True):
    """
    Routine to evaluate ECI using sparse matrix representation
    
    :param mat: matrix representing the adjacency bipartite network
    :type mat: numpy.array, numpy.matrix, pandas.DataFrame
    
    :param witheigenvalue: return both the eigenvalue and the metric
    :type witheigenvalue: bool (False)
    
    :param ztransform: evaluate the standardization of the eigenvector (for ECI, set True)
    :type ztransform: bool (True)
    
    """
    matT = mat.transpose()
    pi = mat.sum(1)
    mat = mat.multiply(np.nan_to_num(1.0/pi))
    matT = matT.multiply(np.nan_to_num(1.0/matT.sum(1)))
    prod = mat*matT
    vals, vecs = la.eigs(prod, k=2, v0=pi/pi.sum())
    x,y = vecs.T
    x = np.array(x.real)
    y = np.array(y.real)
    if ztransform:
        y = (y-y.mean())/y.std()
        y[np.abs(y)<1e-14] = 0
    if witheigenvalue:
        return vals, y    
    return y.flatten()

def pci_sparse(mat, witheigenvalue=False, ztransform = True):
    """
    Routine to evaluate ECI using sparse matrix representation
    
    :param mat: matrix representing the adjacency bipartite network
    :type mat: numpy.array, numpy.matrix, pandas.DataFrame
    
    :param witheigenvalue: return both the eigenvalue and the metric
    :type witheigenvalue: bool (False)
    
    :param ztransform: evaluate the standardization of the eigenvector (for PCI, set True)
    :type ztransform: bool (True)
    
    """
    matT = mat.copy
    mat = matT.transpose()
    pi = mat.sum(1)
    mat = mat.multiply(np.nan_to_num(1.0/pi))
    matT = matT.multiply(np.nan_to_num(1.0/matT.sum(1)))
    prod = mat*matT
    vals, vecs = la.eigs(prod, k=2, v0=pi/pi.sum())
    x,y = vecs.T
    x = np.array(x.real)
    y = np.array(y.real)
    if ztransform:
        y = (y-y.mean())/y.std()
        y[np.abs(y)<1e-14] = 0
    if witheigenvalue:
        return vals, y    
    return y.flatten()

def eci_dense(mat, witheigenvalue=False, ztransform = True, eigenvalue=2, transpose=False):
    """
    Routine to evaluate ECI using dense matrix representation
    
    :param mat: matrix representing the adjacency bipartite network
    :type mat: numpy.array, numpy.matrix, pandas.DataFrame
    
    :param witheigenvalue: return both the eigenvalue and the metric
    :type witheigenvalue: bool (False)
    
    :param ztransform: evaluate the standardization of the eigenvector (for ECI, set True)
    :type ztransform: bool (True)
    
    :param eigenvalue: set the eigenvector considered (2 for PCI)
    :type eigenvalue: integer 2
    
    :param transpose: evaluate the eigenvectors of the transposed matrix (thus evaluate the left eigenvectors)
    :type transpose: bool (False)
    
    """
    prob_mat = (mat.transpose()/mat.sum(1)).transpose()
    prob_mat_t = np.nan_to_num((mat/mat.sum(0)).transpose())
    w = prob_mat.dot(prob_mat_t)
    if transpose:
        values, vectors = np.linalg.eig(w.transpose())
    else:
        values, vectors = np.linalg.eig(w)
    vec = pd.DataFrame(vectors.transpose()[eigenvalue-1].real, index = np.arange(w.shape[0]))
    #vec *= np.sign(np.corrcoef(vec[0].to_numpy(),mat.sum(1)))
    if ztransform:
        vec = (vec-vec.mean())/vec.std()
        vec *= np.sign(mat.sum(1).dot(vec)[0])
    if witheigenvalue:
        return values[eigenvalue-1].real, vec.to_numpy().flatten()
    return vec.to_numpy().flatten()
    
def pci_dense(mat, witheigenvalue=False, ztransform = True, eigenvalue=2, transpose=False):
    """
    Routine to evaluate ECI using dense matrix representation
    
    :param mat: matrix representing the adjacency bipartite network
    :type mat: numpy.array, numpy.matrix, pandas.DataFrame
    
    :param witheigenvalue: return both the eigenvalue and the metric
    :type witheigenvalue: bool (False)
    
    :param ztransform: evaluate the standardization of the eigenvector (for PCI, set True)
    :type ztransform: bool (True)
    
    :param eigenvalue: set the eigenvector considered (2 for PCI)
    :type eigenvalue: integer 2
    
    :param transpose: evaluate the eigenvectors of the transposed matrix (thus evaluate the left eigenvectors)
    :type transpose: bool (False)
    
    """
    prob_mat = (mat.transpose()/mat.sum(1)).transpose()
    prob_mat_t = np.nan_to_num((mat/mat.sum(0)).transpose())
    w = prob_mat_t.dot(prob_mat)
    if transpose:
        values, vectors = np.linalg.eig(w.transpose())
    else:
        values, vectors = np.linalg.eig(w)
    vec = pd.DataFrame(vectors.transpose()[eigenvalue-1].real, index = np.arange(w.shape[0]))
    if ztransform:
        vec = (vec-vec.mean())/vec.std()
        vec *= np.sign(mat.sum(0).dot(vec)[0])
    if witheigenvalue:
        return values[eigenvalue-1].real, vec.to_numpy().flatten()
    return vec.to_numpy().flatten()
