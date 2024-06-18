import numpy as np
import pandas as pd
import random
import scipy

def get_edgelist_from_numpy(mat, label=None):
    if label is None:
        label = range(mat.shape[0])
    spa = scipy.sparse.coo_matrix(mat)
    return [(label[spa.row[i]],label[spa.col[i]], spa.data[i]) for i in range(len(spa.data))] 
    
def cooccurrence_matrix(mat1, mat2, row_proj=False):
    if row_proj:
        mat1 = mat1.transpose()
        mat2 = mat2.transpose()
    return np.matmul( mat1.transpose() , mat2 )

def occurrence_matrix(mat, row_proj=False):
    if row_proj:
        mat = mat.transpose()
    return np.matmul(mat.transpose(), mat)

def assist_matrix(mat1, mat2, row_proj=False):
    if row_proj:
        mat1 = mat1.transpose()
        mat2 = mat2.transpose()
    ubi = mat2.sum(0).astype('float')
    div = mat1.sum(1).astype('float')
    inv_div = np.zeros_like(div, dtype='float')
    inv_ubi = np.zeros_like(ubi, dtype='float')
    np.divide(np.ones_like(div,dtype=float), div, out=inv_div, where=div != 0)
    inv_div[inv_div == np.inf] = 0
    np.divide(np.ones_like(ubi,dtype=float), ubi, out=inv_ubi, where=ubi != 0)
    inv_ubi[inv_ubi == np.inf] = 0
    return np.matmul( np.nan_to_num((mat1).transpose()*inv_div) , np.nan_to_num(mat2) )

def random_binary_matrix(probability_matrix):
    dim = np.shape(probability_matrix)
    r = np.random.rand(dim[0],dim[1])
    return (r < probability_matrix).astype(int)
    
def taxonomy_matrix(mat, row_proj=False):
    if row_proj:
        mat = mat.transpose()
    ubi = mat.sum(0)
    div = mat.sum(1)
    inv_div = np.nan_to_num(1.0/div)
    inv_div[inv_div == np.inf] = 0
    inv_ubi = np.nan_to_num(1.0/ubi)
    inv_ubi[inv_ubi == np.inf] = 0
    tax = np.matmul( np.nan_to_num(mat.transpose()*inv_div) , np.nan_to_num(mat) )
    for p in range(len(ubi)):
        for pp in range(len(ubi)):
            tax[p,pp] /= np.max([ubi[p],ubi[pp]])
    return np.nan_to_num(tax, 0)

def proximity_matrix(mat, row_proj=False):
    if row_proj:
        mat = mat.transpose()
    Cooc = np.matmul(np.transpose(mat),mat)
    ubiquity = mat.sum(0)
    ubiMat = np.tile(ubiquity,[mat.shape[1],1])
    ubiMax = np.maximum(ubiMat,np.transpose(ubiMat)).astype(float)
    np.divide(np.ones_like(ubiMax,dtype=float), ubiMax, out=ubiMax, where=ubiMax != 0)
    return np.multiply(Cooc,ubiMax)

def net_max(tax):
    adj = np.zeros(np.shape(tax))
    NX,NY = np.shape(tax)
    for x in range(NX):
        row = tax[x,:].copy()
        row[x] = 0
        adj[x,np.argmax(row)] = 1
    return adj

def net_top(tax, top=2):
    NX,NY = np.shape(tax)
    adj = np.zeros(np.shape(tax))
    for x in range(NX):
        row = tax[x,:].copy()
        row[x] = 0
        adj[x,np.argsort(row)[::-1][:top]] = 1
    return adj

def net_atleast(tax, val=0.95):
    NX,NY = np.shape(tax)
    adj = np.zeros(np.shape(tax))
    for x in range(NX):
        row = tax[x,:].copy()
        atleast = row[x]*val
        row[x] = 0
        adj[x,np.where(row>atleast)[0]] = 1
    return adj

def net_threshold(tax, threshold=0.95):
    NX,NY = np.shape(tax)
    thr = tax.copy()
    thr[thr < threshold] = 0
    thr[thr > 0] = 1
    for x in range(NX):
        thr.iloc[x,x] = 0
    return np.nan_to_num(thr).astype(int)
