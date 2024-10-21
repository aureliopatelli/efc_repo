import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from collections import Counter


def objective_OT(pi, phi, rx=[], cy=[], epsilon=0, typefunc='entropy'):
    ot = pi * phi
    if typefunc == 'entropy':
        supp = np.log(pi)
        supp[supp == -np.inf] = 0.0
        ot += epsilon * np.multiply(pi, np.nan_to_num(supp) - 1)
    elif typefunc == 'kullback_leibler':
        dim = np.shape(pi)
        if np.shape(rx)[0] != dim[0] or np.shape(cy)[0] != dim[1]:
            return np.nan
        supp = rx.sum() / np.array([[c * r for c in cy] for r in rx])
        supp = np.log(np.multiply(pi, supp))
        supp[supp == -np.inf] = 0.0
        ot -= epsilon * np.multiply(pi, np.nan_to_num(supp))

    return ot.sum().sum()


'''
The computation of Sinkhorn using pandas database input format (although it works with numpy arrays)
mat = input matrix
sum0 = constraint on axis 0
sum1 = constraint on axis 1
max_iteration = max number of iteration before exit
min_distance = L1 distance below which the iterations stop
verbose = being verbose 
ic0 = initial condition along axis 0
norm = if implementing the local rescaling between the Sinkhorn potentials
epsilon = value (single) of epsilon
mass = if True, return also pi
renorm = rescale the matrix in order to simplify the iterations
'''


def sinkhorn_divergence(mat_orig, sum0, sum1, max_iteration=1000, min_distance=1e-14, verbose=0, ic0=[], norm=0, epsilon=0, mass=False, renorm=False, mask_mat=None, tau1=None, tau2=None):
    if type(mat_orig) == pd.core.frame.DataFrame:
        mat = mat_orig.to_numpy()
    else:
        mat = mat_orig
    if type(sum0) == pd.core.frame.Series:
        sum0 = sum0.to_numpy()
    if type(sum1) == pd.core.frame.Series:
        sum1 = sum1.to_numpy()
    dim = np.shape(mat)

    # checks
    if np.shape(sum0)[0] != dim[0]:
        print(np.shape(sum0)[0], dim[0])
        return None, None
    if np.shape(sum1)[0] != dim[1]:
        print(np.shape(sum1)[0], dim[1])
        return None, None
    sum0 = sum0.astype(np.longdouble)
    sum1 = sum1.astype(np.longdouble)

    # remove empty lines
    set0 = list(Counter((list(np.where(sum0 == 0)[0]) + list(np.where(mat.sum(1) == 0)[0]))).keys())
    set1 = list(Counter((list(np.where(sum1 == 0)[0]) + list(np.where(mat.sum(0) == 0)[0]))).keys())
    if len(set0):
        mat = np.delete(mat, set0, axis=0)
        sum0 = np.delete(sum0, set0)
    if len(set1):
        mat = np.delete(mat, set1, axis=1)
        sum1 = np.delete(sum1, set1)
    dim = np.shape(mat)

    # initial conditions
    u_old = np.ones((dim[0]), dtype=np.longdouble)
    u = np.ones((dim[0]), dtype=np.longdouble)
    v = np.ones((dim[1]), dtype=np.longdouble)
    v_old = np.ones((dim[1]), dtype=np.longdouble)

    # the initial conditions
    if np.shape(ic0)[0] == dim[0]:
        u_old[:] = ic0[:]
    else:
        u_old[:] = (sum0 / np.sum(sum0))[:]

    u_old = np.nan_to_num(u_old)

    # compute the runtime matrix
#    if tau1 is not None:
#        mat += tau1
#    if tau2 is not None:
#        mat += tau2
    if epsilon > 0:
        mat = np.exp(mat / epsilon)
#        mat = np.exp((mat - mat.max().max()) / epsilon)
    s0 = mat.sum(1)
    s1 = mat.sum(0)
    if renorm:
        mat = mat / np.array([[s1[i] * s0[j] for i in range(dim[1])] for j in range(dim[0])])
        mat = np.nan_to_num(mat)
    if mask_mat is not None:
        mat = mat*mask_mat
    mat_t = mat.transpose().copy()

    vexponential = 0
    if tau1 is not None:
        vexponential = tau1/(tau1+epsilon)
    uexponential = 0
    if tau2 is not None:
        uexponential = tau2/(tau2+epsilon)

    # the main loop
    for iterat in range(max_iteration):
        # the single interation
        v[:] = np.nan_to_num(sum1 / np.dot(mat_t, u_old))[:]
        u[:] = np.nan_to_num(sum0 / np.dot(mat, v))[:]

        if vexponential:
            v[:] = np.power(v[:],vexponential)
        if uexponential:
            u[:] = np.power(u[:],uexponential)

        # check and renormalize if necessary
        if norm == 1:
            nu = np.sum(u)
            nv = np.sum(v)
            a = np.sqrt(nu / nv)
            u /= a
            v *= a
            if verbose == 2:
                print(a)
        elif norm == 2:  # this is not correct, but who know
            u /= np.sum(u)
            v /= np.sum(v)

        # check the time scale
        distance = np.abs(u - u_old).sum()
        if verbose > 0:
            print(iterat, distance, np.std(u / u_old))
            if verbose == 3:
                print(np.max(u), np.min(u), np.max(v), np.min(v))
                print('')

        if iterat > max_iteration // 50:
            if distance < min_distance:
                break

        # store the latest values
        u_old[:] = u[:]
        v_old[:] = v[:]

    if renorm:
        u[:] /= s0[:]
        v[:] /= s1[:]
        nu = np.sum(u)
        nv = np.sum(v)
        a = np.sqrt(nu / nv)
        u[:] /= a
        v[:] *= a

    pi = (np.transpose(np.transpose(mat) * u) * v)
    obj_ = objective_OT(pi, mat, rx=sum0, cy=sum1, epsilon=epsilon, typefunc='entropy')

    # reimport the empty lines
    if len(set0) or len(set1):
        mask0 = np.array([True for e in range(len(set0)+len(u))])
        mask0[set0] = False
        mask1 = np.array([True for e in range(len(set1)+len(v))])
        mask1[set1] = False

        ufinal = np.zeros_like(mat_orig.sum(1))
        ufinal[~mask0] = np.nan
        ufinal[mask0] = u

        vfinal = np.zeros_like(mat_orig.sum(0))
        vfinal[~mask1] = np.nan
        vfinal[mask1] = v

        u = ufinal
        v = vfinal

        if mass:
            matfinal = np.nan * np.ones_like(mat_orig)
            maskmat = np.outer(mask0, mask1)
            matfinal[maskmat] = mat.flatten()
            pi = (np.transpose(np.transpose(matfinal) * u) * v)

    if mass:
        return u, v, pi, obj_
    else:
        return u, v


def sinkhorn_divergence_logs(mat_org, sum0, sum1, max_iteration=1000, min_distance=1e-14, mass=False, verbose=0, ic1=[], norm=1, epsilon=[1.0, 0.1, 0.01, 0.001, 0.0001], tims_random=10, mask_mat=None):
    '''

    :param mat_org:
    :param sum0:
    :param sum1:
    :param max_iteration: (default 1000)
    :param min_distance: (default 1e-14)
    :param mass: set to return the mass (aka couplings, ...) (default False)
    :param verbose:
    :param ic1:
    :param norm:
    :param epsilon:
    :param tims_random:
    :return: alpha, beta (, pi if mass=True)
    '''
    if type(mat_org) == pd.core.frame.DataFrame:
        mat = mat_org.to_numpy()
    else:
        mat = mat_org
    if mask_mat is not None:
        mat = mat*mask_mat
    if type(sum0) == pd.core.frame.Series:
        sum0 = sum0.to_numpy()
    if type(sum1) == pd.core.frame.Series:
        sum1 = sum1.to_numpy()
    dim = np.shape(mat)

    # checks
    if np.shape(sum0)[0] != dim[0]:
        print(np.shape(sum0)[0], dim[0])
        return None, None
    if np.shape(sum1)[0] != dim[1]:
        print(np.shape(sum1)[0], dim[1])
        return None, None
    if epsilon == 0:
        return None, None

    if type(epsilon) == float:
        epsilon = np.array([epsilon], dtype=np.longdouble)
    else:
        epsilon = np.array(epsilon, dtype=np.longdouble)

    # remove empty lines
    set0 = list(Counter((list(np.where(sum0 == 0)[0]) + list(np.where(mat.sum(1) == 0)[0]))).keys())
    set1 = list(Counter((list(np.where(sum1 == 0)[0]) + list(np.where(mat.sum(0) == 0)[0]))).keys())
    if len(set0):
        mat = np.delete(mat, set0, axis=0)
        sum0 = np.delete(sum0, set0)
    if len(set1):
        mat = np.delete(mat, set1, axis=1)
        sum1 = np.delete(sum1, set1)
    dim = np.shape(mat)

    # set the base
    mat_t = mat.transpose().copy()
    S0 = sum0.astype(np.longdouble).copy()
    S1 = sum1.astype(np.longdouble).copy()
    sum0 = np.nan_to_num(np.log(sum0)).astype(np.longdouble)
    sum1 = np.nan_to_num(np.log(sum1)).astype(np.longdouble)

    S0 /= S0.sum()
    S1 /= S1.sum()

    # initial conditions
    alpha = np.ones((dim[0]), dtype=np.longdouble)
    beta = np.ones((dim[1]), dtype=np.longdouble)
    beta_old = np.ones((dim[1]), dtype=np.longdouble)

    # the initial conditions
    if np.shape(ic1)[0] == dim[1]:
        beta[:] = ic1[:]
    else:
        beta[:] = epsilon[0] * sum1[:]

    beta = np.nan_to_num(beta)
    beta_old[:] = beta[:]

    for eps in epsilon:
        cnt_eps = 0.02  # *np.tanh(eps)
        # the main loop
        for iterat in range(max_iteration):

            # the alpha case
            x = mat - beta
            xmax = x.max(1)
            x = (x.transpose() - xmax).transpose()
            alpha = xmax - eps * (sum0 - np.log(np.exp(x / eps).sum(1)))

            # the beta case
            y = mat_t - alpha
            ymax = y.max(1)
            y = (y.transpose() - ymax).transpose()
            beta = ymax - eps * (sum1 - np.log(np.exp(y / eps).sum(1)))

            if norm == 1:
                deltamean = 0.5 * (np.mean(alpha) - np.mean(beta))
                alpha = alpha - deltamean
                beta = beta + deltamean

            # check the time scale
            distance = np.abs(beta - beta_old).sum()
            if verbose > 0:
                opt = np.dot(S0, alpha) + np.dot(S1, beta)
                print(eps, iterat, opt, distance, np.std(beta / beta_old))
                if verbose > 1:
                    print(np.max(alpha), np.min(alpha), np.max(beta), np.min(beta))
                    print('')

            if iterat > max_iteration // 50:
                if distance < min_distance:
                    break

            if tims_random > 0 and iterat % (max_iteration / tims_random) == 0 and iterat > 0:
                beta = beta * ((1.0 - cnt_eps * 0.5) + cnt_eps * np.random.rand(beta.shape[0]))

            # store the latest values
            beta_old[:] = beta[:]

    # reimport the empty lines
    if len(set0):
        try:
            alpha = np.insert(alpha, set0, np.nan)
        except:
            alpha = np.append(np.insert(alpha, set0[:-1], np.nan), np.nan)
    if len(set1):
        try:
            beta = np.insert(beta, set1, np.nan)
        except:
            beta = np.append(np.insert(beta, set1[:-1], np.nan), np.nan)

    # compute the mass if necessary
    if mass:
        pi = np.exp(((mat_org - beta).transpose() - alpha).transpose() / epsilon[-1])
        return alpha, beta, pi, None
    else:
        return alpha, beta



#def sinkhorn_distance():

