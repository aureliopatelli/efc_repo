import pandas as pd
import numpy as np


def normalize(vector, normalization='mean'):
    """
    Routine to normalize a vector
    
    :param vector: vector to normalize
    :type vector: numpy.ndarray
    
    :param normalization: define the type of normalization
    :type normalization: 'mean', 'max', ('mean') 
    
    :return: numpy.ndarray of the vector
    """
    if normalization == 'mean':
        return vector / vector.sum(0)
    elif normalization == 'max':
        return vector / vector.max(0)
    else:
        return vector / vector.mean(0)
    return vector

def minimum_cossing_time(vector, iterat, tail):
    """
    Routine to estimate the minimum crossing time of the fitness/complexity ranks 
    
    :param vector: vector of fitness+complexity
    :type vector: numpy.ndarray
    
    :param iterat: internal reference of the updating list
    :type iterat: integer
    
    :param tail: dimension of the updating list
    :type tail: integer
    
    :return: integer of minimum crossing time
    """
    newpos = iterat+1
    growth = (np.log(vector[newpos%tail]) - np.log(vector[(iterat)%tail]))/np.log(1.0*newpos/iterat)
    rank_df = pd.DataFrame(zip(vector[newpos%tail],growth), index=vector.index, columns=['value','growth'] ).sort_values('value')
    rank_shift = rank_df[['value','growth']].shift(-1)
    rank_diff = rank_shift[['growth']] - rank_df[['growth']]
    rank_rat = rank_df[['value']] / rank_shift[['value']]
    rank_diff.columns=['value']
    rat = np.power( np.power(newpos,rank_diff)*rank_rat , 1.0/rank_diff)
    return rat[rank_diff<0]['value'].min()

def fitness_complexity(bin_rca, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, redundant=False, add_noise=False, consider_dummy=False):
    """
    Routine to evaluate the fitness and the complexity from Tacchella et al. (2012), 'A new metrics for countries' Fitness and products Complexity', Sci.Rep.
    
    :param bin_rca: matrix representing the bipartite network
    :type bin_rca: numpy.array, pandas.DataFrame, scpi.sparse
    
    :param max_iteration: maximum number of iterations
    :type max_iteration: integer
    
    :param check_stop: stop condition to consider
    :type check_stop: 'distance', 'crossing time', None ('distance')
    
    :param min_distance: L1 distance of the iteration updates below which the loop is terminated
    :type min_distance: float
    
    :param normalization: type of normalization condition
    :type normalization: 'mean', 'max' ('mean')
    
    :param fit_ic: if defined, array of the fitness initial condition
    :type fit_ic: numpy.array, list
    
    :param com_ic: if defined, array of the complexity initial condition 
    :type com_ic: numpy,array, list
    
    :param removelowdegrees: threshold of the degree of the columns to be removed from the analysis
    :type removelowdegrees: bool, integer (False)
    
    :param verbose: being verbose
    :type verbose:  bool (False)
    
    :param redundant: evaluate the fitness using the newly updated complexity
    :type redundant:  bool (False)
    
    :param add_noise: add a noise in the first tenth of iteration
    :type add_noise:  bool (False)
    
    :param consider_dummy: consider the algorithm as if a dummy row was added
    :type consider_dummy: bool (False)
    
    :return: numpy.ndarray of fitness and complexity
    """
    dim = bin_rca.shape
    typemat = 'df'
    
    if consider_dummy == True and verbose:
        print('considering also a dummy')
    
    if hasattr(bin_rca,'getformat'):
        if bin_rca.getformat()=='csr':
            index = np.arange(dim[0])
            columns = np.arange(dim[1])
            typemat = 'sp'
            if verbose:
                print('# sparse matrix ({},{})'.format(dim[0],dim[1]))
        else:
            return None, None
        if removelowdegrees:
            if type(removelowdegrees) == int:
                for i in range(1,removelowdegrees+1):
                    where = np.where(bin_rca.sum(0)<=i)[1]
                    bin_rca[:,where] = np.array([0 for j in range(len(where))])
                bin_rca.eliminate_zeros()
            elif type(removelowdegrees) == float:
                where = np.where(bin_rca.sum(0) <= removelowdegrees)[0]
                bin_rca[:, where] = np.array([0 for j in range(len(where))])
                bin_rca.eliminate_zeros()
    elif isinstance(bin_rca, np.ndarray):
        index = np.arange(dim[0])
        columns = np.arange(dim[1])
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0],dim[1]))
        if removelowdegrees:
            if type(removelowdegrees) == int:
                for i in range(1,removelowdegrees+1):
                    where = np.where(bin_rca.sum(0)<=i)[0]
                    bin_rca[:,where] = 0
            elif type(removelowdegrees) == float:
                where = np.where(bin_rca.sum(0) <= removelowdegrees)[0]
                bin_rca[:, where] = 0
    else:
        index = bin_rca.index
        columns = bin_rca.columns
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0],dim[1]))
        if removelowdegrees:
            if type(removelowdegrees) == int:
                for i in range(1,removelowdegrees+1):
                    where = np.where(bin_rca.sum(0)<=i)[0]
                    bin_rca.iloc[:,where] = 0
            elif type(removelowdegrees) == float:
                where = np.where(bin_rca.sum(0) <= removelowdegrees)[0]
                bin_rca.iloc[:, where] = 0

    bin_rca_t = bin_rca.transpose()
    tail = 2
    if check_stop == 'crossing time':
        tail = 3
    
    # initial condition
    fit = pd.DataFrame(np.ones((dim[0],tail)), index=index, columns=range(tail), dtype=np.longdouble)
    com = pd.DataFrame(np.ones((dim[1],tail)), index=columns, columns=range(tail), dtype=np.longdouble)
    if np.shape(fit_ic)[0] == dim[0]:
        fit[0] = fit_ic
    if np.shape(com_ic)[0] == dim[1]:
        com[0] = com_ic
    if consider_dummy:
        fit /= com.sum(0)
        com /= com.sum(0)
    else:
        fit = normalize(fit, normalization)
        com = normalize(com, normalization)
    
    ones_row = np.ones(dim[0])
    ones_col = np.ones(dim[1])
    
    # loop
    for iterat in range(max_iteration):
        colpos = iterat%tail

        # evaluate the single iteration
        fit_here = np.zeros(dim[0])
        np.divide(ones_row, fit[colpos], out=fit_here, where=fit[colpos] != 0)
        fit_here[fit_here == np.inf] = 0.0
        if consider_dummy:
            if typemat == 'sp':
                com_here = 1.0 + bin_rca_t.dot(fit_here)
                np.divide(ones_col, com_here, out=com_here, where=com_here != 0)
                com_here /= com_here.sum()
                if redundant:
                    fit_here = bin_rca.dot(com_here)
                else:
                    fit_here = bin_rca.dot(com[colpos])
                    fit_here /= com[colpos].sum()
            else:
                com_here = 1.0 + np.dot(bin_rca_t,fit_here)
                np.divide(ones_col, com_here, out=com_here, where=com_here != 0)
                com_here /= com_here.sum()
                if redundant:
                    fit_here = np.dot(bin_rca,com_here)
                else:
                    fit_here = np.dot(bin_rca,com[colpos])
        else:
            if typemat == 'sp':
                com_here = bin_rca_t.dot(fit_here)
                np.divide(ones_col, com_here, out=com_here, where=com_here != 0)
                if redundant:
                    fit_here = bin_rca.dot(com_here)
                else:
                    fit_here = bin_rca.dot(com[colpos])
            else:
                com_here = np.dot(bin_rca_t,fit_here)
                np.divide(ones_col, com_here, out=com_here, where=com_here != 0)
                if redundant:
                    fit_here = np.dot(bin_rca,com_here)
                else:
                    fit_here = np.dot(bin_rca,com[colpos])

        # add a possible multiplicative noise on the Fitness
        if add_noise and iterat%100 == 0 and iterat>0 and iterat < max_iteration//2:
            strength = (max_iteration - 2*iterat) / max_iteration
            fit_here *= 1.0-strength + strength*np.random.uniform(size = fit_here.shape)

        # normalization
        newpos = (iterat+1)%tail
        if consider_dummy:
            fit[newpos] = fit_here
            com[newpos] = com_here
        else:
            fit[newpos] = normalize(fit_here, normalization)
            com[newpos] = normalize(com_here, normalization)

        # check the time scale
        if check_stop == 'crossing time':
            if (iterat%8 == 7) and (iterat > max_iteration//10):
                minimum = minimum_cossing_time(fit, iterat, tail)
                if minimum+iterat+1>max_iteration:
                    break
        elif check_stop == 'distance':
            distance = np.abs(fit[newpos]-fit[colpos]).sum()
            if verbose:
                print(iterat,distance)
            if iterat > max_iteration//10:
                if distance < min_distance:
                    break
    
    if normalization == 'mean' and consider_dummy == False:
        return fit[newpos]*dim[0], com[newpos]*dim[1]
    
    return fit[newpos], com[newpos]



def fitness_complexity_servedio(bin_rca, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, redundant=False, add_noise=False, delta=1.0):
    """
    Routine to evaluate the fitness and the complexity from Servedio et al. (2018), 'A New and Stable Estimation Method of Country Economic Fitness and Product Complexity', Entropy
    
    :param bin_rca: matrix representing the bipartite network
    :type bin_rca: numpy.array, pandas.DataFrame, scpi.sparse
    
    :param max_iteration: maximum number of iterations
    :type max_iteration: integer
    
    :param check_stop: stop condition to consider
    :type check_stop: 'distance', 'crossing time', None ('distance')
    
    :param min_distance: L1 distance of the iteration updates below which the loop is terminated
    :type min_distance: float
    
    :param normalization: type of normalization condition
    :type normalization: 'mean', 'max' ('mean')
    
    :param fit_ic: if defined, array of the fitness initial condition
    :type fit_ic: numpy.array, list
    
    :param com_ic: if defined, array of the complexity initial condition 
    :type com_ic: numpy,array, list
    
    :param removelowdegrees: threshold of the degree of the columns to be removed from the analysis
    :type removelowdegrees: bool, integer (False)
    
    :param verbose: being verbose
    :type verbose:  bool (False)
    
    :param redundant: evaluate the fitness using the newly updated complexity
    :type redundant:  bool (False)
    
    :param add_noise: add a noise in the first tenth of iteration
    :type add_noise:  bool (False)
    
    :param delta: the parameter delta of the algorithm
    :type delta: float (1.0)
    
    :return: numpy.ndarray of fitness and complexity
    """
    dim = bin_rca.shape
    typemat = 'df'
        
    if hasattr(bin_rca,'getformat'):
        if bin_rca.getformat()=='csr':
            index = np.arange(dim[0])
            columns = np.arange(dim[1])
            typemat = 'sp'
            if verbose:
                print('# sparse matrix ({},{})'.format(dim[0],dim[1]))
        else:
            return None, None
        if removelowdegrees:
            for i in range(1,removelowdegrees+1):
                where = np.where(bin_rca.sum(0)==i)[1]
                bin_rca[:,where] = np.array([0 for j in range(len(where))])
            bin_rca.eliminate_zeros()
    elif isinstance(bin_rca, np.ndarray):
        index = np.arange(dim[0])
        columns = np.arange(dim[1])
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0],dim[1]))
        if removelowdegrees:
            for i in range(1,removelowdegrees+1):
                where = np.where(bin_rca.sum(0)==i)[0]
                bin_rca[:,where] = 0
    else:
        index = bin_rca.index
        columns = bin_rca.columns
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0],dim[1]))
        if removelowdegrees:
            for i in range(1,removelowdegrees+1):
                where = np.where(bin_rca.sum(0)==i)[0]
                bin_rca.iloc[:,where] = 0
            
    
    bin_rca_t = bin_rca.transpose()
    tail = 2
    if check_stop == 'crossing time':
        tail = 3
    
    # initial condition
    fit = pd.DataFrame(np.ones((dim[0],tail)), index=index, columns=range(tail), dtype=np.longdouble)
    com = pd.DataFrame(np.ones((dim[1],tail)), index=columns, columns=range(tail), dtype=np.longdouble)
    if np.shape(fit_ic)[0] == dim[0]:
        fit[0] = fit_ic
    if np.shape(com_ic)[0] == dim[1]:
        com[0] = com_ic
    fit = normalize(fit, normalization)
    com = normalize(com, normalization)
    
    ones_row = np.ones(dim[0])
    ones_col = np.ones(dim[1])
    
    # loop
    for iterat in range(max_iteration):
        colpos = iterat%tail

        # evaluate the single iteration
        fit_here = np.zeros(dim[0])
        np.divide(ones_row, fit[colpos], out=fit_here, where=fit[colpos] != 0)
        fit_here[fit_here == np.inf] = 0.0
        if typemat == 'sp':
            com_here = 1.0 + bin_rca_t.dot(fit_here)
            np.divide(ones_col, com_here, out=com_here, where=com_here != 0)
            com_here /= com_here.sum()
            if redundant:
                fit_here = bin_rca.dot(com_here)
            else:
                fit_here = bin_rca.dot(com[colpos])
                fit_here /= com[colpos].sum()
        else:
            com_here = 1.0 + np.dot(bin_rca_t,fit_here)
            np.divide(ones_col, com_here, out=com_here, where=com_here != 0)
            com_here /= com_here.sum()
            if redundant:
                fit_here = np.dot(bin_rca,com_here)
            else:
                fit_here = np.dot(bin_rca,com[colpos])
                
        fit_here += delta

        # add a possible multiplicative noise on the Fitness
        if add_noise and iterat%100 == 0 and iterat>0 and iterat < max_iteration//2:
            strength = (max_iteration - 2*iterat) / max_iteration
            fit_here *= 1.0-strength + strength*np.random.uniform(size = fit_here.shape)

        # normalization
        newpos = (iterat+1)%tail
        fit[newpos] = normalize(fit_here, normalization)
        com[newpos] = normalize(com_here, normalization)

        # check the time scale
        if check_stop == 'crossing time':
            if (iterat%8 == 7) and (iterat > max_iteration//10):
                minimum = minimum_cossing_time(fit, iterat, tail)
                if minimum+iterat+1>max_iteration:
                    break
        elif check_stop == 'distance':
            distance = np.abs(fit[newpos]-fit[colpos]).sum()
            if verbose:
                print(iterat,distance)
            if iterat > max_iteration//10:
                if distance < min_distance:
                    break
    
    if normalization == 'mean':
        return fit[newpos]*dim[0], com[newpos]*dim[1]
    
    return fit[newpos], com[newpos]

def fitness_complexity_mazzolini(bin_rca, max_iteration = 1000, check_stop='distance', min_distance=1e-14, normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False, gamma=-1, alpha=1.0):
    """
    Routine to evaluate the fitness and the complexity from Tacchella et al. (2012), 'A new metrics for countries' Fitness and products Complexity', Sci.Rep.
    
    :param bin_rca: matrix representing the bipartite network
    :type bin_rca: numpy.array, pandas.DataFrame, scpi.sparse
    
    :param max_iteration: maximum number of iterations
    :type max_iteration: integer
    
    :param check_stop: stop condition to consider
    :type check_stop: 'distance', 'crossing time', None ('distance')
    
    :param min_distance: L1 distance of the iteration updates below which the loop is terminated
    :type min_distance: float
    
    :param normalization: type of normalization condition
    :type normalization: 'mean', 'max' ('mean')
    
    :param fit_ic: if defined, array of the fitness initial condition
    :type fit_ic: numpy.array, list
    
    :param com_ic: if defined, array of the complexity initial condition 
    :type com_ic: numpy,array, list
    
    :param removelowdegrees: threshold of the degree of the columns to be removed from the analysis
    :type removelowdegrees: bool, integer (False)
    
    :param verbose: being verbose
    :type verbose:  bool (False)
    
    :param redundant: evaluate the fitness using the newly updated complexity
    :type redundant:  bool (False)
    
    :param add_noise: add a noise in the first tenth of iteration
    :type add_noise:  bool (False)
    
    :return: numpy.ndarray of fitness and complexity
    """
    dim = bin_rca.shape
    typemat = 'df'
    
    if hasattr(bin_rca,'getformat'):
        if bin_rca.getformat()=='csr':
            index = np.arange(dim[0])
            columns = np.arange(dim[1])
            typemat = 'sp'
            if verbose:
                print('# sparse matrix ({},{})'.format(dim[0],dim[1]))
        else:
            return None, None
        if removelowdegrees:
            for i in range(1,removelowdegrees+1):
                where = np.where(bin_rca.sum(0)<=i)[1]
                bin_rca[:,where] = np.array([0 for j in range(len(where))])
            bin_rca.eliminate_zeros()
    elif isinstance(bin_rca, np.ndarray):
        index = np.arange(dim[0])
        columns = np.arange(dim[1])
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0],dim[1]))
        if removelowdegrees:
            for i in range(1,removelowdegrees+1):
                where = np.where(bin_rca.sum(0)<=i)[0]
                bin_rca[:,where] = 0
    else:
        index = bin_rca.index
        columns = bin_rca.columns
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0],dim[1]))
        if removelowdegrees:
            for i in range(1,removelowdegrees+1):
                where = np.where(bin_rca.sum(0)<=i)[0]
                bin_rca.iloc[:,where] = 0
            
    
    bin_rca_t = bin_rca.transpose()
    tail = 2
    if check_stop == 'crossing time':
        tail = 3
    
    # initial condition
    fit = pd.DataFrame(np.ones((dim[0],tail)), index=index, columns=range(tail), dtype=np.longdouble)
    com = pd.DataFrame(np.ones((dim[1],tail)), index=columns, columns=range(tail), dtype=np.longdouble)
    if np.shape(fit_ic)[0] == dim[0]:
        fit[0] = fit_ic
    if np.shape(com_ic)[0] == dim[1]:
        com[0] = com_ic
    fit = normalize(fit, normalization)
    com = normalize(com, normalization)
        
    # loop
    for iterat in range(max_iteration):
        colpos = iterat%tail
        
        # evaluate the single iteration
        x = fit[colpos].copy()
        y = com[colpos].copy() 
        mask_x = (x>0)
        mask_y = (y>0)
        x[mask_x] = np.power(x[mask_x], gamma)
        y[mask_y] = np.power(y[mask_y], gamma)
        
        if typemat == 'sp':
            com_here = bin_rca_t.dot(x)
            fit_here = bin_rca.dot(y)
        else:
            com_here = np.dot(bin_rca_t,x)
            fit_here = np.dot(bin_rca,y)
            
        fit_here[fit_here<min_distance] = 0
        com_here[com_here<min_distance] = 0

        # normalization
        newpos = (iterat+1)%tail
        fit[newpos] = (1.-alpha)*fit[colpos] + alpha*normalize(fit_here, normalization)
        com[newpos] = (1.-alpha)*com[colpos] + alpha*normalize(com_here, normalization)
        
        if np.sum(fit[newpos] > 1e4*dim[0])>0 or np.sum(com[newpos] > 1e4*dim[1])>0:
            if verbose:
                print('there are values very large')
            break
        
        if np.sum(np.isnan(fit[newpos]))>0 or np.sum(np.isnan(com[newpos]))>0:
            if verbose:
                print('there are NaNs')
            newpos = colpos
            break
    
    
        # check the time scale
        if check_stop == 'crossing time':
            if (iterat%8 == 7) and (iterat > max_iteration//10):
                minimum = minimum_cossing_time(fit, iterat, tail)
                if minimum+iterat+1>max_iteration:
                    break
        elif check_stop == 'distance':
            distance = np.abs(fit[newpos]-fit[colpos]).sum()
            if verbose:
                print(iterat,distance)
            if iterat > max_iteration//10:
                if distance < min_distance:
                    break
                    
    return fit[newpos], com[newpos]


def fitness_complexity_mariani(bin_rca, max_iteration=1000, check_stop='distance', min_distance=1e-14,
                                 normalization='mean', fit_ic=[], com_ic=[], removelowdegrees=False, verbose=False):
    """
    Routine to evaluate the fitness and the complexity from Tacchella et al. (2012), 'A new metrics for countries' Fitness and products Complexity', Sci.Rep.

    :param bin_rca: matrix representing the bipartite network
    :type bin_rca: numpy.array, pandas.DataFrame, scpi.sparse

    :param max_iteration: maximum number of iterations
    :type max_iteration: integer

    :param check_stop: stop condition to consider
    :type check_stop: 'distance', 'crossing time', None ('distance')

    :param min_distance: L1 distance of the iteration updates below which the loop is terminated
    :type min_distance: float

    :param normalization: type of normalization condition
    :type normalization: 'mean', 'max' ('mean')

    :param fit_ic: if defined, array of the fitness initial condition
    :type fit_ic: numpy.array, list

    :param com_ic: if defined, array of the complexity initial condition
    :type com_ic: numpy,array, list

    :param removelowdegrees: threshold of the degree of the columns to be removed from the analysis
    :type removelowdegrees: bool, integer (False)

    :param verbose: being verbose
    :type verbose:  bool (False)

    :param redundant: evaluate the fitness using the newly updated complexity
    :type redundant:  bool (False)

    :param add_noise: add a noise in the first tenth of iteration
    :type add_noise:  bool (False)

    :return: numpy.ndarray of fitness and complexity
    """
    dim = bin_rca.shape
    typemat = 'df'

    if hasattr(bin_rca, 'getformat'):
        if bin_rca.getformat() == 'csr':
            index = np.arange(dim[0])
            columns = np.arange(dim[1])
            typemat = 'sp'
            if verbose:
                print('# sparse matrix ({},{})'.format(dim[0], dim[1]))
        else:
            return None, None
        if removelowdegrees:
            for i in range(1, removelowdegrees + 1):
                where = np.where(bin_rca.sum(0) <= i)[1]
                bin_rca[:, where] = np.array([0 for j in range(len(where))])
            bin_rca.eliminate_zeros()
    elif isinstance(bin_rca, np.ndarray):
        index = np.arange(dim[0])
        columns = np.arange(dim[1])
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0], dim[1]))
        if removelowdegrees:
            for i in range(1, removelowdegrees + 1):
                where = np.where(bin_rca.sum(0) <= i)[0]
                bin_rca[:, where] = 0
    else:
        index = bin_rca.index
        columns = bin_rca.columns
        if verbose:
            print('# dense matrix ({},{})'.format(dim[0], dim[1]))
        if removelowdegrees:
            for i in range(1, removelowdegrees + 1):
                where = np.where(bin_rca.sum(0) <= i)[0]
                bin_rca.iloc[:, where] = 0

    bin_rca_t = bin_rca.transpose()
    tail = 2
    if check_stop == 'crossing time':
        tail = 3

    # initial condition
    fit = pd.DataFrame(np.ones((dim[0], tail)), index=index, columns=range(tail), dtype=np.longdouble)
    com = pd.DataFrame(np.ones((dim[1], tail)), index=columns, columns=range(tail), dtype=np.longdouble)
    if np.shape(fit_ic)[0] == dim[0]:
        fit[0] = fit_ic
    if np.shape(com_ic)[0] == dim[1]:
        com[0] = com_ic
    fit = normalize(fit, normalization)
    com = normalize(com, normalization)

    # loop
    for iterat in range(max_iteration):
        colpos = iterat % tail

        # evaluate the single iteration
        x = fit[colpos].copy().to_numpy()
        y = com[colpos].copy().to_numpy()

        if typemat == 'sp':
            print('problem!!!! this code is not yet defined for sparse array')
#            com_here = bin_rca_t.dot(x)
            fit_here = bin_rca.dot(y)
        else:
            arr = bin_rca_t*x
            arr[arr == 0] = arr.max()
            com_here = arr.min(axis=1)
            fit_here = np.dot(bin_rca, y)

        fit_here[np.where(fit_here < min_distance)] = 0
        com_here[np.where(com_here < min_distance)] = 0

        # normalization
        newpos = (iterat + 1) % tail
        fit[newpos] = normalize(fit_here, normalization)
        com[newpos] = normalize(com_here, normalization)

        # check the time scale
        if check_stop == 'crossing time':
            if (iterat % 8 == 7) and (iterat > max_iteration // 10):
                minimum = minimum_cossing_time(fit, iterat, tail)
                if minimum + iterat + 1 > max_iteration:
                    break
        elif check_stop == 'distance':
            distance = np.abs(fit[newpos] - fit[colpos]).sum()
            if verbose:
                print(iterat, distance)
            if iterat > max_iteration // 10:
                if distance < min_distance:
                    break

    return fit[newpos], com[newpos]
