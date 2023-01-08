# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:40:16 2022

@author: Stefano
"""

import numpy as np
import pandas as pd  

   
def q2bin (mydata, bins):
    """
    Parameters
    ----------
    mydata : Pandas' Dataframe.
        A dataframe from pandas, with 'q2' as one of its columns.
    bins: list of tuples
        The extremi of the bins for the q2 values. 

    Returns
    -------
    data_binned : list of Pandas DataFrames
        For j = 0, 1, ..., len(bins) - 1, each entry corresponds to the jth bin
        of q2 values in bins, being the associated portion of the data
        DataFrame.
    """
    # Prepare binning of q2
    q2_list = mydata['q2']
    N = len(bins)
    data_binned = []
    for j in range(N):
        data_binned.append([])
    
    for i in range(len(q2_list)):
        q2 = q2_list[i]
        for j in range(N):
            binj = bins[j]
            if binj[0] <= q2 <= binj[1]:
                data_binned[j].append(mydata.iloc[i])

    for j in range(N):
        data_binned[j] = pd.DataFrame(data_binned[j])
    
    return data_binned 



def q2bin2 (data, x, N = 20, giveedges = 0, coolbins = 0):
    """
    Parameters
    ----------
    data : Pandas' Dataframe.
        A dataframe from pandas, with 'q2' as one of its columns.
    x : str, list of str
        Other column(s) of the Dataframe.
    N : int
        The number of bins for q2. Default is 20.
    giveedges: array-like
        If not 0, use giveedges as edges of bins fr q2. Default is 0.
    coolbins: float.
        If different than 0, say equal to a float c, then the edges of the
        binned distribution of q2 will be [q2min, n*c, (n+1)*c, ..., 
        (n + m) * c, q2max]. Deafult is 0.

    Returns
    -------
    q2_edges : list
        List of N+1 edges of the binned distribution of q2.
    x_binned : dict
        If x is just a string, it is a dictionary, with keys "bini" as i = 
        0,1,...,N. Each key is associated with a list of x values
        corresponding to ith bin of q2.
        
        If x is a list of strings: a dictionary with keys given by the elements
        x[n] of x. Each key corresponds to a subdict with subkeys "bini" 
        as i = 0,1,...,N. Each subkey is associated with a list of x[n] values
        corresponding to ith bin of q2.
    """
    # Analyse and prepare binning of q2
    q2_list = data['q2']
    if giveedges:
        q2_edges = giveedges
    else:
        N = int(N)
        q2min = min(q2_list); q2max = max(q2_list); span = q2max - q2min
        q2width = span / N
        q2_edges = [q2min]
        if not coolbins:
            for i in range(1, N+1):
                q2_edges.append(q2min + i * q2width)
        else:
            c = coolbins
            n = np.ceil(q2min / c)
            m = np.floor(q2max / c)
            r = np.arange(n * c, (m + 1) * c, c)
            for i in r:
                q2_edges.append(i)
            q2_edges.append(q2max)
    
    # Extract and order lists of values for the quantity(s) in x
    if isinstance(x, str): #not list, just one
        # list of values of x ordered for increasing values of q2
        x_list = [x for _, x in sorted(zip(q2_list, data[x]))]
        # sort q2 too
        q2_list = sorted(q2_list)
    else:
        x_list = []
        for xk in x:
            xk_list = [x for _, x in sorted(zip(q2_list, data[xk]))]
            #ith sublist of xlist has ordered values of x[i]
            x_list.append(xk_list) 
        # sort q2 too
        q2_list = sorted(q2_list)
            
    # Rearrange the x valuesbased on the q2 bins
    x_binned = {} #dict with a list of x values for each q2 bin
    j = 0; threshold = q2_edges[1]
    iprev = 0
    if isinstance(x, str): #not list, just one
        for i in range(len(q2_list)):
            if q2_list[i] >= threshold:
                # append sublist
                x_binned[f"bin{j}"] = x_list[iprev:i] 
                # update 
                try: #it will fail at very last value of i
                    j += 1; threshold = q2_edges[j + 1]
                    iprev = i * 1
                except:
                    if giveedges: break
                    else: pass
                        
    else:
        for xk in x:
            #one dict per x quantity, each with one list for q2 bin
            x_binned[xk] = {}
        for i in range(len(q2_list)):
            if q2_list[i] >= threshold:
                # append sublist
                for k in range(len(x)):
                    xk = x[k]
                    x_binned[xk][f"bin{j}"] = x_list[k][iprev:i]#append sublist
                # update 
                try: #it will fail at very last value of i
                    j += 1; threshold = q2_edges[j + 1]
                    iprev = i * 1
                except:
                    if giveedges: break
                    else: pass
    
    return q2_edges, x_binned