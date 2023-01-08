# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:12:45 2022

@author: Stefano
"""

import numpy as np
import pandas as pd

from TBPS_pdf_functions2 import *
from TBPS_bin_functions import *

from scipy import stats
from scipy.optimize import NonlinearConstraint
from iminuit import Minuit

from tqdm import tqdm
#%%NEGATIVE LOG LIKELIHOODS

def nLL_ctl (data, acceptance_l, coeff, _bin):
    """
    Parameters
    ----------
    coeff:
        Fl and Afb.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihood of the observed cos(theta_l) distribution
    """
    _bin = int(_bin)
    ctl = data[_bin]['costhetal']
    acc_poly = acceptance_l[_bin]
    
    P = np.array(ctl_acc_PDF(ctl, coeff, acc_poly))
    if P.ndim == 1:
        if np.amin(P) <= 0:
            return np.float("nan")
        else:
            nLL = - np.sum(np.log(P), axis = 0)
            return nLL
    else: 
        nLL = - np.sum(np.log(P), axis = 0)
        return nLL


def nLL_ctk (data, fl, _bin):
    """
    Parameters
    ----------
    fl : float
        The factor F_L in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed cos(theta_k) distribution
    """
    _bin = int(_bin)
    ctk = data[_bin]['costhetak']
    normalised_P = ctk_PDF(costhetak = ctk, fl=fl)
    if np.amin(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL
    


def nLL_phi (data, coeff, _bin):
    """
    Parameters
    ----------
    coeff: 3, ndarray
        Fl, A_t, A_I_,
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed phi distribution
    """
    _bin = int(_bin)
    phi = data[_bin]['phi']
    normalised_P = phi_PDF(phi, coeff)
    if np.amin(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL




#%%% Mixed Likelihoods
# Negative Log Likelihoods (NLL) for overall angular distributions. 

def nLL_cts (data, coeff, _bin):
    """
    Parameters
    ----------
    fl : float
        The factor F_L in the distribution.
    afb : float
        The factor A_fb in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed distribution of costheta_l and
        costheta_k
    """
    return nLL_ctl(data, coeff, _bin) + nLL_ctk(data, coeff[0], _bin)



def nLL_ang (data, coeff, _bin):
    """
    Parameters
    ----------
    coeff:
        Fl, Afb, A_t, A_I_m
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed angular distribution
    """
    nLL  = nLL_ctl(data, coeff[:1], _bin)
    nLL += nLL_ctk(data, coeff[0], _bin)
    nLL += nLL_phi(data, coeff[0, 2, 3], _bin)
    return nLL


#%% The Minimiser 

def fit_cts (data, f = ctl_acc_PDF, guess = [0,0], guess_per_bin = False, 
             limits = 0, limits_per_bin = False, Nlayers = 3, 
             args = float("nan"), xtrack = [(0), (1)], coeftrack = [(0,1),(0)],
             vary = 0, flexible = True):
    """
    Minimise the NLL associated to the cos(theta_l) and/or cos(theta_k) 
    distributions using Minuit class from iminuit, hence enjoying its 
    functionalities, with its scipy function, allowing to set constraints.
    
    Parameters
    ----------
    data : ndarray of DataFrames
        The binned data with columns "costhetal" and "costhetak".
    f : callable
        The function(s) to be fitted, with choices being:
            - ctl_acc_PDF, i.e. the PDF associated to cos(theta_l) (default)
            - ctk_acc_PDF, i.e. the PDF associated to cos(theta_k)
         Can be a list with both.
    guess : list or tuple
        Initial guess for Fl and Afb. A different pair of guesses can be
        provided for each bin, then guess is a ndarray of shape (N, 2), where
        N is the number of bins.
    guess_per_bin : Bool
        If true, each i element of guess is a set of guesses for the ith bin. 
    limits : ndarray
        List of 2-tuples with (min, max) for each parameter. Default 0 means 
        standard limits are used: Fl and Afb between -1 and 1.
    guess_per_bin : Bool
        If true, each i element of limits is a set of limits for the ith bin. 
    Nlayers : int
        Number of layers of minimisations. The total number of minimisations is
        4 * [1 + 3 + ... + 3 **(Nlayers- 1)]. The default is 3.\n
    args : ndarray
        If float("nan"), function takes no args. Otherwise  must be a list of
        args arg_i. Each arg_i being the one of the ith bin. 
        If ftofit is list of functions, then arg_i is a list too. If ftofit[i]
        takes no args, use None. 
        Default = float("nan").
    xtrack : ndarrays of tuples
        If ftofit is a ndarray, specify each function which variables they
        take by their index in myguess. Default is 0. If unspecified in the 
        case ftofit is a ndarray, returns error. \n
        Example: say you want to fit 3 function fa, fb, fc. Then ftofit is
        [fa, fb, fc]. The x variables are [xa, xb, xc, xd]. Say fa takes the 
        first two, fb takes the middle two, fc the last one. Then \n
                       xtrack = [(0,1), (1,2), (4)].\n
    coeftrack : ndarray of tuples
        If ftofit is a ndarray, specify each function which parameters they
        take by their index in myguess. Default is 0. If unspecified in the 
        case ftofit is a ndarray, returns error. \n
        Example: say you want to fit 3 function fa, fb, fc. Then ftofit is
        [fa, fb, fc]. The parameters are [A, B, C, D, E]. Say fa takes the 
        first two, fb takes the middle three, fc the last one. Then \n
                    coeftrack = [(0,1), (1,2,3), (5)].\n
    flexible : bool
        If flexible, use just one constraint if with 2 it gives an error (note,
        it will ALWAYS use just one constraint if minimisation with all cannot
        reach accurate result, hence NaN, but if flexible it will try with just 
        one also if it finds an error with all). Turn False for debugging 
        purposes. Deafult True. 
    
    Returns
    -------
    estimates : tuple of 2 lists of floats:
        One list of estimates of F_L for each bin.
        One list of estimates of A_fb for each bin.
    errors : tuple of 2 lists of floats:
        One list of errors of F_L for each bin.
        One list of errors of A_fb for each bin.
    """

    fls, fl_errs = [], []
    afbs, afb_errs = [], []
    
    #Define constraints
    con = [NonlinearConstraint(posit_con, lb = 0, ub = np.inf)]
    lb = 1 - 1e-10; ub = 1 + 1e-10
    con.append(NonlinearConstraint(norm_ctl, lb = lb, ub = ub))
    con.append(NonlinearConstraint(norm_ctk, lb = lb, ub = ub))
    
    if not limits:
        limits = [(-1, 1), (-1, 1)]
           
    
    for i in tqdm( range (len(data)), "Bins" ):
        
        #Get guesses and limits right
        if guess_per_bin: guessi = guess[i]
        else: guessi = guess
        if limits_per_bin: limitsi = limits[i]
        else: limitsi = limits
        
        # Tell the function WHAT you are fitting
        if hasattr(f, "__len__"):
            if len(f)==2:
                x = data[i][["costhetal"], ["costhetak"]]
            else: 
                print("Wrong number of functions")
        elif f.__name__[:3] == "ctl":
            x = data[i]["costhetal"]
            xtrack = 0
            coeftrack = 0
        elif f.__name__[:3] == "ctk":
            x = data[i]["costhetal"]
            xtrack = 0
            coeftrack = 0
        else:
            print("Something is wrong with function's name!")
        
        # Do the magic
        if not flexible:
            results = multiFit(f,x,float("nan"),guessi, limitsi, Nlayers,
                          "nll", args = args[i], xtrack = xtrack, 
                          coeftrack = coeftrack, vary = vary, 
                          constraints = con[0], only_constrained = True)
        else:
            try:
                results = multiFit(f,x,float("nan"),guessi, limitsi, Nlayers,
                          "nll", args = args[i], xtrack = xtrack, 
                          coeftrack = coeftrack, vary = vary, 
                          constraints = con[0], only_constrained = True)
            except:
                #Apply only positiveness constraint
                print("\nOnly positiveness constraint applied, due to ERROR")
                print("in minimisation with list of constraints.")
                print("For more info, set flexible = False.")
                results = multiFit(f,x,float("nan"),guessi, limitsi, Nlayers,
                          "nll", args = args[i], xtrack = xtrack, 
                          coeftrack = coeftrack, vary = vary, 
                          constraints = con[0], only_constrained = True)
        
        #Take results
        fl,  fl_err  = results[0][0], results[1][0] 
        afb, afb_err = results[0][1], results[1][1]
        
        fls.append(fl);      afbs.append(afb)
        fl_errs.append(fl_err);  afb_errs.append(afb_err)
            
    return (fls, afbs), (fl_errs, afb_errs)


#Define constraints
#Define constraining function: positive PDFs
def posit_con (coeff):  
    fl, afb = coeff[0], coeff[1]
    ndarray_ctl = ctl_PDF(np.array([-1, -0.5, 0, 0.5,  1]), coeff)
    ndarray_ctk = ctk_PDF(np.array([-1, -0.5, 0, 0.5, 1]), fl)
    return min([min(ndarray_ctl), min(ndarray_ctk)]) 

#Define cnonstraint function: normalised PDF
def norm_ctl (coeff): 
    fl, afb = coeff[0], coeff[1]
    Actl = quad(ctl_PDF, -1, 1, args = coeff)
    return Actl
def norm_ctk (coeff): 
    fl, afb = coeff[0], coeff[1]
    Actk = quad(ctk_PDF, -1, 1, args = (fl))
    return Actk



#%% The general minimiser acting behind the scenes

def multiFit (ftofit, x, y, myguess, mylimits, Nlayers = 3, statf = "NLL",
              args = float("nan"), usenan = False, 
              xtrack = 0, coeftrack = 0, vary = 0, 
              constraints = 0,  only_constrained = True):
    """
    Perform a tree of series of minimisations with iminuit.Minuit functions
    simplex, migrad and scipy, and return best result. 
    
    Parameters
    ----------
    ftofit : callable or ndarray of callables
        If callable: the function to be fitted. \n
        If ndarray: list of functions to be minimised together. 
    x : ndarray
        Indipendent variable of ftofit. \n
        If ndarray of ndarrays: indipendent variables of the functions.
    y : ndarray
        Measured output variable of ftofit, which will represent the data for 
        either the Chi squared or Negative Log Likelihhod to be mnimised.\n
        If ndarray of ndarrays: output of the functions to fit. Useless if
        statf is nll.
    myguess : ndarray
        Initial guess of parameters.\n
    mylimits : ndarray (n,2)
        Limits on parameters values. Must be an array of N tuples (N being the 
        number of parameters to fit). If there are no bounds on the parameter
        i, the ith tuple should be (-np.inf, np.inf), or, even better, must
        be substituted with keywrd None. Default 0 means no limits.\n
    Nlayers : int
        Number of layers of minimisations. The total number of minimisations is
        4 * [1 + 3 + ... + 3 **(Nlayers- 1)]. The default is 3.\  
    statf : str or int
        Defines the function to be minimised: either Chi squared ("chi2, 0) or
        Negative Log Likelihood ("nll" or 1). The default is "NLL".\n
        If many functions are fitted together:
            - chi2 is sum of individual chi2_i
            - NLL is sum of individual NLL_i
    args : float, ndarray
        IF FUNCTION TAKES EXTRA ARGUMENT, YOU MUST SPECIFY THEM, ALSO IF 
        DEFAULT ONES ARE WANTED. \n
        Args of ftofit. If ftofit doesn't take args, use float("nan"). \n
        If ftofit is a list of functions, then args must be a
        list of args_i for each ith function. If ith function doesn't take 
        arg, set args[i] = None. \n
    usenan : Bool
        If True use args if it is 'nan'. Otherwise, don't use args at all.
    xtrack : ndarrays of tuples
        If ftofit is a ndarray, specify each function which variables they
        take by their index in myguess. Default is 0. If unspecified in the 
        case ftofit is a ndarray, returns error. \n
        Example: say you want to fit 3 function fa, fb, fc. Then ftofit is
        [fa, fb, fc]. The x variables are [xa, xb, xc, xd]. Say fa takes the 
        first two, fb takes the middle two, fc the last one. Then \n
                       xtrack = [(0,1), (1,2), (4)].\n
    coeftrack : ndarray of tuples
        If ftofit is a ndarray, specify each function which parameters they
        take by their index in myguess. Default is 0. If unspecified in the 
        case ftofit is a ndarray, returns error. \n
        Example: say you want to fit 3 function fa, fb, fc. Then ftofit is
        [fa, fb, fc]. The parameters are [A, B, C, D, E]. Say fa takes the 
        first two, fb takes the middle three, fc the last one. Then \n
                    coeftrack = [(0,1), (1,2,3), (5)].\n
    vary : float
        If 0, the given result is the one with lower statf. If vary = v,
        the result is the one with statf in [min, min+abs(v)] such that the 
        sum of the scaled uncertainties squared is minimum. Default is 0. \n
    constraints : Scipy.NonlinearConstraint object
        If given, use constraints with iminuit.Minuit.scipy.
        Default is 0. \n
    only_constrained : Bool
        If constraints were applied and only_constraints = True, then  return
        ONLY value from scipy minimisation, which indeed used constraints. 
        If False, return whatever leads to lower Chi2. 
        
    Returns
    -------
    coefs
        Best fit for parameters.
    stnds
        Statistical uncertainties on parameters.
    statf_result
        Value of the minimised statf.
    """
    
    ##########################################################################
    # STEP 0: PREPARE VARIABLES
    fit_coefs = []
    fit_stnds = []
    fit_statf = []
    fit_tag   = []
    
    if not mylimits:
        mylimits = [None for i in myguess]
    
    
    ##########################################################################
    # STEP 1 Define statistical function to be minimised
    try:
        if np.isnan(args) and not usenan:
            mystatf = mystatf_noargs(ftofit, statf, x, y, 
                                             xtrack, coeftrack)
        else:
            mystatf = mystatf_yesargs(ftofit, statf, x, y, 
                                             xtrack, coeftrack, args)
    except ValueError: #if it fails definitely it is not nan
        mystatf = mystatf_yesargs(ftofit, statf, x, y, 
                                             xtrack, coeftrack, args)
    
    
    ##########################################################################
    #STEP 2: Define Minimising Function
    def minuitmin(guess, func = 0):
        #set problem
        guess = np.array(guess)
        m = Minuit(mystatf, guess)
        m.limits = mylimits
        m.strategy = 2
        
        # minimise
        if func == 0: m.simplex()
        elif func == 1: m.migrad()
        elif func == 2 and constraints: m.scipy(constraints = constraints)
        elif func == 2: m.scipy()
        elif func == 3: m.scan()
        else: print("If this was printed, there's something wrong!")
        
        # find uncertainty
        m.hesse()
        try:
            if hasattr(m.errors, "__len__"):
                if np.isnan(m.errors[0]):
                    m.minos()
            elif np.isnan(m.errors):
                m.minos()
        except:
            pass
        
        #take results
        values , errors = [], []
        for i, (v, e) in enumerate(zip(m.values, m.errors)):
            values.append(v)
            errors.append(e)
        fit_coefs.append(values)
        fit_stnds.append(errors)
        fit_statf.append(m.fval)
        fit_tag.append(func)
    
    
    ##########################################################################
    # STEP 3: Minimise and Save Solutions
    Nlayers  = 3
    for j in range(Nlayers-1):
        if j == 0:
            for method in [0,1,2,3]:
                minuitmin(myguess, func = method)
        else:
            layer_i = - 4 * 3**j #just take values from prev. layer
            fit_coefs_j = fit_coefs[layer_i:]
            fit_tag_j = fit_tag[layer_i:]
            for ind in range(len(fit_coefs_j)):
                guess, tag = fit_coefs_j[ind], fit_tag_j[ind]
                if not np.isnan(guess[0]):
                    for method in [x for x in [0,1,2] if x != tag]:
                        minuitmin(guess, func = method)
                        minuitmin(guess, func = 3) #do scan again for all
    
    
    ##########################################################################
    # STEP 4 : Pick Best Solution   
    # Only leave results from Minuit.scipy(), which use constraints
    if constraints and only_constrained:
        jrange = range(len(fit_coefs))
        fit_coefs = [fit_coefs[j] for j in jrange if fit_tag[j] == 2]
        fit_stnds = [fit_stnds[j] for j in jrange if fit_tag[j] == 2]
        fit_statf = [fit_statf[j] for j in jrange if fit_tag[j] == 2]
    
    # Check for Nan
    check_for_nan = []
    for fit in fit_statf:
        check_for_nan.append(np.isnan(fit))
    if not( False in check_for_nan):
        return [float("nan")] * len(myguess), [float("nan")] * len(myguess), float("nan")
    
    # Pick Best Result
    choice    = np.nanargmin(fit_statf)
    coefs     = fit_coefs[choice]
    stnds     = fit_stnds[choice] 
    statf_min = fit_statf[choice]
        
    # Possibly change result based on uncertainty and error
    if vary:
        for j, statf_j in enumerate(fit_statf):
            if statf_j - statf_min <= abs(vary):
                stndsj = fit_stnds[j]
                coefsj = fit_coefs[j]
                a = sum(np.array(stnds)**2  / np.array(coefs)**2 )
                b = sum(np.array(stndsj)**2 / np.array(coefsj)**2 )
                if a > b:
                   choice = j*1
                   stnds = stndsj * 1
                   coefs = coefsj * 1
        statf_min = fit_statf[choice]
        
    # Ladies and Gentlemen, the end
    return coefs, stnds, statf_min



def mystatf_noargs(ftofit, statf, x, y, xtrack, coeftrack):
    # If just one function
    if not hasattr(ftofit, "__len__"):
        if statf == ("chi2" or "Chi2" or "chisquare" or "Chisquare" or 0):
            def mystatf (coefs): #Chi2 Test
                expected = ftofit(x, coefs)
                ddof = len(coefs)
                chi2 = stats.chisquare(y, expected, ddof = ddof)[0]
                if chi2 > 0:
                    return chi2
                else:
                    return float("nan")
            mystatf.errordef = Minuit.LEAST_SQUARES
        elif statf == ("nll" or "NLL" or "nLL" or "likelihood" or 1):
            def mystatf (coefs): #Negative Log Likelihod
                P = ftofit(x, coefs)
                if min(P) <= 0:
                    return np.float("nan")
                else:
                    nLL = - np.sum(np.log(P))
                    return nLL
            mystatf.errordef = Minuit.LIKELIHOOD
        else:
            raise Exception ("statf doesn't have an accepted value, i.e.:\n\
                             0 or 'chi2' for Chi2, \n\
                             1 or 'nll'  for Negative Log Likelihood.")
    # If more functions
    elif hasattr(ftofit, "__len__"):
        if statf == ("chi2" or "Chi2" or "chisquare" or "Chisquare" or 0):
            def mystatf (coefs): #Chi2 Test
                chi2 = 0
                for i, ftofit_i in enumerate(ftofit):
                    #Take right coeffcients for ith function
                    coefs_i = [coefs[j] for j in coeftrack[i]]
                    x_i     = [x[j] for j in xtrack[j]]
                    if len(coefs_i) == 1: coefs_i = coefs_i[0]
                    if len(x_i)     == 1: x_i = x_i[0]
                    #Compute Chi2 for ith function
                    ddof_i = len(coefs_i)
                    expected_i = ftofit_i(x_i, coefs_i)
                    chi2_i=stats.chisquare(y[i], expected_i, ddof = ddof_i)[0]
                    if chi2_i < 0:
                        return float("nan")
                    else:
                        chi2 += chi2_i
                return chi2
            mystatf.errordef = Minuit.LEAST_SQUARES
        elif statf == ("nll" or "NLL" or "nLL" or "likelihood" or 1):
            def mystatf (coefs): #Negative Log Likelihod
                NLL = 0
                for i, ftofit_i in enumerate(ftofit):
                    #Take right coeffcients and xs for ith function
                    coefs_i = [coefs[j] for j in coeftrack[i]]
                    x_i     = [x[j] for j in xtrack[j]]
                    if len(coefs_i) == 1: coefs_i = coefs_i[0]
                    if len(x_i)     == 1: x_i = x_i[0]
                    #Compute NLL for ith function
                    P_i = ftofit_i(x_i, coefs_i)
                    if min(P_i) < 0:
                        return float("nan")
                    else:
                        NLL_i = - np.sum(np.log(P_i))
                    NLL += NLL_i
                return NLL
            mystatf.errordef = Minuit.LIKELIHOOD
        else:
            raise Exception ("statf doesn't have an accepted value, i.e.:\n\
                             0 or 'chi2' for Chi2, \n\
                             1 or 'nll'  for Negative Log Likelihood.")
    return mystatf




def mystatf_yesargs(ftofit, statf, x, y, xtrack, coeftrack, args):
    # If just one function
    if not hasattr(ftofit, "__len__"):
        if statf == ("chi2" or "Chi2" or "chisquare" or "Chisquare" or 0):
            def mystatf (coefs): #Chi2 Test
                expected = ftofit(x, coefs, args)
                ddof = len(coefs)
                chi2 = stats.chisquare(y, expected, ddof = ddof)[0]
                if chi2 > 0:
                    return chi2
                else:
                    return float("nan")
            mystatf.errordef = Minuit.LEAST_SQUARES
        elif statf == ("nll" or "NLL" or "nLL" or "likelihood" or 1):
            def mystatf (coefs): #Negative Log Likelihod
                P = ftofit(x, coefs, args)
                if min(P) <= 0:
                    return np.float("nan")
                else:
                    nLL = - np.sum(np.log(P))
                    return nLL
            mystatf.errordef = Minuit.LIKELIHOOD
        else:
            raise Exception ("statf doesn't have an accepted value, i.e.:\n\
                             0 or 'chi2' for Chi2, \n\
                             1 or 'nll'  for Negative Log Likelihood.")
    # If more functions
    elif hasattr(ftofit, "__len__"):
        if statf == ("chi2" or "Chi2" or "chisquare" or "Chisquare" or 0):
            def mystatf (coefs): #Chi2 Test
                chi2 = 0
                for i, ftofit_i in enumerate(ftofit):
                    #Take right coeffcients for ith function
                    coefs_i = [coefs[j] for j in coeftrack[i]]
                    x_i     = [x[j] for j in xtrack[j]]
                    if len(coefs_i) == 1: coefs_i = coefs_i[0]
                    if len(x_i)     == 1: x_i = x_i[0]
                    #Compute Chi2 for ith function
                    ddof_i = len(coefs_i)
                    if args[i] != None:
                        expected_i = ftofit_i(x_i, coefs_i, args[i])
                    else:
                        expected_i = ftofit_i(x_i, coefs_i)
                    chi2_i=stats.chisquare(y[i], expected_i, ddof = ddof_i)[0]
                    if chi2_i < 0:
                        return float("nan")
                    else:
                        chi2 += chi2_i
                return chi2
            mystatf.errordef = Minuit.LEAST_SQUARES
        elif statf == ("nll" or "NLL" or "nLL" or "likelihood" or 1):
            def mystatf (coefs): #Negative Log Likelihod
                NLL = 0
                for i, ftofit_i in enumerate(ftofit):
                    #Take right coeffcients and xs for ith function
                    coefs_i = [coefs[j] for j in coeftrack[i]]
                    x_i     = [x[j] for j in xtrack[j]]
                    if len(coefs_i) == 1: coefs_i = coefs_i[0]
                    if len(x_i)     == 1: x_i = x_i[0]
                    #Compute NLL for ith function
                    if args[i] != None:
                        P_i = ftofit_i(x_i, coefs_i, args[i])
                    else:
                        P_i = ftofit_i(x_i, coefs_i)
                    if min(P_i) < 0:
                        return float("nan")
                    else:
                        NLL_i = - np.sum(np.log(P_i))
                    NLL += NLL_i
                return NLL
            mystatf.errordef = Minuit.LIKELIHOOD
        else:
            raise Exception ("statf doesn't have an accepted value, i.e.:\n\
                             0 or 'chi2' for Chi2, \n\
                             1 or 'nll'  for Negative Log Likelihood.")
    return mystatf
             
