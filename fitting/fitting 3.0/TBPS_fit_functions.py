# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:12:45 2022

@author: Stefano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TBPS_pdfs1 import *
from TBPS_pdfs2 import *
from TBPS_pdf_full import *
from TBPS_bin_functions import *

from scipy import stats
from scipy.optimize import NonlinearConstraint, minimize
from scipy.optimize import brute, dual_annealing, shgo, differential_evolution
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
    fl, afb = coeff
    
    # If list of coeffs is NOT given
    if not hasattr(fl, "__len__") and not hasattr(afb, "__len__"):
        P = np.array(ctl_acc_PDF(ctl, coeff, acc_poly))
        #print(np.log(P))
        nLL = - np.sum(np.log(P), axis = 0)
        return nLL
     
    else:   
        fl = np.array(fl)
        afb = np.array(afb)
        #1D lists of cefficients given, to be manually meshed 
        if fl.ndim == 1 and afb.ndim == 1:
            P =[]
            for j, afbj in enumerate(afb):
                Pi = []
                for i, fli in enumerate(fl):
                    Pij = np.array(ctl_acc_PDF(ctl, [fli, afbj], acc_poly))
                    Pi.append(Pij)
                Pi = np.array(Pi)
                P.append(Pi)
            P = np.array(P)
            nLL = - np.sum(np.log(P), axis = 2)
        #Meshgrid of cefficients given 
        elif fl.ndim == 2 and afb.ndim == 2:
            P = ctl_acc_PDF(ctl, [fl,afb], acc_poly)
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
    P = ctk_PDF(costhetak = ctk, fl=fl)
    amin = np.amin(P)
    if amin <= 0 or np.isnan(amin):
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(P))
        return nLL
    

def nLL_phi (data, acceptance_phi, coeff, _bin):
    """
    Parameters
    ----------
    data : list
    acceptance_phi : list
    coeff : 2, ndarray
        s3, A_I_
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed phi distribution
    """
    _bin = int(_bin)
    phi = data[_bin]['phi']
    acc_poly = acceptance_phi[_bin]
    fl, afb = coeff  #ACTUALLY S3 AND AIM, BUT TOO LONG REWRITING EVERYTHING
    
    # If list of coeffs is NOT given
    if not hasattr(fl, "__len__") and not hasattr(afb, "__len__"):
        P = np.array(phi_acc_PDF(phi, coeff, acc_poly))
        #print(np.log(P))
        nLL = - np.sum(np.log(P), axis = 0)
        return nLL
     
    else:   
        fl = np.array(fl)
        afb = np.array(afb)
        #1D lists of cefficients given, to be manually meshed 
        if fl.ndim == 1 and afb.ndim == 1:
            P =[]
            for j, afbj in enumerate(afb):
                Pi = []
                for i, fli in enumerate(fl):
                    Pij = np.array(phi_acc_PDF(phi, [fli, afbj], acc_poly))
                    Pi.append(Pij)
                Pi = np.array(Pi)
                P.append(Pi)
            P = np.array(P)
            nLL = - np.sum(np.log(P), axis = 2)
        #Meshgrid of cefficients given 
        elif fl.ndim == 2 and afb.ndim == 2:
            P = phi_acc_PDF(phi, [fl,afb], acc_poly)
            nLL = - np.sum(np.log(P), axis = 0)
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

def fit_angs (data, f = ctl_acc_PDF, guess = [0,0], guess_per_bin = False, 
             limits = 0, limits_per_bin = False, Nlayers = 3, 
             args = None, xtrack = [(0,), (1,)], coeftrack = [(0,1),(0,)],
             fixtrack = 0, vary = 0, flexible = True, want_scipy = True):
    """
    Minimise the NLL associated to the cos(theta_l) and/or cos(theta_k) 
    distributions using Minuit class from iminuit, hence enjoying its 
    functionalities, with its scipy function, allowing to set constraints.
    
    Parameters
    ----------
    data : ndarray of DataFrames
        The binned data with columns "costhetal", "costhetak", "phi".
    f : callable
        The function(s) to be fitted, with choices being:
            - ctl_acc_PDF, i.e. the PDF associated to cos(theta_l) (default)
            - ctk_acc_PDF, i.e. the PDF associated to cos(theta_k)
            - phi_acc_PDF, i.e. the PDF associated to phi
         Can be a list with any of them, but IMPORTANT: in this order.\n
         Alternatively, it can be a function with name sj_acc_PDF, where j is
         a number. Such a functions must be given alone
    guess : list or tuple
        Initial guess for Fl and Afb. A different pair of guesses can be
        provided for each bin, then guess is a ndarray of shape (N, 2), where
        N is the number of bins.
    guess_per_bin : Bool
        If true, each i element of guess is a set of guesses for the ith bin.
    limits : ndarray
        List of 2-tuples with (min, max) for each parameter. Default 0 means 
        standard limits are used: Fl and Afb between -1 and 1.
    limits_per_bin : Bool
        If true, each i element of limits is a set of limits for the ith bin. 
    Nlayers : int
        Number of layers of minimisations. The total number of minimisations is
        4 * [1 + 3 + ... + 3 **(Nlayers- 1)]. The default is 3.\n
    args : ndarray
        If None, function takes no args. Otherwise  must be a list of
        args arg_i. Each arg_i being the one of the ith bin. 
        If ftofit is list of functions, then arg_i is a list too. If ftofit[i]
        takes no args, use args_i = None. 
        Default = float("nan").
    xtrack : ndarrays of tuples
        If ftofit is a ndarray, specify each function which variables they
        take by their index. Default is 0. If unspecified in the 
        case ftofit is a ndarray, returns error. \n
        Example: say you want to fit 3 function fa, fb, fc. Then ftofit is
        [fa, fb, fc]. The x variables are [xa, xb, xc, xd]. Say fa takes the 
        first two, fb takes the middle two, fc the last one. Then \n
                       xtrack = [(0,1), (1,2), (3)].\n
    coeftrack : ndarray of tuples
        If ftofit is a ndarray, specify each function which parameters they
        take by their index in myguess. Default is 0. If unspecified in the 
        case ftofit is a ndarray, returns error. \n
        Example: say you want to fit 3 functions fa, fb, fc. Then ftofit is
        [fa, fb, fc]. The parameters are [A, B, C, D, E]. Say fa takes the 
        first two, fb takes the middle three, fc the last one. Then \n
                    coeftrack = [(0,1), (1,2,3), (5)].\n
    fixtrack : ndarray
        Specifies the parameters of the function to be kept fixed, by their
        index. \n
        For example, say you have 3 parameters and fixtrack = [0,2]: then only
        the parameter indexed 1 will be fitted, the others will be kept 
        constant.\n
        Default is 0, meaning none is fixed.
    flexible : bool
        If flexible, use just one constraint if with 2 it gives an error (note,
        it will ALWAYS use just one constraint if minimisation with all cannot
        reach accurate result, hence NaN, but if flexible it will try with just 
        one also if it finds an error with all). Turn False for debugging 
        purposes. Deafult True. 
    want_scipy : list or 0
        If list, use an initial stochastic scipy minimisation. List must have
        at least one among [0(brute),1(dual_annealing),2(diff_evolution)]. If
        0, don't use any.
        
    Returns
    -------
    coeffs : ndarray
        ndarray with n sub-ndarray, one for each parameter, of length 
        len(data) (one value per bin)
    errors : tuple of 2 lists of floats:
        One list of errors of F_L for each bin.
        One list of errors of A_fb for each bin.
    fits : ndarray
        List of Minuit fit objects, one per bin
    """

    coefs = []
    errs = []
    fits = []
    
    try:
        if args == None:
            args = [None]*len(data)
    except KeyboardInterrupt:
        raise Exception("Keyboard Interrupt")
    except: #definitely not None
        pass
    
    # Not needed, as trivially satsified for all
    # con.append(NonlinearConstraint(norm_ctl, lb = lb, ub = ub))
    # con.append(NonlinearConstraint(norm_ctk, lb = lb, ub = ub))
           
    for i in tqdm( range (len(data)), "Fitting Bins" ):
        
        #Get guesses and limits right
        if guess_per_bin: 
            guessi = guess[i]
            if not limits:
                limitsi = [(-1, 1)]*len(guessi)
            else:
                limitsi = limits*1
        else: 
            guessi = guess
            if not limits:
                limitsi = [(-1, 1)]*len(guessi)
            else:
                limitsi = limits*1
        
        if limits_per_bin: limitsi = limits[i]
        only_con = True
        # print(limitsi)
        # print(guessi)
        
        # Tell the function WHAT you are fitting, and use the adequate
        # constraints
        if hasattr(f, "__len__"):
            if len(f)==3:
                a = data[i]["costhetal"]*1
                b = data[i]["costhetak"]*1
                c = data[i]["phi"]*1
                x = np.array([a,b,c])
                del a,b,c
                xtrack = [[0], [1], [2]] 
                coeftrack = [[0,1], [0], [2, 3]]
                #Define constraints
                con = NonlinearConstraint(posit_con_lphi,lb = 0,ub = np.inf)
                
            elif len(f)==2:
                x = [0,0]
                xtrack = [[0],[1]]
                #print("\n", "xtrack = ",xtrack)
                coeftrack = [0,0]
                if "ctl" in f[0].__name__:
                    x[0] = data[i]["costhetal"] *1
                    coeftrack[0] = [0,1]
                    if "ctk" in f[1].__name__:
                        x[1] = data[i]["costhetak"] *1
                        coeftrack[1] = [0]
                        #Define constraints
                        con =NonlinearConstraint(posit_con_l,lb = 0,ub = np.inf)
                    else:
                        x[1] = data[i]["phi"] *1
                        coeftrack[1] = [0, 2, 3]
                        #Define constraints
                        con =NonlinearConstraint(posit_con_lphi,lb = 0,ub = np.inf)
                else:
                    x = data[i][["costhetak","phi"]] *1
                    coeftrack = [[0], [0,1,2]]
                    #Define constraints
                    con =NonlinearConstraint(posit_con_phi,lb = 0,ub = np.inf)
            else: 
                print("Wrong number of functions")
        elif "ctl" in f.__name__:
            x = data[i]["costhetal"]
            xtrack = 0
            coeftrack = 0
            #Define constraints
            con = NonlinearConstraint(posit_con_l, lb = 0, ub = np.inf)
        elif "ctk" in f.__name__:
            x = data[i]["costhetak"]
            xtrack = 0
            coeftrack = 0
            #Define constraints
        elif "phi" in f.__name__:
            x = data[i]["phi"]
            xtrack = 0
            coeftrack = 0
            #Define constraints
            con = NonlinearConstraint(posit_con_phi, lb = 0, ub = np.inf)
        elif "s" in f.__name__ and "PDF" in f.__name__:
            a = data[i]["costhetal"]*1
            b = data[i]["costhetak"]*1
            c = data[i]["phi"]*1
            x = np.array([a,b,c])
            del a,b,c
            xtrack = 0
            coeftrack = 0
            if not hasattr(fixtrack, "__len__"):
                fl = guessi[0]
                afb = guessi[1]
                limitsi = [[fl-1e6, fl+1e6], [afb-1e-6, afb+1e6], [-1, 1], [-1, 1]]
                #fixtrack = [0,1]
            #Define constraints
            def posit_con(coeff):
                posit_f = "s"+f.__name__[1]+"_PDF"
                posit_f = globals()[posit_f]
                if len(coeff) == 2:
                    all_coeffs = np.concatenate((guessi[:2], coeff))
                else:
                    all_coeffs = coeff
                return posit_con_S(posit_f, all_coeffs)
            con = NonlinearConstraint(posit_con, lb = 0 ,ub = np.inf)
            only_con = False
        else:
            print("Something is wrong with function's name!")
        
        # Do the magic
        try:
            results = multiFit(f,x,float("nan"),guessi, limitsi, Nlayers,
                          "nll", args = args[i], xtrack = xtrack, 
                          coeftrack = coeftrack, fixtrack = fixtrack,vary = vary, 
                          constraints = con, only_constrained = only_con,
                          want_scipy = want_scipy)
        except KeyboardInterrupt:
            raise Exception("Keyboard Interrupt")
        except:
            print("\n An error ooccurred with following parameters:")
            print(f"guessi = {guessi}, \nlimitsi = {limitsi},\
                  \nargs[i] = {args[i]}, \nxtrack = {xtrack}, \
                      \ncoeftrack = {coeftrack}")
            results = multiFit(f,x,float("nan"),guessi, limitsi, Nlayers,
                          "nll", args = args[i], xtrack = xtrack, 
                          coeftrack = coeftrack, fixtrack = fixtrack,vary = vary, 
                          constraints = con, only_constrained = only_con,
                          want_scipy = want_scipy)

# ##############################################################################
#         return results#######################################################
########(for quick debugging)##################################################
        
        #Take results
        coefs.append(results[0]) 
        errs.append(results[1])
        fits.append(results[-1])
    
    final_coefs = coefs[0]
    final_errs = errs[0]
    for i in range(1, len(data)):
        final_coefs = np.column_stack([final_coefs, coefs[i]])
        final_errs = np.column_stack([final_errs, errs[i]])
                
    return final_coefs, final_errs, fits


#Define constraints
#Define constraining function: positive PDFs
# k is trivially positive for Fl in [0,1] -> limit of Fl
def posit_con_lphi (coeff):  
    fl, afb, s3, aim = coeff[0], coeff[1], coeff[2], coeff[3]
    ndarray_ctl = ctl_PDF(np.array([-1, -0.5, 0, 0.5,  1]), [fl, afb])
    ndarray_phi = phi_PDF(np.array([0, 0.5, 1, 1.5, 2])*np.pi, [s3, aim])
    return np.amin([np.amin(ndarray_ctl), np.amin(ndarray_phi)]) 

def posit_con_l (coeff):  
    ndarray_ctl = ctl_PDF(np.array([-1, -0.5, 0, 0.5,  1]), coeff)
    return np.amin(ndarray_ctl) 

def posit_con_phi (coeff):  
    s3, aim = coeff[0], coeff[1]
    ndarray_phi = phi_PDF(np.array([0, 0.5, 1, 1.5, 2])*np.pi, [s3, aim])
    return np.amin(ndarray_phi)

def posit_con_S (f, coeff):
    a = np.arange(-1, 1.5, 0.5)
    u = np.array([a -1, a-1, a* np.pi])
    pdf = f(u, coeff)
    return np.amin(pdf)



#%% The general minimiser acting behind the scenes

def multiFit (ftofit, x, y, myguess, mylimits, Nlayers = 3, statf = "NLL",
              args = None, usenone = False, 
              xtrack = 0, coeftrack = 0, fixtrack = 0, vary = 0, 
              constraints = 0,  only_constrained = True,
              give_statf = 0, want_scipy = True):
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
        Args of ftofit. If ftofit doesn't take args, use None keyword. \n
        If ftofit is a list of functions, then args must be a
        list of args_i for each ith function. If ith function doesn't take 
        arg, set args[i] = None. \n
    usenone : Bool
        If True use args if it is None. Otherwise, don't use args at all.
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
    fixtrack : ndarray
        Specifies the parameters of the function to be kept fixed, by their
        index. Must be an array, even if of length one. \n
        For example, say you have 3 parameters and fixtrack = [0,2]: then only
        the parameter indexed 1 will be fitted, the others will be kept 
        constant.\n
        Default is 0, meaning none is fixed.\n
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
    give_statf : Bool
        If True don't do minimisation but return statf(coeff)
    want_scipy : list or 0
        If list, use an initial stochastic scipy minimisation. List must have
        at least one among [0(brute),1(dual_annealing),2(diff_evolution)]. If
        0, don't use any.
        
    Returns
    -------
    coefs
        Best fit for parameters.
    stnds
        Statistical uncertainties on parameters.
    statf_result
        Value of the minimised statf.
    fit
        The whole Minuit minimisation object
    """
    
    ##########################################################################
    # STEP 0: PREPARE VARIABLES
    fit_coefs = []
    fit_stnds = []
    fit_statf = []
    fit_tag   = []
    fit_object= []
    
    if not mylimits:
        mylimits = [None for i in myguess]
    
    
    ##########################################################################
    # STEP 1 Define statistical function to be minimised
    try:
        if args == None and not usenone:
            mystatf = mystatf_noargs(ftofit, statf, x, y, 
                                             xtrack, coeftrack)
        else:
            mystatf = mystatf_yesargs(ftofit, statf, x, y, 
                                             xtrack, coeftrack, args)
    except ValueError: #if it fails definitely it is not None
        mystatf = mystatf_yesargs(ftofit, statf, x, y, 
                                             xtrack, coeftrack, args)

    if give_statf:
        return mystatf
    
    # For the scipy stochastic ones, use this, as they don't allow to keep
    # some fixed
    if hasattr(fixtrack, "__len__"):
        fixtrack = np.array(fixtrack)
        eff_fixtrack = [fixtrack[j]-j for j in range(len(fixtrack))]
        fixed_guesses = []
        vary_guesses = []
        vary_bounds = []
        for i in range(len(myguess)):
            if i in fixtrack:
                fixed_guesses.append(myguess[i])
            else:
                vary_guesses.append(myguess[i])
                vary_bounds.append(mylimits[i])
        fixed_guesses = np.array(fixed_guesses)
        vary_guesses = np.array(vary_guesses)
        vary_bounds = np.array(vary_bounds)
        
        def my_fixed_statf(vary_coeff):
            all_coeffs = np.insert(vary_coeff, eff_fixtrack, fixed_guesses)
            # print("vary_coeff", vary_coeff)
            # print("fixtrack", fixtrack)
            # print("fixed_guesses", fixed_guesses)
            # print("all_coeffs", all_coeffs)
            # raise Exception()
            return mystatf(all_coeffs)
    
    
    
    ##########################################################################
    #STEP 2: Define Minimising Function
    def minuitmin(guess, func = 0):
        #Set problem
        guess = np.array(guess)
        m = Minuit(mystatf, guess)
        m.limits = mylimits
        m.strategy = 2
        if hasattr(fixtrack, "__len__"):
            for fixi in fixtrack:
                m.fixed[fixi] = True
        
        # Minimise
        if   func == 0: m.simplex()
        elif func == 1: m.migrad()
        elif func == 2 and constraints: m.scipy(constraints = constraints)
        elif func == 2: m.scipy()
        elif func == 3: m.scan()
        else: print("If this was printed, there's something wrong!")
        
        # Find uncertainty
        m.hesse()
        try:
            if hasattr(m.errors, "__len__"):
                if np.isnan(m.errors[0]):
                    m.minos()
            elif np.isnan(m.errors):
                m.minos()
        except KeyboardInterrupt:
            raise Exception("Keyboard Interrupt")
        except:
            pass
        
        #Take results
        values , errors = [], []
        for i, (v, e) in enumerate(zip(m.values, m.errors)):
            values.append(v)
            errors.append(e)
        fit_coefs.append(values)
        fit_stnds.append(errors)
        fit_statf.append(m.fval)
        fit_object.append(m)
        fit_tag.append(func)
    
    
    def scipy_stoc_min(guess, func = 0):
        #Set problem
        guess = np.array(guess)
        bounds = mylimits
        
        # This Changes if success is achieved
        x, fval = (None, None)
        
        if hasattr(fixtrack, "__len__"):
            if func == 0:
                x = brute(my_fixed_statf, vary_bounds)
                fval = my_fixed_statf(x)
            elif func == 1:
                result = dual_annealing(my_fixed_statf, vary_bounds)
                if result.success:
                    x = result.x
                    fval = result.fun
            elif func == 2 and constraints:
                try:
                    result = differential_evolution(my_fixed_statf, vary_bounds, 
                                                    constraints = constraints)
                    if result.success:
                        x = result.x
                        fval = result.fun
                except ValueError:
                    return 0
            elif func == 2:
                try:
                    result = differential_evolution(my_fixed_statf, vary_bounds)
                    if result.success:
                        x = result.x
                        fval = result.fun
                except ValueError:
                    return 0
                
        elif not hasattr(fixtrack, "__len__"):
            if func == 0:
                x = brute(mystatf, bounds)
                fval = mystatf(x)
            elif func == 1:
                result = dual_annealing(mystatf, bounds)
                if result.success:
                    x = result.x
                    fval = result.fun
                if result.success:
                    x = result.x
                    fval = result.fun
        
            elif func == 2 and constraints:
                try:
                    result = differential_evolution(mystatf, bounds, 
                                                    constraints = constraints)
                    if result.success:
                        x = result.x
                        fval = result.fun
                except ValueError:
                    return 0
            elif func == 2:
                try:
                    result = differential_evolution(mystatf, bounds)
                    if result.success:
                        x = result.x
                        fval = result.fun
                except ValueError:
                    return 0

        if fval != None:
            if hasattr(fixtrack, "__len__"):
                x = np.insert(x, eff_fixtrack, fixed_guesses)
            else:
                x = x
                
            m = Minuit(mystatf, x)
            m.limits = mylimits
            if hasattr(fixtrack, "__len__"):
                for fixi in fixtrack:
                    m.fixed[fixi] = True
            
            # Find uncertainty
            m.hesse()
            try:
                if hasattr(m.errors, "__len__"):
                    if np.isnan(m.errors[0]):
                        m.minos()
                elif np.isnan(m.errors):
                    m.minos()
            except KeyboardInterrupt:
                raise Exception("Keyboard Interrupt")
            except:
                pass
    
            #Take results
            values , errors = [], []
            for i, (v, e) in enumerate(zip(x, m.errors)):
                values.append(v)
                errors.append(e)
            fit_coefs.append(values)
            fit_stnds.append(errors)
            fit_statf.append(fval)
            fit_object.append(m)
            fit_tag.append(10)
        
    
    
    ##########################################################################
    # STEP 3: Minimise and Save Solutions
    Lcount = 0
    if want_scipy:
        # First a round of stochastic scipy
        for method in want_scipy:
            scipy_stoc_min(myguess, func = method)
        Lcount += 1
        print("Stochastic-Scipy Layer Executed")
        minuit_list = [0,1,2]
    else:
        minuit_list = [0,1,2,3]
    
    #Then a round of minuit
    for method in minuit_list:
        minuitmin(myguess, func = method)
        Lcount += 1
    print("Initial Minuit Layer Executed")
    
    #Then go with the minimisation tree
    N = len(fit_coefs)
    for j in range(Nlayers-Lcount):
        layer_i = - N * 3**(j) #just take values from prev. layer
        fit_coefs_j = fit_coefs[layer_i:]
        fit_tag_j = fit_tag[layer_i:]
        for ind in range(len(fit_coefs_j)):
            guess, tag = fit_coefs_j[ind], fit_tag_j[ind]
            if not np.isnan(guess[0]):
                # If last layer, and constraints, only scipy's needed
                if j == Nlayers-count-1 and constraints and only_constrained:
                    minuitmin(guess, func =2)
                else:
                    for method in [x for x in [0,1,2] if x != tag]:
                        minuitmin(guess, func = method)
    
    
    
    ##########################################################################
    # STEP 4 : Pick Best Solution   
    # Only leave results from Minuit.scipy(), which use constraints
    if constraints and only_constrained:
        jrange = range(len(fit_coefs))
        fit_coefs = [fit_coefs[j] for j in jrange if fit_tag[j] == 2]
        fit_stnds = [fit_stnds[j] for j in jrange if fit_tag[j] == 2]
        fit_statf = [fit_statf[j] for j in jrange if fit_tag[j] == 2]
        fit_object = [fit_object[j] for j in jrange if fit_tag[j] == 2]
    
    # Check for Nan
    check_for_nan = []
    for fit in fit_statf:
        check_for_nan.append(np.isnan(fit))
    if not( False in check_for_nan):
        print("This Minimisation Returned Nan, I'm sorry")
        return [float("nan")] * len(myguess), [float("nan")] * len(myguess), float("nan")
    
    # Pick Best Result
    choice    = np.nanargmin(fit_statf)
    coefs     = fit_coefs[choice]
    stnds     = fit_stnds[choice] 
    statf_min = fit_statf[choice]
    fit       = fit_object[choice]
        
    # Possibly change result based on uncertainty and error
    if vary:
        for j, statf_j in enumerate(fit_statf):
            if statf_j - statf_min <= abs(vary):
                stndsj = fit_stnds[j]
                coefsj = fit_coefs[j]
                a = sum(np.array(stnds)**2  / np.array(coefs)**2 )
                b = sum(np.array(stndsj)**2 / np.array(coefsj)**2 )
                if (np.isnan(a) and not np.isnan(b)):
                   choice = j*1
                   stnds = stndsj * 1
                   coefs = coefsj * 1
                elif not np.isnan(a) and not np.isnan(b) and a > b:
                    choice = j*1
                    stnds = stndsj * 1
                    coefs = coefsj * 1
        statf_min = fit_statf[choice]
        fit       = fit_object[choice]
        
    # Ladies and Gentlemen, the end
    return coefs, stnds, statf_min, fit



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
                amin = np.amin(P)
                if amin < 0 or np.isnan(amin):
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
                    x_i     = [x[j] for j in xtrack[i]]
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
                    x_i     = [x[j] for j in xtrack[i]]
                    if len(coefs_i) == 1: coefs_i = coefs_i[0]
                    if len(x_i)     == 1: x_i = x_i[0]
                    #Compute NLL for ith function
                    P_i = ftofit_i(x_i, coefs_i)
                    amin = np.amin(P_i)
                    if amin < 0 or np.isnan(amin):
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
                try:
                    amin = np.amin(P)
                except KeyboardInterrupt:
                    raise Exception("Keyboard Interrupt")
                except:
                    print("x:", x)
                    print("coefs:", coefs)
                    print("args:", args)
                    print("P:", P)
                    raise Exception("Go and look at mystatf")
                if amin < 0 or np.isnan(amin):
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
                    x_i     = [x[j] for j in xtrack[i]]
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
                    #print("xtrack and xtracki = ",xtrack, xtrack[i])
                    x_i     = [x[j] for j in xtrack[i]]
                    if len(coefs_i) == 1: coefs_i = coefs_i[0]
                    if len(x_i)     == 1: x_i = x_i[0]
                    #Compute NLL for ith function
                    try:
                        if args[i] == None:
                            P_i = ftofit_i(x_i, coefs_i)
                        else:
                            P_i = ftofit_i(x_i, coefs_i, args[i]) 
                    except KeyboardInterrupt:
                        raise Exception("Keyboard Interrupt")
                    except: #definitely not None
                        P_i = ftofit_i(x_i, coefs_i, args[i])
                    amin = np.amin(P_i)
                    if amin < 0 or np.isnan(amin):
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



def get_FlAfb_guesses(bindata, acceptance_ctl, N = 100, plot = False):
    fl = np.linspace(0, 1, N); afb = np.linspace(-1, 1, N)
    X,Y = np.meshgrid(fl,afb)
    Z = []
    guess = []
    for i in tqdm(range(0, len(bindata)), "Getting Fl-Afb Guesses"):
        Z.append(nLL_ctl(bindata, acceptance_ctl, [X, Y], i))
        try:
            afbmin_i, flmin_i = np.unravel_index(np.nanargmin(Z[-1]), Z[-1].shape)
            flmin = fl[flmin_i]; afbmin = afb[afbmin_i]
        except KeyboardInterrupt:
            raise Exception("Keyboard Interrupt")
        except:
            print(f"Bin {i}: all nans")
            flmin = 0.5; afbmin = 0.0
        guess.append([flmin, afbmin])
    return guess
    
    if plot:
        print("OOOK, we are plotting NLL_ctl, but it may take time!")
        N = 200
        delta = 0.1
        for i in range(len(bindata)):
        
            fig, axes = plt.subplots(2,1, figsize = (7, 10), 
                                    gridspec_kw={'height_ratios': [1.3, 1]})
            plt.suptitle(r"NLL of $cos(\theta_l)$;"+f" Bin {i}: {mybins[i]} $GeV^2$")
            ax0 = axes[0]; ax1 = axes[1]
            
            cntr0 = ax0.contourf(X, Y, Z[i], 200, cmap = "nipy_spectral")
            ax0.set_xlabel(r"$F_l$"); ax0.set_ylabel(r"$A_fb$")
            ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
            
            plt.subplot(2,1,2)
            flmin = guess[i][0]; afbmin = guess[i][1]
            fl0 = max(0, flmin - delta);    fl1 = min(1, flmin + delta)
            afb0 = max(-1, afbmin - delta); afb1 = min(1, afbmin + delta)
            fl = np.linspace(fl0, fl1, N); afb = np.linspace(afb0, afb1, N)
            Xmin,Ymin = np.meshgrid(fl,afb)
            Zmin = nLL_ctl(bindata, acceptance_l, [X, Y], i)
            cntr1 = ax1.contourf(Xmin, Ymin, Zmin, 200, cmap = "nipy_spectral")
            ax1.set_xlabel(r"$F_l$"); ax1.set_ylabel(r"$A_fb$")
            ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")
        
            plt.subplots_adjust(hspace=0.2)
            fig.colorbar(cntr0, ax = ax0)
            fig.colorbar(cntr1, ax = ax1)
            #plt.savefig( f"MinPlots/NLL_ctl_bin{i}.pdf" )
            plt.show()
    
        del fig, axes, ax0, ax1, cntr0, cntr1
        del N, delta 
        del Xmin, Ymin, Zmin, fl0, fl1, afb0, afb1



             
