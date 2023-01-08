# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:01:16 2022

@author: Stefano
"""
from TBPS_bin_functions import *
from TBPS_pdfs import *
import pickle

import numpy as np
import pandas as pd

from iminuit import Minuit
from scipy.optimize import minimize, basinhopping, NonlinearConstraint
from scipy.integrate import quad, quadrature, romberg

from tqdm import tqdm


#DEFINE data and acceptance for following functions
data = pd.read_pickle('../data/signal.pkl').head(50000)

# Create acceptance_l: a list of np.polynomial, one for each bin. 
with open("../data/acceptance_function_coefs.pkl", "rb") as f:
    acceptance_coefs = pickle.load(f)
acceptance_l = []
for coeffsi in acceptance_coefs:
    acceptance_l.append(np.poly1d(coeffsi))

mybins = [(0.1, 0.98),
          (1.1, 2.5),
          (2.5, 4.0),
          (4.0, 6.0),
          (6.0, 8.0),
          (15.0, 17.0),
          (17.0, 19.0),
          (11.0, 12.5),
          (1.0, 6.0),
          (15.0, 17.9)] 
data = q2binbin(data, mybins)

#%%NEGATIVE LOG LIKELIHOODS

def nLL_ctl (fl, afb, _bin):
    """
    Parameters
    ----------
    fl : float
        The factor F_L in the distribution.
    afb : float
        The factor A_FB in the distribution.
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
    
    normalised_P = ctl_acc_PDF(ctl, fl, afb, acc_poly)
    if min(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL


def nLL_ctk (fl, afb, _bin):
    """
    Parameters
    ----------
    fl : float
        The factor F_L in the distribution.
    afb : float
        The factor A_fb in the distribution. It doesn't contribute, at all, to
        ctk_PDF, but is needed as an argument to let nLL_ctk and other nLLs
        have same number of arguments.
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
    if min(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL
    


def nLL_phi (fl, at, aim, _bin):
    """
    Parameters
    ----------
    fl : float
        The factor F_L in the distribution.
    at : float
        The factor A_t in the distribution.
    aim : float
        The factor A_I_m in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed phi distribution
    """
    _bin = int(_bin)
    phi = data[_bin]['phi']
    normalised_P = phi_PDF(phi, fl, at, aim)
    if min(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL




#%%% Mixed Likelihoods
# Negative Log Likelihoods (NLL) for overall angular distributions. 

def nLL_cts (fl, afb, _bin):
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
    return nLL_ctl(fl, afb, _bin) + nLL_ctk(fl, afb, _bin)


def nLL_cts2 (fl, afb, _bin):
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
    return nLL_ctl(fl, afb, _bin) + nLL_ctk2(fl, afb, _bin)


def nLL_ang (fl, afb, at, aIm, _bin):
    """
    Parameters
    ----------
    fl : float
        The factor F_L in the distribution.
    afb : float
        The factor A_fb in the distribution.
    at : float
        The factor A_t in the distribution.
    aim : float
        The factor A_I_m in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed angular distribution
    """
    return nLL_ctl(fl, afb, _bin)+nLL_ctk(fl, afb, _bin)+nLL_phi(fl, at, aIm, _bin)


#%% FIT_COSTHETAL AND COSTHETAK functions
def fit_cts (f = nLL_ctl, guess = [0,0], bin_number_to_check = 5, 
             flexible = True):
    """
    Minimise the NLL associated to the cos(theta_l) and/or cos(theta_k) 
    distributions using Minuit class from iminuit, hence enjoying its 
    functionalities, with its scipy function, allowing to set constraints.
    
    Parameters
    ----------
    f: callable
        The function to be minimised, with choices being:
            - nLL_ctl, i.e. the NLL associated to cos(theta_l) (default)
            - nLL_ctk, i.e. the NLL associated to cos(theta_k)
            - nLL_cts, i.e. the NLL associated to both
    guess : list or tuple
        Initial guess for Fl and Afb. A different pair of guesses can be
        provided for each bin, then guess is a ndarray of shape (N, 2), where
        N is the number of bins.
    bin_number_to_check : int, optional
        If an int between 0 and len(data), then return bin_results_to_check
        for further analysis. The default is 5.
    flexible: bool
        If flexible, use just one constraint if with 2 it gives an error (note,
        it will ALWAYS use just one constraint if minimisation with all cannot
        reach accurate result, hence NaN, but if flexible it will try with just 
        one also if it finds an error with all). Turn False for debugging 
        purposes. Deafult True. 

    Returns
    -------
    estimates: tuple of 2 lists of floats:
        One list of estimates of F_L for each bin.
        One list of estimates of A_fb for each bin.
    errors: tuple of 2 lists of floats:
        One list of errors of F_L for each bin.
        One list of errors of A_fb for each bin.
    bin_results_to_check : minuit estimate
        Return iff bin_number_to_check is an int between 0 and len(data) for 
        further anlysis.
    """
    f.errordef = Minuit.LIKELIHOOD
    decimal_places = 4
    fls, fl_errs = [], []
    afbs, afb_errs = [], []
    
    #Define constraining function: positive PDFs
    def posit_con (fl, afb, i): #needs same args as nLL
        i = int(i)
        ndarray_ctl = ctl_PDF(np.array([-1, -0.5, 0, 0.5,  1]), fl, afb)
        ndarray_ctk = ctk_PDF(np.array([-1, -0.5, 0, 0.5, 1]), fl)
        return min([min(ndarray_ctl), min(ndarray_ctk)]) 
    con = [NonlinearConstraint(posit_con, lb = 0, ub = np.inf)]
    
    #Define cnonstraint function: normalised PDF
    def norm_ctl (fl, afb, i): #needs same args as nLL
        i = int(i)
        Actl = quad(ctl_PDF, -1, 1, args = (fl, afb))
        return Actl
    def norm_ctk (fl, afb, i): #needs same args as nLL
        i = int(i)
        Actk = quad(ctk_PDF, -1, 1, args = (fl))
        return Actk
    lb = 1 - 1e-10; ub = 1 + 1e-10
    con.append(NonlinearConstraint(norm_ctl, lb = lb, ub = ub))
    con.append(NonlinearConstraint(norm_ctk, lb = lb, ub = ub))
    
    
    bin_results_to_check = 0             
    if len(guess) == len(data):
        for i in tqdm( range (len(data)), "Bins" ):
           
            m = Minuit(f, 
                       fl = guess[i][0], 
                       afb = guess[i][1], 
                       _bin = i)
            
            # First run a simplex minimisation
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            m.simplex()
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
                fl_err = m.errors[0]
                afb_err = m.errors[1]
            else:
                fl = guess[i][0]
                afb = guess[i][1]
            
            #Then a scipy-constrained minimisation from simplex's solution
            m = Minuit(f, fl = fl, afb = afb, _bin = i)
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            #con = NonlinearConstraint(posit_con, lb = 0, ub = np.inf)
            if not flexible:
                m.scipy(constraints = con)
            else:
                try:
                    m.scipy(constraints = con)
                except:
                    #Apply only positiveness constraint
                    print("\nOnly positiveness constraint applied, due to ERROR")
                    print("in minimisation with list of constraints.")
                    print("For more info, set flexible = False.")
                    m = Minuit(f, fl = fl, afb = afb, _bin = i)
                    m.fixed['_bin'] = True
                    m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
                    #con = NonlinearConstraint(posit_con, lb = 0, ub = np.inf)
                    m.scipy(constraints = con[0])
                    m.hesse()
                
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
                fl_err = m.errors[0]
                afb_err = m.errors[1]
            else:
                #Apply only positiveness constraint
                print("\nOnly positiveness constraint applied, due to failure")
                print("in minimisation with all constraints.")
                m = Minuit(f, fl = fl, afb = afb, _bin = i)
                m.fixed['_bin'] = True
                m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
                #con = NonlinearConstraint(posit_con, lb = 0, ub = np.inf)
                m.scipy(constraints = con[0])
                m.hesse()
            
            
            if i == bin_number_to_check:
                bin_results_to_check = m
                
            fls.append(fl);      afbs.append(afb)
            fl_errs.append(m.errors[0]);  afb_errs.append(m.errors[1])
            
    else:
        for i in tqdm( range (len(data)), "Bins" ):
            
            m = Minuit(f, fl = guess[0], afb = guess[1], _bin = i)
            
            # First run a simplex minimisation
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            m.simplex()
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
                fl_err = m.errors[0]
                afb_err = m.errors[1]
            else:
                fl = guess[0]
                afb = guess[1] 
            
            #Then a scipy-constrained minimisation from simplex's solution
            m = Minuit(f, fl = fl, afb = afb, _bin = i)
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            m.strategy = 2
            #con = NonlinearConstraint(posit_con, lb = 0, ub = np.inf)
            
            if not flexible:
                m.scipy(constraints = con)
            else:
                try:
                    m.scipy(constraints = con)
                except:
                    #Apply only positiveness constraint
                    print("\nOnly positiveness constraint applied, due to ERROR")
                    print("in minimisation with list of constraints.")
                    print("For more info, set flexible = False.")
                    m = Minuit(f, fl = fl, afb = afb, _bin = i)
                    m.fixed['_bin'] = True
                    m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
                    con = NonlinearConstraint(posit_con, lb = 0, ub = np.inf)
                    m.scipy(constraints = con)
                    m.hesse()
                    if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                        fl = m.values[0]
                        afb = m.values[1]
                        fl_err = m.errors[0]
                        afb_err = m.errors[1]
                
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
                fl_err = m.errors[0]
                afb_err = m.errors[1]
            else:
                #Apply only positiveness constraint
                print("\nOnly positiveness constraint applied, due to FAILURE")
                print("in minimisation with all constraints.")
                m = Minuit(f, fl = fl, afb = afb, _bin = i)
                m.fixed['_bin'] = True
                m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
                con = NonlinearConstraint(posit_con, lb = 0, ub = np.inf)
                m.scipy(constraints = con)
                m.hesse()
                if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                    fl = m.values[0]
                    afb = m.values[1]
                    fl_err = m.errors[0]
                    afb_err = m.errors[1]
                
            fls.append(fl);      afbs.append(afb)
            fl_errs.append(fl_err);  afb_errs.append(afb_err)
            
            if i == bin_number_to_check: 
                bin_results_to_check = m
                
            
    if bin_results_to_check:
        return (fls, afbs), (fl_errs, afb_errs), bin_results_to_check  
    else:
        return (fls, afbs), (fl_errs, afb_errs)
    





    
# #%%
# f = nLL_ctl
# guess = [0.5, 0.0]

# f.errordef = Minuit.LIKELIHOOD
# decimal_places = 4
# fls, fl_errs = [], []
# afbs, afb_errs = [], []

# #Define constraining function: positive PDFs
# def PDF_con (fl, afb, i):
#     i = int(i)
#     # ndarray_ctl = ctl_PDF( data[i]['costhetal'], fl, afb )
#     # ndarray_ctk = ctk_PDF( data[i]['costhetak'], fl )
#     ndarray_ctl = ctl_PDF(np.array([-1, -0.5, 0, 0.5,  1]), fl, afb)
#     ndarray_ctk = ctk_PDF(np.array([-1, -0.5, 0, 0.5, 1]), afb)
#     return min([min(ndarray_ctl), min(ndarray_ctk)]) 
# con = [NonlinearConstraint(PDF_con, lb = 0, ub = np.inf)]

# #Define cnonstraint function: normalised PDF
# def norm_ctl (fl, afb, i):
#     i = int(i)
#     Actl = quad(ctl_PDF, -1, 1, args = (fl, afb))
#     return Actl
# def norm_ctk (fl, afb, i):
#     i = int(i)
#     Actk = quad(ctk_PDF, -1, 1, args = fl)
#     return Actk
# lb = 1 - 1e-6; ub = 1 + 1e6
# con.append(NonlinearConstraint(norm_ctl, lb = lb, ub = ub))
# con.append(NonlinearConstraint(norm_ctk, lb = lb, ub = ub))

# i = 1
# m = Minuit(f, 
#            fl = guess[0], 
#            afb = guess[1], 
#            _bin = i)

# # First run a simplex minimisation
# m.fixed['_bin'] = True
# m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
# m.simplex()

# fls.append(m.values[0]);      afbs.append(m.values[1])
# fl_errs.append(m.errors[0]);  afb_errs.append(m.errors[1])

# #Then a scipy-constrained minimisation from simplex's solution
# m = Minuit(f, fl = m.values[0], afb = m.values[1], _bin = i)
# m.fixed['_bin'] = True
# m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
# m.scipy(constraints = con)
# m.hesse()

# fls.append(m.values[0]);      afbs.append(m.values[1])
# fl_errs.append(m.errors[0]);  afb_errs.append(m.errors[1])
# #%%
# # Finally a migrad minimisation starting from scipy's solution
# # mainly to obtain errors on estimates
# m = Minuit(f, fl = m.values[0], afb = m.values[1], _bin = i)
# m.strategy = 2
# m.fixed['_bin'] = True
# m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
# m.migrad()
# m.hesse()
    
# fls.append(m.values[0]);      afbs.append(m.values[1])
# fl_errs.append(m.errors[0]);  afb_errs.append(m.errors[1])


