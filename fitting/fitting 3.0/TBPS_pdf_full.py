# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:03:39 2022

@author: Stefano
"""
import numpy as np
from scipy.integrate import quad, dblquad, tplquad

from find_coeff_numpy import legendre_eval, legendre_eval_project_1D
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FULL CP-AVERAGED PDF
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def full_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with values of observables, in order ctl, ctk, phi. If these 
             are lists, they must have same sizes. 
    
    coeffs : [fl, afb, s3, s4, s5, s7, s8, s9]

    Returns
    -------
    normalised_P : ndarray or scalar
        Full CP-averaged Probability Distribution. Scalar if ctl, ctk, and phi
        are just scalars.
    """
    ctl, ctk, phi = angles
    fl, afb, s3, s4, s5, s7, s8, s9 = coeffs

    if not hasattr(ctl, "__len__"):
        # theta_l = np.arccos(ctl)
        # theta_k = np.arccos(ctk)
        
        sq_ctl = ctl * ctl           # cos(theta_l) squared
        sq_stl = 1 - sq_ctl          # sin(theta_l) squared
        stl    = np.sqrt(sq_stl)     # sin(theta_l)
        ctl_2  = sq_ctl - sq_stl     # cos(2 * theta_l)
        stl_2  = 2*stl*ctl           # sin(2 * theta_l)
        
        sq_ctk = ctk * ctk           # cos(theta_k) squared
        sq_stk = 1-sq_ctk            # sin(theta_k) squared
        stk    = np.sqrt(sq_stk)     # sin(theta_k)
        ctk_2  = sq_ctk - sq_stk     # cos(2 * theta_k)
        stk_2  = 2 * stk * ctk       # sin(2 * theta_k)
        
        cphi   = np.cos(phi)         # cos(phi)
        c2phi  = np.cos(2*phi)       # cos(phi)
        sphi   = np.cos(phi)         # cos(phi)
        s2phi  = np.cos(2*phi)       # cos(phi)
        
        subP  = (3/4) * (1-fl) * (sq_stk) + fl * sq_ctk
        subP += (1/4) * (1-fl) * sq_stk * ctl_2 - fl * sq_ctk * ctl_2
        subP += s3 * sq_stk * sq_stl * c2phi
        subP += s4 * stk_2 * stl_2 * cphi
        subP += s5 * stk_2 * stl * cphi
        subP += 4/3*afb * sq_stk * ctl
        subP += s7 * stk_2 * stl * sphi
        subP += s8 * stk_2 * stl_2 * sphi
        subP += s9 * sq_stk * sq_stl * s2phi
        P = 9 * subP / (32 * np.pi) 
        return P
    
    else:
        P = []
        for i, phi_angle in enumerate(phi):
            P.append(full_PDF([ctl[i], ctk[i], phi_angle], coeffs))    
        return np.array(P)


def full_acc_PDF(angles, coeffs, acc_poly):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s4, s5, s7, s8, s9]
             Must be one dimensional array.
    
    acc_poly : numpy polynomial of three variables (ctl, ctk, phi)

    Returns
    -------
    res : ndarray
        Normalised Full CP-averaged Probability Distribution modulated by 
        acceptance function.
    """
    if acc_poly == 1:
        return full_PDF(angles, coeffs)
    ctl, ctk, phi = angles
    pdf_not_normal = full_acc_PDF_NotNormal(ctl, ctk, phi, coeffs, acc_poly)
    A = tplquad(full_acc_PDF_NotNormal, 
                     0, 2*np.pi,
                     -1, 1, 
                     -1, 1,
                     args = (coeffs, acc_poly),
                     epsabs = 1e-3, epsrel = 1e-2)[0]
    acc_pdf = pdf_not_normal / A
    return acc_pdf


def full_acc_PDF_NotNormal(ctl, ctk, phi, coeffs, acc_poly):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s4, s5, s7, s8, s9]
             Must be one dimensional array.
    
    acc_poly : numpy polynomial of three variables (ctl, ctk, phi)

    Returns
    -------
    res : ndarray
        Not Normalised Full CP-averaged Probability Distribution modulated by 
        acceptance function.
    """
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if callable(acc_poly):
        if hasattr(phi, "__len__"):
            for i in range(len(ctl)):
                angles_i = [ctl[i], ctk[i], phi[i]]
                res.append(full_PDF(angles_i, coeffs) * \
                           acc_poly(ctl[i], ctk[i], phi[i]))
            res = np.array(res)
        else:
            res = full_PDF([ctl, ctk, phi], coeffs) * acc_poly(ctl, ctk, phi)
        return res
    elif hasattr(acc_poly, "__len__"):
        C = acc_poly*1
        if hasattr(phi, "__len__"):
            for i in range(len(ctl)):
                angles_i = [ctl[i], ctk[i], phi[i]]
                res.append(full_PDF(angles_i, coeffs) \
                           * legendre_eval(C, ctl[i], ctk[i], phi[i]))
            res = np.array(res)
        else:
            res = full_PDF([ctl, ctk, phi], coeffs) \
                  * legendre_eval(C, ctl, ctk, phi)
        return res