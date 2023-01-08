# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:27:37 2022

@author: Stefano
"""
import numpy as np
from scipy.integrate import quad, dblquad, tplquad

import numpy.polynomial.legendre as npl
from functools import reduce
from find_coeff_numpy import legendre_eval, legendre_eval_project_1D
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% "EASY" PDFS with just 1,2 or 3 parameters
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ctl_PDF(costhetal, coeff):
    """
    Parameters
    ----------
    costhetal : ndarray
        Array of experimentally observed cos(theta_l).
    coeff: 2,ndarray
        Fl and Afb
    
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    fl, afb = coeff
    ctl = costhetal
    c2tl = 2 * ctl ** 2 - 1
    
    # If list of coeffs is NOT given
    if not hasattr(fl, "__len__") and not hasattr(afb, "__len__"):
        P = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +\
                              8/3 * afb * ctl)
     
    else:   
        fl = np.array(fl)
        afb = np.array(afb)
        #1D lists of cefficients given, to be manually meshed 
        if fl.ndim == 1 and afb.ndim == 1:
            if hasattr(ctl, "__len__"):
                P =[]
                for i, (ctli, c2tli) in enumerate(zip(ctl, c2tl)):
                    Pi = []
                    for j, afbj in enumerate(afb):
                        Pij =3/8*(3/2 - 1/2 * fl + 1/2 * c2tli * (1 - 3 * fl)+\
                                              8/3 * afbj * ctli)
                        Pi.append(Pij)
                    Pi = np.array(Pi)
                    P.append(Pi)
                P = np.array(P)
            #If list of coeffs only is given
            else:
                P = []
                for j, afbj in enumerate(afb):
                    Pi =3/8*(3/2 - 1/2 * fl + 1/2 * c2tli * (1 - 3 * fl)+\
                                          8/3 * afbj * ctli)
                    P.append(Pi)
                P = np.array(P)
        #Meshgrid of cefficients given       
        elif fl.ndim == 2 and afb.ndim == 2:
            if hasattr(ctl, "__len__"):
                P =[]
                for i, (ctli, c2tli) in enumerate(zip(ctl, c2tl)):
                    Pi = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tli * (1 - 3 * fl) +\
                                          8/3 * afb * ctli)
                    P.append(Pi)
                P = np.array(P)   
            #If list of coeffs only is given
            else:
                P = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) +\
                                      8/3 * afb * ctl)          
    return P


def ctk_PDF(costhetak, fl):
    """
    Parameters
    ----------
    costhetak : ndarray
        Array of experimentally observed cos(theta_k).
    fl : float
        The factor F_L in the distribution.

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_k
    """
    ctk = costhetak
    
    # If list of coeffs is NOT given
    if not hasattr(fl, "__len__"):
        P = 3/4 * ( (3*fl - 1) * ctk**2 + 1 - fl)
    
    #If list of coeff and costhetas is given
    elif hasattr(ctk, "__len__"):
        P =[]
        for i, (ctki) in enumerate(ctk):
            Pi = 3/4 * ( (3*fl - 1) * ctki**2 + 1 - fl)
            P.append(Pi)
        P = np.array(P)
    
    #If list of coeffs only is given
    else:
        P = 3/4 * ( (3*fl - 1) * ctk**2 + 1 - fl)

    return P



def phi_PDF(phi, coeff):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi.
    coeff: 2,ndarray
        [s3, A_I_m]

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing phi
    """
    s3, aim = coeff
    P = (1 + s3*np.cos(2*phi) + aim*np.sin(2*phi)) / (2 * np.pi)  
    return P


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%ACCEPTANCE-MODULATED EASY PDFs
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ctl_acc_PDF(costhetal, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    costhetal : ndarray
        Array of experimentally observed cos(theta_l), for a bin of q2.
    coeff : 2,ndarray
        Fl and Afb
    acc_poly : ndarray
        The acceptance function for costhetal, for a bin of q2.
    accept : Bool
        Accept also negative values of acc
        
    Returns
    -------
    normalised_P : ndarray
        Normalised Acceptance-modulated Probability Distribution for observing 
        costheta_l.
    """
    pdf_not_normal = ctl_acc_PDF_NotNormal(costhetal, coeff, acc_poly, accept)
    #print(pdf_not_normal)
    if not hasattr(coeff[0], "__len__") and not hasattr(coeff[1], "__len__"):
        A = quad(ctl_acc_PDF_NotNormal, -1, 1, args = (coeff, acc_poly, accept),
                 epsabs = 1e-3, epsrel = 1e-3)[0]
        acc_pdf = pdf_not_normal / A
    else:
        fl = np.array(coeff[0])
        afb = np.array(coeff[1])
        if fl.ndim == 1 and afb.ndim == 1:
            for i, fli in enumerate(fl):
                for j, afbj in enumerate(afb):
                    A = quad(ctl_acc_PDF_NotNormal, -1, 1,
                             args = ([fli, afbj], acc_poly, accept),
                             epsabs = 1e-3, epsrel = 1e-3)[0]
                    #lower convergence threshold to improve velocity
                    pdf_not_normal[:,i][:,j] /= A 
            acc_pdf = pdf_not_normal
        elif fl.ndim == 2 and afb.ndim == 2:
            for i, fli in enumerate(fl[0]):
                for j, afbj in enumerate(afb[:,0]):
                    A = quad(ctl_acc_PDF_NotNormal, -1, 1,
                             args = ([fli, afbj], acc_poly, accept),
                             epsabs = 1e-3, epsrel = 1e-3)[0]
                    #lower convergence threshold to improve velocity
                    pdf_not_normal[:,j][:,i] /= A 
            acc_pdf = pdf_not_normal
    return acc_pdf


def ctk_acc_PDF(costhetak, fl, acc_poly, accept = False):
    """
    Parameters
    ----------
    costhetak : ndarray
        Array of experimentally observed cos(theta_k), for a bin of q2.
    fl : float or ndarray
        Fl coefficient.
    acc_poly : ndarray
        The acceptance function for costhetak, for a bin of q2.
    accept : Bool
        Accept also negative values of acc
        
    Returns
    -------
    normalised_P : ndarray
        Normalised Acceptance-modulated Probability Distribution for observing 
        costheta_k.
    """
    pdf_not_normal = ctk_acc_PDF_NotNormal(costhetak, fl, acc_poly, accept)
    if not hasattr(fl, "__len__"):
        A = quad(ctk_acc_PDF_NotNormal, -1, 1,args = (fl, acc_poly, accept))[0]
        acc_pdf = pdf_not_normal / A
    else:
        for i, fli in enumerate(fl):
            A = quad(ctk_acc_PDF_NotNormal, -1, 1,
                     args = (fli, acc_poly, accept),
                     epsabs = 1e-3, epsrel = 1e-3)[0]
            #lower convergence threshold to improve velocity
            pdf_not_normal[:,i] /= A 
        acc_pdf = pdf_not_normal
    return acc_pdf


def phi_acc_PDF(phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi, for a bin of q2.
    coeff: 2,ndarray
        [s3, A_I_m]
    acc_poly : ndarray
        The acceptance function for costhetal, for a bin of q2.
    accept : Bool
        Accept also negative values of acc
        
    Returns
    -------
    normalised_P : ndarray
        Normalised Acceptance-modulated Probability Distribution for observing 
        phi.
    """
    pdf_not_normal = phi_acc_PDF_NotNormal(phi, coeff, acc_poly, accept)
    #print(pdf_not_normal)
    if not hasattr(coeff[0], "__len__") and not hasattr(coeff[1], "__len__"):
        A = quad(phi_acc_PDF_NotNormal, -np.pi, np.pi, args = (coeff, acc_poly, accept),
                 epsabs = 1e-3, epsrel = 1e-3)[0]
        acc_pdf = pdf_not_normal / A
    else:
        s3 = np.array(coeff[0])
        aim = np.array(coeff[1])
        if s3.ndim == 1 and aim.ndim == 1:
            for i, s3i in enumerate(s3):
                for j, aimj in enumerate(aim):
                    A = quad(phi_acc_PDF_NotNormal, -1, 1,
                             args = ([s3i, aimj], acc_poly, accept),
                             epsabs = 1e-3, epsrel = 1e-3)[0]
                    #lower convergence threshold to improve velocity
                    pdf_not_normal[:,i][:,j] /= A 
            acc_pdf = pdf_not_normal
        elif s3.ndim == 2 and aim.ndim == 2:
            for i, s3i in enumerate(s3[0]):
                for j, aimj in enumerate(aim[:,0]):
                    A = quad(phi_acc_PDF_NotNormal, -1, 1,
                             args = ([s3i, aimj], acc_poly, accept),
                             epsabs = 1e-3, epsrel = 1e-3)[0]
                    #lower convergence threshold to improve velocity
                    pdf_not_normal[:,j][:,i] /= A 
            acc_pdf = pdf_not_normal
    return acc_pdf
    
    


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%BEHIND THE SCENES
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ctl_acc_PDF_NotNormal(costhetal, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    costhetal : ndarray
        Array of experimentally observed cos(theta_l), for a bin of q2.
    coeff: 2,ndarray
        Fl and Afb
    acc_poly: callable, ndarray, or 1
        The acceptance function for costhetal, for a bin of q2.
    accept : Bool
        Accept also negative values of acc
        
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing costheta_l
        modulated by the acceptance function. 
    """
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if hasattr(costhetal, "__len__"):
        for ctli in costhetal:
            
            if callable(acc_poly):
                acc = acc_poly(ctli)
            elif hasattr(acc_poly, "__len__"): 
                acc = legendre_eval_project_1D(acc_poly, ctli, 0)
            elif acc_poly == 1:
                acc = 1
           
            if acc > 0 or accept:
                p = ctl_PDF(ctli, coeff)
                res.append( p * acc)
            else:
                continue
        res = np.array(res)
        
    else:
        if callable(acc_poly):
            acc = acc_poly(costhetal)
        elif hasattr(acc_poly, "__len__"): 
            acc = legendre_eval_project_1D(acc_poly, costhetal, 0)
        elif acc_poly == 1:
            acc = 1
        
        if acc > 0 or accept:
            res = ctl_PDF(costhetal, coeff) * acc
        else:
            res = 1
            
    return res


def ctk_acc_PDF_NotNormal(costhetak, fl, acc_poly, accept = False):
    """
    Parameters
    ----------
    costhetak : ndarray
        Array of experimentally observed cos(theta_k).
    fl : float
        The factor F_L in the distribution, for a bin of q2.
    acc_poly: ndarray
        The acceptance function for costhetak, for a bin of q2.
    accept : Bool
        Accept also negative values of acc

    Returns
    -------
    res : ndarray
        Probability Distribution for observing costheta_k modulated by the
        acceptance function. 
    """
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if hasattr(costhetak, "__len__"):
        for ctki in costhetak:
            if callable(acc_poly):
                acc = acc_poly(ctki)
            elif hasattr(acc_poly, "__len__"): 
                acc = legendre_eval_project_1D(acc_poly, ctki, 1)
            elif acc_poly == 1:
                acc = 1
            
            if acc > 0 or accept:
                res.append(ctk_PDF(ctki, fl) * acc)
                
        res = np.array(res)
    else:
        if callable(acc_poly):
            acc = acc_poly(costhetak)
        elif hasattr(acc_poly, "__len__"): 
            acc = legendre_eval_project_1D(acc_poly, costhetak, 1)
        elif acc_poly == 1:
            acc = 1
            
        if acc > 0 or accept:
            res = ctk_PDF(costhetak, fl) * acc
        else:
            res = 1
    return res

def phi_acc_PDF_NotNormal(phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi, for a bin of q2.
    coeff: 3,ndarray
        [Fl, s3, A_I_m]
    acc_poly: ndarray
        The acceptance function for phi, for a bin of q2.
    accept : Bool
        Accept also negative values of acc
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing phi modulated by
        acceptance function.
    """
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if hasattr(phi, "__len__"):
        for phii in phi:
            if callable(acc_poly):
                acc = acc_poly(phii)
            elif hasattr(acc_poly, "__len__"): 
                acc = legendre_eval_project_1D(acc_poly, phii, 2)
            elif acc_poly == 1:
                acc = 1
            
            if acc > 0 or accept:
                res.append(phi_PDF(phii, coeff) * acc)
        res = np.array(res)
    else:
        if callable(acc_poly):
            acc = acc_poly(phi)
        elif hasattr(acc_poly, "__len__"): 
            acc = legendre_eval_project_1D(acc_poly, phi, 2)
        elif acc_poly == 1:
            acc = 1
        
        if acc > 0 or accept:
            res = phi_PDF(phi, coeff) * acc
        else:
            res = 1
    return res


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%BEHIND THE SCENES
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ctl_acc_PDF_NotNormal(costhetal, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    costhetal : ndarray
        Array of experimentally observed cos(theta_l), for a bin of q2.
    coeff: 2,ndarray
        Fl and Afb
    acc_poly: callable, ndarray, or 1
        The acceptance function for costhetal, for a bin of q2.
    accept : Bool
        Accept also negative values of acc
        
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing costheta_l
        modulated by the acceptance function. 
    """
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if hasattr(costhetal, "__len__"):
        for ctli in costhetal:
            
            if callable(acc_poly):
                acc = acc_poly(ctli)
            elif hasattr(acc_poly, "__len__"): 
                acc = legendre_eval_project_1D(acc_poly, ctli, 0)
            elif acc_poly == 1:
                acc = 1
           
            if acc > 0 or accept:
                p = ctl_PDF(ctli, coeff)
                res.append( p * acc)
            else:
                continue
        res = np.array(res)
        
    else:
        if callable(acc_poly):
            acc = acc_poly(costhetal)
        elif hasattr(acc_poly, "__len__"): 
            acc = legendre_eval_project_1D(acc_poly, costhetal, 0)
        elif acc_poly == 1:
            acc = 1
        
        if acc > 0 or accept:
            res = ctl_PDF(costhetal, coeff) * acc
        else:
            res = 1
            
    return res


def ctk_acc_PDF_NotNormal(costhetak, fl, acc_poly, accept = False):
    """
    Parameters
    ----------
    costhetak : ndarray
        Array of experimentally observed cos(theta_k).
    fl : float
        The factor F_L in the distribution, for a bin of q2.
    acc_poly: ndarray
        The acceptance function for costhetak, for a bin of q2.
    accept : Bool
        Accept also negative values of acc

    Returns
    -------
    res : ndarray
        Probability Distribution for observing costheta_k modulated by the
        acceptance function. 
    """
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if hasattr(costhetak, "__len__"):
        for ctki in costhetak:
            if callable(acc_poly):
                acc = acc_poly(ctki)
            elif hasattr(acc_poly, "__len__"): 
                acc = legendre_eval_project_1D(acc_poly, ctki, 1)
            elif acc_poly == 1:
                acc = 1
            
            if acc > 0 or accept:
                res.append(ctk_PDF(ctki, fl) * acc)
                
        res = np.array(res)
    else:
        if callable(acc_poly):
            acc = acc_poly(costhetak)
        elif hasattr(acc_poly, "__len__"): 
            acc = legendre_eval_project_1D(acc_poly, costhetak, 1)
        elif acc_poly == 1:
            acc = 1
            
        if acc > 0 or accept:
            res = ctk_PDF(costhetak, fl) * acc
        else:
            res = 1
    return res

def phi_acc_PDF_NotNormal(phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi, for a bin of q2.
    coeff: 3,ndarray
        [Fl, s3, A_I_m]
    acc_poly: ndarray
        The acceptance function for phi, for a bin of q2.
    accept : Bool
        Accept also negative values of acc
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing phi modulated by
        acceptance function.
    """
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if hasattr(phi, "__len__"):
        for phii in phi:
            if callable(acc_poly):
                acc = acc_poly(phii)
            elif hasattr(acc_poly, "__len__"): 
                acc = legendre_eval_project_1D(acc_poly, phii, 2)
            elif acc_poly == 1:
                acc = 1
            
            if acc > 0 or accept:
                res.append(phi_PDF(phii, coeff) * acc)
        res = np.array(res)
    else:
        if callable(acc_poly):
            acc = acc_poly(phi)
        elif hasattr(acc_poly, "__len__"): 
            acc = legendre_eval_project_1D(acc_poly, phi, 2)
        elif acc_poly == 1:
            acc = 1
        
        if acc > 0 or accept:
            res = phi_PDF(phi, coeff) * acc
        else:
            res = 1
    return res