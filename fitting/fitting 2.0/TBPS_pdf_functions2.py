# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:27:37 2022

@author: Stefano
"""
import numpy as np

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# "EASY" PDFS with just 1,2 or 3 parameters
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
            
    #If list of coeff and costhetas is given
    elif hasattr(ctl, "__len__"):
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
    P = 3/2 * fl * ctk**2 + 3/4 *(1-fl)*(1-ctk**2) 
    return P



def phi_PDF(phi, coeff):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi.
    coeff: 3,ndarray
        Fl and A_t and A_I_m

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing phi
    """
    fl, at, aim = coeff
    P = 1 + 1/2*(1-fl)*at*np.cos(2*phi) + aim * np.sin(2*phi)
    P /= (2 * np.pi) 
    return P


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ACCEPTANCE-MODULATED EASY PDFs
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ctl_acc_PDF(costhetal, coeff, acc_poly):
    """
    Parameters
    ----------
    costhetal : ndarray
        Array of experimentally observed cos(theta_l), for a bin of q2.
    coeff: 2,ndarray
        Fl and Afb
    acc_poly: ndarray
        The acceptance function for costhetal, for a bin of q2.
        
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    
    res = []
    # need to loop for the case when coeff is an array of arrays.
    if hasattr(costhetal, "__len__"):
        for ctli in costhetal:
            res.append(ctl_PDF(ctli, coeff) * acc_poly(ctli))
        res = np.array(res)
    else:
        res = ctl_PDF(costhetal, coeff) * acc_poly(costhetal)
    return res


def ctk_acc_PDF(costhetak, fl, acc_poly):
    """
    Parameters
    ----------
    costhetak : ndarray
        Array of experimentally observed cos(theta_k).
    fl : float
        The factor F_L in the distribution, for a bin of q2.
    acc_poly: ndarray
        The acceptance function for costhetak, for a bin of q2.

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_k
    """
    return ctk_PDF(costhetak, fl) * acc_poly(costhetak)



def phi_acc_PDF(phi, coeff, acc_poly):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi, for a bin of q2.
    coeff: 3,ndarray
        Fl and A_t and A_I_m
    acc_poly: numpy 4th order polynomial
        The acceptance function for phi, for a bin of q2.

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing phi
    """
    return phi_PDF(phi, coeff) * acc_poly(phi)