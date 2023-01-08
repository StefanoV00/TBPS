# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:27:37 2022

@author: Stefano
"""
import numpy as np

#%% "EASY" PDFS with just 1,2 or 3 parameters

def ctl_PDF(costhetal, fl, afb):
    """
    Parameters
    ----------
    costhetal : ndarray
        Array of experimentally observed cos(theta_l).
    fl : float
        The factor F_L in the distribution.
    afb : float
        The factor A_FB in the distribution.
    
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    ctl = costhetal
    c2tl = 2 * ctl ** 2 - 1
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



def phi_PDF(phi, fl, at, aim):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi.
    fl : float
        The factor F_L in the distribution.
    at : float
        The factor A_t in the distribution.
    aim : float
        The factor A_I_m in the distribution.

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing phi
    """
    P = 1 + 1/2*(1-fl)*at*np.cos(2*phi) + aim * np.sin(2*phi)
    P /= (2 * np.pi) 
    return P

#%%ACCEPTANCE-MODULATED PDFs

def ctl_acc_PDF(costhetal, fl, afb, acc_poly):
    """
    Parameters
    ----------
    costhetal : ndarray
        Array of experimentally observed cos(theta_l), for a bin of q2.
    fl : float
        The factor F_L in the distribution, for a bin of q2.
    afb : float
        The factor A_FB in the distribution, for a bin of q2.
    acc_poly: ndarray
        The acceptance function for costhetal, for a bin of q2.
        
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    return ctl_PDF(costhetal, fl, afb) * acc_poly(costhetal)


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



def phi_acc_PDF(phi, fl, at, aim, acc_poly):
    """
    Parameters
    ----------
    phi : ndarray
        Array of experimentally observed phi, for a bin of q2.
    fl : float
        The factor F_L in the distribution, for a bin of q2.
    at : float
        The factor A_t in the distribution, for a bin of q2.
    aim : float
        The factor A_I_m in the distribution, for a bin of q2.
    acc_poly: numpy 4th order polynomial
        The acceptance function for phi, for a bin of q2.

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing phi
    """
    return phi_PDF(phi, fl, at, aim) * acc_poly(phi)
