# - * - coding: utf-8 - * -
"""
Created on Mon Mar  7 18:37:42 2022

@author: Stefano
"""

import numpy as np
from scipy.integrate import quad, dblquad, tplquad

from find_coeff_numpy import legendre_eval, legendre_eval_project_1D
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# "Folded-S" PDFS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def s4_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s4]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for the Trasnformed Variables (isolating
        S4).
    """
    ctl, ctk, phi = angles

    fl = coeffs[0]
    afb = coeffs[1]     
    s3 = coeffs[2]
    s4 = coeffs[3]


    if not hasattr(phi, "__len__"):

        theta_l = np.arccos(ctl)
        theta_k = np.arccos(ctk)

        if phi < 0:
            phi = -phi
        
        if theta_l > np.pi/2:
            phi = np.pi - phi
            theta_l = np.pi - theta_l
        
        ctl_2 = np.cos(2 * theta_l) # cos(2 * theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared
        sq_stl = 1 - sq_ctl         # sin(theta_l) squared
        sq_ctk = ctk * ctk          # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared
        
        subP  = (3/4) * (1-fl) * (sq_stk) + fl * sq_ctk
        subP += (1/4) * (1-fl) * sq_stk * ctl_2 - fl * sq_ctk * ctl_2
        subP += s3 * sq_stk * sq_stl * np.cos(2 * phi)
        subP += s4 * np.sin(2 * theta_k) * np.sin(2 * theta_l) * np.cos(phi)
        P = ( 9 * subP / (8 * np.pi) )
        return P
    
    else:
        P = []
        for i, phi_angle in enumerate(phi):
            P.append(s4_PDF([ctl[i], ctk[i], phi_angle], coeffs))    
        return np.array(P)



def s5_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s5]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for the Trasnformed Variables (isolating
        S5).
    """
    ctl, ctk, phi = angles

    fl = coeffs[0]
    afb = coeffs[1]     
    s3 = coeffs[2]
    s5 = coeffs[3]


    if not hasattr(phi, "__len__"):
        theta_l = np.arccos(ctl)
        theta_k = np.arccos(ctk)

        if phi < 0:
            phi = -phi
        
        if theta_l > np.pi/2:
            theta_l = np.pi - theta_l
        
        ctl_2 = np.cos(2 * theta_l)   # cos(2 * theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared
        sq_stl = 1 - sq_ctl         # sin(theta_l) squared
        sq_ctk = ctk * ctk      # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared
        
        subP  = (3/4) * (1-fl) * (sq_stk) + fl * sq_ctk 
        subP += (1/4) * (1-fl) * sq_stk * ctl_2 - fl * sq_ctk * ctl_2 
        subP += s3 * sq_stk * sq_stl * np.cos(2 * phi) 
        subP += s5 * np.sin(2 * theta_k) * np.sin(theta_l) * np.cos(phi)
        P = ( 9 * subP / (8 * np.pi) )
        return P
    else:
        P = []
        for i, phi_angle in enumerate(phi):
            P.append(s5_PDF([ctl[i], ctk[i], phi_angle], coeffs))    
        return np.array(P)



def s7_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s7]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for the Trasnformed Variables (isolating
        S7).
    """
    
    ctl, ctk, phi = angles

    fl = coeffs[0]
    afb = coeffs[1]     
    s3 = coeffs[2]
    s7 = coeffs[3]


    if not hasattr(phi, "__len__"):
        theta_l = np.arccos(ctl)
        theta_k = np.arccos(ctk)

        if phi > np.pi/2:
            phi = np.pi-phi
        
        if phi < -np.pi/2:
            phi = -np.pi - phi

        if theta_l > np.pi/2:
            theta_l = np.pi - theta_l
        
        ctl_2 = np.cos(2 * theta_l)   # cos(2 * theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared
        sq_stl = 1 - sq_ctl         # sin(theta_l) squared
        sq_ctk = ctk * ctk          # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared

        subP =  (3/4) * (1-fl) * (sq_stk) + fl * sq_ctk 
        subP += (1/4) * (1-fl) * sq_stk * ctl_2 - fl * sq_ctk * ctl_2
        subP += s3 * sq_stk * sq_stl * np.cos(2 * phi) 
        subP += s7 * np.sin(2 * theta_k) * np.sin(theta_l) * np.sin(phi)
        P = ( 9 * subP / (8 * np.pi) )
        
        return P
    
    else:
        P = []
        for i, phi_angle in enumerate(phi):
            P.append(s7_PDF([ctl[i], ctk[i], phi_angle], coeffs))    
        return np.array(P)



def s8_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s8]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for the Trasnformed Variables (isolating
        S8).
    """
    
    ctl, ctk, phi = angles

    fl = coeffs[0]
    afb = coeffs[1]     
    s3 = coeffs[2]
    s8 = coeffs[3]


    if not hasattr(phi, "__len__"):
        theta_l = np.arccos(ctl)
        theta_k = np.arccos(ctk)

        if phi > np.pi/2:
            phi = np.pi - phi

        if phi < -np.pi/2:
            phi = -np.pi - phi
        
        if theta_l > np.pi/2:
            theta_l = np.pi - theta_l
            theta_k = np.pi - theta_k
        
        ctl_2 = np.cos(2 * theta_l)   # cos(2 * theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared
        sq_stl = 1 - sq_ctl         # sin(theta_l) squared
        sq_ctk = ctk * ctk          # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared

        subP  = (3/4) * (1-fl) * (sq_stk) + fl * sq_ctk
        subP += (1/4) * (1-fl) * sq_stk * ctl_2 - fl * sq_ctk * ctl_2 
        subP += s3 * sq_stk * sq_stl * np.cos(2 * phi) 
        subP += s8 * np.sin(2 * theta_k) * np.sin(2 * theta_l) * np.sin(phi)
        P = ( 9 * subP / (8 * np.pi) )
        return P
    
    else:
        P = []
        for i, phi_angle in enumerate(phi):
            P.append(s8_PDF([ctl[i], ctk[i], phi_angle], coeffs))    
        return np.array(P)



def s9_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s9]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for the Trasnformed Variables (isolating S3
        and S9).
    """
    ctl, ctk, phi = angles

    fl, afb, s3, s9 = coeffs

    if not hasattr(phi, "__len__"):
        theta_l = np.arccos(ctl)
        theta_k = np.arccos(ctk)

        if phi < 0:
            phi = np.pi + phi
        
        ctl_2 = np.cos(2 * theta_l)   # cos(2 * theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared
        sq_stl = 1 - sq_ctl         # sin(theta_l) squared
        sq_ctk = ctk * ctk          # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared

        subP  = (3/4) * (1-fl) * sq_stk + fl * sq_ctk 
        subP += (1/4) * (1-fl) * sq_stk * ctl_2 - fl * sq_ctk * ctl_2
        subP += s3 * sq_stk * sq_stl * np.cos(2 * phi) 
        subP += (4/3) * afb * sq_stk * ctl 
        subP += s9 * sq_stk * sq_stl * np.cos(2 * phi)
        P = ( 9 * subP / (16 * np.pi) )
        return P
    
    else:
        P = []
        for i, phi_angle in enumerate(phi):
            P.append(s9_PDF([ctl[i], ctk[i], phi_angle], coeffs))    
        return np.array(P)
    



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%% ACCEPTANCE MODULATED "Folded-S" PDFS 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def s4_acc_PDF(angles, coeff, acc_poly, accept = False):
    """
    Normalised Acceptance Modulated pdf with Si isolated, NOT suitable for
    multidimensional plotting over array of coefficients. 
    
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s4]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi).
        
    Returns
    -------
    res : ndarray
        Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    ctl, ctk, phi = angles
    if acc_poly == 1:
        return s4_PDF(angles, coeff)
    pdf_not_normal = s4_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)
    if not hasattr(coeff[0], "__len__") and not hasattr(coeff[1], "__len__"):
        A = tplquad(s4_acc_PDF_NotNormal, #tpl quad goes in order z, y, x
                     0, 2*np.pi,
                     -1, 1, 
                     -1, 1,
                     args = (coeff, acc_poly, accept),
                     epsabs = 1e-3, epsrel = 1e-2)[0]
        acc_pdf = pdf_not_normal / A
    return acc_pdf


def s5_acc_PDF(angles, coeff, acc_poly, accept = False):
    """
    Normalised Acceptance Modulated pdf with Si isolated, NOT suitable for
    multidimensional plotting over array of coefficients. 
    
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s5]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi).
        
    Returns
    -------
    res : ndarray
        Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    ctl, ctk, phi = angles
    if acc_poly == 1:
        return s5_PDF(angles, coeff)
    pdf_not_normal = s5_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)
    if not hasattr(coeff[0], "__len__") and not hasattr(coeff[1], "__len__"):
        A = tplquad(s4_acc_PDF_NotNormal, #tpl quad goes in order z, y, x
                     0, 2*np.pi,
                     -1, 1, 
                     -1, 1,
                     args = (coeff, acc_poly, True),
                     epsabs = 1e-3, epsrel = 1e-2)[0]
        acc_pdf = pdf_not_normal / A
    return acc_pdf


def s7_acc_PDF(angles, coeff, acc_poly, accept = False):
    """
    Normalised Acceptance Modulated pdf with Si isolated, NOT suitable for
    multidimensional plotting over array of coefficients. 
    
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s7]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi).
        
    Returns
    -------
    res : ndarray
        Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    ctl, ctk, phi = angles
    if acc_poly == 1:
        return s7_PDF(angles, coeff)
    pdf_not_normal = s7_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)
    if not hasattr(coeff[0], "__len__") and not hasattr(coeff[1], "__len__"):
        A = tplquad(s4_acc_PDF_NotNormal, #tpl quad goes in order z, y, x
                     0, 2*np.pi,
                     -1, 1, 
                     -1, 1,
                     args = (coeff, acc_poly, True),
                     epsabs = 1e-3, epsrel = 1e-2)[0]
        acc_pdf = pdf_not_normal / A
    return acc_pdf


def s8_acc_PDF(angles, coeff, acc_poly, accept = False):
    """
    Normalised Acceptance Modulated pdf with Si isolated, NOT suitable for
    multidimensional plotting over array of coefficients. 
    
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s8]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi).
        
    Returns
    -------
    res : ndarray
        Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    ctl, ctk, phi = angles
    if acc_poly == 1:
        return s8_PDF(angles, coeff)
    pdf_not_normal = s8_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)
    if not hasattr(coeff[0], "__len__") and not hasattr(coeff[1], "__len__"):
        A = tplquad(s4_acc_PDF_NotNormal, #tpl quad goes in order z, y, x
                     0, 2*np.pi,
                     -1, 1, 
                     -1, 1,
                     args = (coeff, acc_poly, True),
                     epsabs = 1e-3, epsrel = 1e-2)[0]
        acc_pdf = pdf_not_normal / A
    return acc_pdf


def s9_acc_PDF(angles, coeff, acc_poly, accept = False):
    """
    Normalised Acceptance Modulated pdf with Si isolated, NOT suitable for
    multidimensional plotting over array of coefficients. 
    
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi. 
    
    coeffs : [fl, afb, s3, s9]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi).
        
    Returns
    -------
    res : ndarray
        Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    ctl, ctk, phi = angles
    if acc_poly == 1:
        return s9_PDF(angles, coeff)
    pdf_not_normal = s9_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)
    if not hasattr(coeff[0], "__len__") and not hasattr(coeff[1], "__len__"):
        A = tplquad(s4_acc_PDF_NotNormal, #tpl quad goes in order z, y, x
                     0, 2*np.pi,
                     -1, 1, 
                     -1, 1,
                     args = (coeff, acc_poly, True),
                     epsabs = 1e-3, epsrel = 1e-2)[0]
        acc_pdf = pdf_not_normal / A
    return acc_pdf

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%FOR FIT
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def s4_acc_PDF_forfit(angles, coeff, acc_poly, accept = False):
    ctl, ctk, phi = angles
    return s4_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)

def s5_acc_PDF_forfit(angles, coeff, acc_poly, accept = False):
    ctl, ctk, phi = angles
    return s5_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)

def s7_acc_PDF_forfit(angles, coeff, acc_poly, accept = False):
    ctl, ctk, phi = angles
    return s7_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)

def s8_acc_PDF_forfit(angles, coeff, acc_poly, accept = False):
    ctl, ctk, phi = angles
    return s8_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)

def s9_acc_PDF_forfit(angles, coeff, acc_poly, accept = False):
    ctl, ctk, phi = angles
    return s9_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#%%BEHIND THE SCENES
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def s4_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s4]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi)
        
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    
    res = []; angles = [ctl, ctk, phi]
    # need to loop for the case when coeff is an array of arrays.
    ctl = angles[0]
    if callable(acc_poly):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = acc_poly(ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s4_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc)
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = acc_poly(ctl, ctk, phi)
            if acc > 0 or accept:   
                res = s4_PDF(angles, coeff) * acc
            else:
                return 1
        return res
    elif hasattr(acc_poly, "__len__"):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = legendre_eval(acc_poly, ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s4_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc )
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = legendre_eval(acc_poly, ctl, ctk, phi)
            if acc > 0 or accept:
                res = s4_PDF(angles, coeff) * acc
            else:
                res = 1
        return res



def s5_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s5]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi)
        
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    
    res = []; angles = [ctl, ctk, phi]
    # need to loop for the case when coeff is an array of arrays.
    
    if callable(acc_poly):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = acc_poly(ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s5_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc)
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = acc_poly(ctl, ctk, phi)
            if acc > 0 or accept:   
                res = s5_PDF(angles, coeff) * acc
            else:
                return 1
        return res
    elif hasattr(acc_poly, "__len__"):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = legendre_eval(acc_poly, ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s5_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc )
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = legendre_eval(acc_poly, ctl, ctk, phi)
            if acc > 0 or accept:
                res = s5_PDF(angles, coeff) * acc
            else:
                res = 1
        return res



def s7_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s7]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi)
        
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    
    res = []; angles = [ctl, ctk, phi]
    # need to loop for the case when coeff is an array of arrays.

    if callable(acc_poly):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = acc_poly(ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s7_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc)
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = acc_poly(ctl[i], ctk[i], phi[i])
            if acc > 0 or accept:   
                res = s7_PDF(angles, coeff) * acc
            else:
                return 1
        return res
    elif hasattr(acc_poly, "__len__"):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = legendre_eval(acc_poly, ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s7_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc )
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = legendre_eval(acc_poly, ctl, ctk, phi)
            if acc > 0 or accept:
                res = s7_PDF(angles, coeff) * acc
            else:
                res = 1
        return res



def s8_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s8]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi)
        
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    
    res = []; angles = [ctl, ctk, phi]
    # need to loop for the case when coeff is an array of arrays.
    ctl = angles[0]
    if callable(acc_poly):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = acc_poly(ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s8_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc)
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = acc_poly(ctl, ctk, phi)
            if acc > 0 or accept:   
                res = s8_PDF(angles, coeff) * acc
            else:
                return 1
        return res
    elif hasattr(acc_poly, "__len__"):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = legendre_eval(acc_poly, ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s8_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc )
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = legendre_eval(acc_poly, ctl, ctk, phi)
            if acc > 0 or accept:
                res = s8_PDF(angles, coeff) * acc
            else:
                res = 1
        return res



def s9_acc_PDF_NotNormal(ctl, ctk, phi, coeff, acc_poly, accept = False):
    """
    Parameters
    ----------
    angles : list with 3 lists of observables, in order ctl, ctk, phi.
    
    coeffs : [fl, afb, s3, s9]
        Lists of the coefficients, ordering defined as 
            fl, afb, s3, s4, s5, s7, s8, s9
    
    acc_poly : numpy polynomial of 3 variables (ctl, ctk, phi)
        
    Returns
    -------
    res : ndarray
        Not Normalised Probability Distribution for observing trasnformed angles
        with Si isolated. 
    """
    
    res = []; angles = [ctl, ctk, phi]
    # need to loop for the case when coeff is an array of arrays.
    if callable(acc_poly):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = acc_poly(ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s9_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc)
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = acc_poly(ctl, ctk, phi)
            if acc > 0 or accept:   
                res = s9_PDF(angles, coeff) * acc
            else:
                res = 1
        return res
    elif hasattr(acc_poly, "__len__"):
        if hasattr(ctl, "__len__"):
            for i in range(len(ctl)):
                acc = legendre_eval(acc_poly, ctl[i], ctk[i], phi[i])
                if acc > 0 or accept:
                    res.append(s9_PDF([ctl[i], ctk[i], phi[i]], coeff) * acc )
            res = np.array(res)
            if not np.any(res):
                return float("nan")
        else:
            acc = legendre_eval(acc_poly, ctl, ctk, phi)
            if acc > 0 or accept:
                res = s9_PDF(angles, coeff) * acc
            else:
                res = 1
        return res







