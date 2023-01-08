import numpy as np

def omega_s4_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of the observables
    coeffs: lists of the coefficients, ordering defined as 
            FL, Afb, aT, S3, S4, S5, S7, S8, S9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for the full decay rate w all using S4 coeff
    """
    
    ctl = angles[0]
    ctk = angles[1]
    phi = angles[2]

    fl = coeffs[0]
    afb = coeffs[1]     
    aT = coeffs[2]
    s4 = coeffs[4]

    P = []

    for i, phi_angle in enumerate(phi):
        theta_l = np.arccos(ctl[i])
        theta_k = np.arccos(ctk[i])

        if phi_angle < 0:
            phi[i] = -phi[i]
        
        if theta_l > np.pi/2:
            phi[i] = np.pi - phi[i]
            theta_l = np.pi - theta_l
        
        sq_ctk = ctk[i]*ctk[i]      # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared
        ctl_2 = np.cos(2*theta_l)   # cos(2*theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared

        subP = (3/4)*(1-fl)*(sq_stk) + fl*sq_ctk + (1/4)*(1-fl)*sq_stk*ctl_2 - fl*sq_ctk*ctl_2 \
            + (1/2)*(1-fl)*aT*sq_ctk*sq_ctl*np.cos(2*phi[i]) + s4*np.sin(2*theta_k)*np.sin(2*theta_l)*np.cos(phi[i])
        
        P.append((9/8*np.pi)*subP)
    
    return P

def omega_s5_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of the observables
    coeffs: lists of the coefficients, ordering defined as 
            FL, Afb, aT, S3, S4, S5, S7, S8, S9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    
    ctl = angles[0]
    ctk = angles[1]
    phi = angles[2]

    fl = coeffs[0]
    afb = coeffs[1]     
    aT = coeffs[2]
    s5 = coeffs[5]

    P = []

    for i, phi_angle in enumerate(phi):
        theta_l = np.arccos(ctl[i])
        theta_k = np.arccos(ctk[i])

        if phi_angle < 0:
            phi[i] = -phi[i]
        
        if theta_l > np.pi/2:
            theta_l = np.pi - theta_l
        
        sq_ctk = ctk[i]*ctk[i]      # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared
        ctl_2 = np.cos(2*theta_l)   # cos(2*theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared

        subP = (3/4)*(1-fl)*(sq_stk) + fl*sq_ctk + (1/4)*(1-fl)*sq_stk*ctl_2 - fl*sq_ctk*ctl_2 \
            + (1/2)*(1-fl)*aT*sq_ctk*sq_ctl*np.cos(2*phi[i]) + s5*np.sin(2*theta_k)*np.sin(theta_l)*np.cos(phi[i])
        
        P.append((9/8*np.pi)*subP)
    
    return P

def omega_s7_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of the observables
    coeffs: lists of the coefficients, ordering defined as 
            FL, Afb, aT, S3, S4, S5, S7, S8, S9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    
    ctl = angles[0]
    ctk = angles[1]
    phi = angles[2]

    fl = coeffs[0]
    afb = coeffs[1]     
    aT = coeffs[2]
    s7 = coeffs[6]

    P = []

    for i, phi_angle in enumerate(phi):
        theta_l = np.arccos(ctl[i])
        theta_k = np.arccos(ctk[i])

        if phi_angle > np.pi/2:
            phi[i] = np.pi-phi[i]
        
        if phi_angle < -np.pi/2:
            phi[i] = -np.pi - phi[i]

        if theta_l > np.pi/2:
            theta_l = np.pi - theta_l
        
        sq_ctk = ctk[i]*ctk[i]      # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared
        ctl_2 = np.cos(2*theta_l)   # cos(2*theta_l)
        sq_ctl = (1+ctl_2)/2        # cos(theta_l) squared

        subP = (3/4)*(1-fl)*(sq_stk) + fl*sq_ctk + (1/4)*(1-fl)*sq_stk*ctl_2 - fl*sq_ctk*ctl_2 \
            + (1/2)*(1-fl)*aT*sq_ctk*sq_ctl*np.cos(2*phi[i]) + s7*np.sin(2*theta_k)*np.sin(theta_l)*np.cos(phi[i])
        
        P.append((9/8*np.pi)*subP)
    
    return P

def omega_s8_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of the observables
    coeffs: lists of the coefficients, ordering defined as 
            FL, Afb, aT, S3, S4, S5, S7, S8, S9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    
    ctl = angles[0]
    ctk = angles[1]
    phi = angles[2]

    fl = coeffs[0]
    afb = coeffs[1]     
    aT = coeffs[2]
    s8 = coeffs[7]

    P = []

    for i, phi_angle in enumerate(phi):
        theta_l = np.arccos(ctl[i])
        theta_k = np.arccos(ctk[i])

        if phi_angle > np.pi/2:
            phi[i] = np.pi - phi[i]

        if phi_angle < -np.pi/2:
            phi[i] = -np.pi - phi[i]
        
        if theta_l > np.pi/2:
            theta_l = np.pi - theta_l
            theta_k = np.pi - theta_k
        
        sq_ctk = ctk[i]*ctk[i]      # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared
        ctl_2 = np.cos(2*theta_l)   # cos(2*theta_l)
        sq_stl = (1-ctl_2)/2        # sin(theta_l) squared

        subP = (3/4)*(1-fl)*(sq_stk) + fl*sq_ctk + (1/4)*(1-fl)*sq_stk*ctl_2 - fl*sq_ctk*ctl_2 \
            + (1/2)*(1-fl)*aT*sq_stk*sq_stl*np.cos(2*phi[i]) + s8*np.sin(2*theta_k)*np.sin(2*theta_l)*np.sin(phi[i])
        
        P.append((9/8*np.pi)*subP)
    
    return P

def omega_s9_PDF(angles, coeffs):
    """
    Parameters
    ----------
    angles : list with 3 lists of the observables
    coeffs: lists of the coefficients, ordering defined as 
            FL, Afb, aT, S3, S4, S5, S7, S8, S9
    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_l
    """
    
    ctl = angles[0]
    ctk = angles[1]
    phi = angles[2]

    fl = coeffs[0]
    afb = coeffs[1]     
    aT = coeffs[2]
    s3 = coeffs[3]
    s9 = coeffs[8]

    P = []

    for i, phi_angle in enumerate(phi):
        theta_k = np.arccos(ctk[i])

        if phi_angle < 0:
            phi[i] = np.pi + phi[i]
        
        sq_ctk = ctk[i]*ctk[i]      # cos(theta_k) squared
        sq_stk = 1-sq_ctk           # sin(theta_k) squared
        ctl_2 = 2*ctl[i]*ctl[i]-1   # cos(2*theta_l)
        sq_stl = (1-ctl_2)/2        # cos(theta_l) squared

        subP = (3/4)*(1-fl)*(sq_stk) + fl*sq_ctk + (1/4)*(1-fl)*sq_stk*ctl_2 - fl*sq_ctk*ctl_2 \
            + s3*sq_stk*sq_stl*np.cos(2*phi[i]) + (4/3)*afb*sq_stk*ctl[i] + s9*sq_stk*sq_stl*np.cos(2*phi[i])
        
        P.append((9/8*np.pi)*subP)
    
    return P