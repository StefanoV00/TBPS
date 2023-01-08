# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:17:43 2022

@author: Stefano
"""

import numpy as np
import matplotlib.pyplot as pl
from uncertainties import ufloat
from scipy.integrate import quad, tplquad
from tqdm import tqdm
import random as rand

from find_coeff_numpy import legendre_eval, legendre_eval_project_1D
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance


###############################################################################
#%% FUNCTIONS FOR USER
###____________________________________________________________________________

def count (bindata, acceptance, avg_acc = True, norm = True, n=1, test = True,
           crystalball = 0, modify = 0):
    """
    Extract Observables by Means of Counting Method
    
    Parameters
    ----------
    bindata : list
        List of Dataframes, one per bin
    acceptance : list
        List of acceptnce functions, one per bin
    avg_acc : Bool
        If True(default) find average acceptance per bin by integrating, 
        otherwise just take value at midpoint (muuuuch faster).
    n : float
        The intermediate number of bins used is the multiple of 4 which
        is closest to n sqrt(len(ctl)).

    Returns
    -------
    result : list
        [ [afb_count , afb_count_std],   \n
          [s4_count  , s4_count_std ],   \n
          [s5_count,   s5_count_std ],   \n
          [s7_count,   s7_count_std ],   \n
          [s8_count,   s8_count_std ] ]  \n
    specifics : list
        specifics = [avg_acc, norm, n]
    """

    afb_count = []; afb_count_std = []
    s4_count  = []; s4_count_std  = []
    s5_count  = []; s5_count_std  = []
    s7_count  = []; s7_count_std  = []
    s8_count  = []; s8_count_std  = []
    
    for i in tqdm(range(len(bindata)), "Counting Bins"):
        I, Istds = angles_bin(bindata[i], acceptance[i], avg_acc, norm, n, 
                              crystalball = 0)
        
        afb, afb_std = count_afb(I, Istds)
        afb_count.append(afb); afb_count_std.append(afb_std)
        
        s4, s4_std = count_s4(I, Istds)
        s4_count.append(s4); s4_count_std.append(s4_std)
        
        s5, s5_std = count_s5(I, Istds)
        s5_count.append(s5); s5_count_std.append(s5_std)
        
        s7, s7_std = count_s7(I, Istds)
        s7_count.append(s7); s7_count_std.append(s7_std)
        
        s8, s8_std = count_s8(I, Istds)
        s8_count.append(s8); s8_count_std.append(s8_std)
    
    res = [ [afb_count , afb_count_std],
             [s4_count  , s4_count_std ],
             [s5_count,   s5_count_std ],
             [s7_count,   s7_count_std ],
             [s8_count,   s8_count_std ] ]  
    specifics = [avg_acc, norm, n]

    if not test:
        try:
            np.save(f"fitting/Counting{crystalball}_Min_{avg_acc}_mod{modify}.npy",
                np.array(res), allow_pickle = True)
        except FileNotFoundError:
           np.save(f"Counting{crystalball}_Min_{avg_acc}_mod{modify}.npy",
                np.array(res), allow_pickle = True)
    return res, specifics


def plot_counting_results(count_all_res, true = 0, M = 5e4, 
                          crystalball = 0, modify = 0):
    count = count_all_res[0]
    spec = count_all_res[1] 
    avg_acc = spec[0]
    
    if len(spec) != 3:
        raise Exception ("Specifics must have length 3. Count_all_res should \
                         \nliterally be the output of count function.")
    
    rc = len(count); c = int(np.ceil(np.sqrt(rc))); r = int( np.ceil( rc / c) )
    figsize = (c * 4, r * 3)
    pl.figure(figsize = figsize, tight_layout = True)
    if true:
        pl.suptitle(f"Counting Method Test ({M} Datapoints)")
    else:
        pl.suptitle("Counting Method Fit")
    tags = [r"$A_{fb}$", r"$S_4$", r"$S_5$", r"$S_7$", r"$S_8$"]
    bins = np.arange(0, len(count[0][0]))
    for i in range(len(count)):
        pl.subplot(r, c, i + 1)
        coeff = count[i][0]
        err   = count[i][1]
        pl.errorbar(bins, coeff, err, fmt = ".", capsize = 4, 
                    label = "Counting Results")
        if true:
            pl.plot(bins, true[i], "x", label = "True Values")
            pl.legend()
        pl.xlabel("Bins")
        pl.ylabel(tags[i])
        pl.ylim(min(-0.55, pl.ylim()[0]), max(0.55, pl.ylim()[1]))
        pl.grid(alpha = 0.4)
        
    if true:
        pl.savefig( f"TestPlots/Counting/CountingTest_M{M}_spec{spec}" )
    else:
        pl.savefig (f"MinPlots/Counting{crystalball}_Res_{avg_acc}_mod{modify}_spec{spec}")
    pl.show()

###############################################################################
#%% COUNTING BEHIND THE SCENES
###____________________________________________________________________________
def angles_bin(data, acceptance, avg_acc = True, norm = True, n = 1,
               crystalball = 0):
    """
    Parameters
    ----------
    data : pd.Dataframe
        Has costhetal, costhetak, phi as keys
    acceptance : callable
        Function of the 3 parameters, called as acceptance([ctl, ctk, phi])
    avg_acc : Bool
        If
        - True(default) find average acceptance per bin by integrating 
        - False just take value at midpoint (muuuuch faster).
        - 'auto', use acceptance to keep/reject datapoints to bin
    n : int
        The intermediate number of bins used is the multiple of 4 which
        is closest to n sqrt(len(ctl)).
        
    Returns
    -------
    Ifinal : ndarray, shape(4,4,4)
        Binning of angles modulated by acceptance function. Binedges are:
            [0, pi/2, pi, 3pi/2, 2pi] (last 2 empty)
            [0, pi/2, pi, 3pi/2, 2pi] (last 2 empty)
            [-pi, -pi/2, 0, pi/2, pi]
            
    Istds_final : ndarray, shape(4,4,4)
        Std on binning of angles modulated by acceptance function.

    """
    ###########################################################################
    #Rearrange data
    try:
        try:
            Bm = data["B0_MM"]
            tl = np.arccos(data["costhetal"])
            tk = np.arccos(data["costhetak"])
            phi = data["phi"]
        except KeyError:
            tl = np.arccos(data["costhetal"])
            tk = np.arccos(data["costhetak"])
            phi = data["phi"]
    except KeyboardInterrupt:
        raise KeyboardInterrupt 
    except:
        print("I should inform you the data was not a Dataframe, but don't \
              \n worry, I took care of it (assuming it was [ctl, ctk, phi]).")
        tl = np.arccos(data[0])
        tk = np.arccos(data[1])
        phi = data[2]
    
    
    # ##########################################################################
    # # Accout for Crystal Ball if required (as rejection method)
    # if not isinstance(crystalball, int) and not isinstance(crystalball, float):
    #     Bm_new = []
    #     tl_new = []
    #     tk_new = []
    #     phi_new = []
    #     cb_scan = crystalball(np.linspace(min(Bm), max(Bm), 100))
    #     C = max(cb_scan)
    #     for i in range(len(Bm)):
    #         p = rand.random() * C
    #         if cb_scan(Bm[i]) > p:
    #             Bm_new.append(Bm[i])
    #             tl_new.append(tl[i])
    #             tk_new.append(tk[i])
    #             phi_new.append(phi_new[i])
    #     Bm  = np.array(Bm_new)
    #     tl  = np.array(tl_new)
    #     tk  = np.array(tk_new)
    #     phi = np.array(phi_new)
    
                
    data = [tl, tk, phi]
    ###########################################################################
    # Bin in "smaller bins", more accurate acceptance modulation
    myrange = [ (0, np.pi), (0, np.pi), (-np.pi, np.pi) ]
    if n == 0:
        N = 8
    else:
        N = int(n*np.cbrt(len(tl)) / 4) * 4 #multiple of 4 approx n*sqrt(len(data))
    binned , edge = np.histogramdd(data, (N, N, N), myrange, False)
    stds = np.sqrt(binned)
    if norm:
        norm_binned   = np.histogramdd(data, (N, N, N), myrange, True)[0]
        A = norm_binned / binned
        norm_stds = stds*A
    else:
        norm_binned = binned
        norm_stds = stds
    
    
    ###########################################################################
    # Modulate by acceptance function (smaller bins = more accurate)
    try:
        if acceptance == 1:
            I = norm_binned
            Istd = norm_stds
    except:
        pass
    if callable(acceptance):
        if avg_acc == "auto" or avg_acc =="Auto":
            ctl_acc = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,  0, 0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1, 1, 1, 1])
            ctk_acc = np.array([-1,-1,-1, 0, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 0, 1, 1, 1])
            phi_acc = np.array([-1, 0, 1,-1, 0, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 0, 1,-1, 0, 1])
            C = max(acceptance(ctl_acc, ctk_acc, phi_acc * np.pi))
            p = acceptance(np.cos(tl), np.cos(tk), phi)
            r = np.random.random(len(tl)) * C
            tl, tk, phi  = tl[p>=r], tk[p>=r], phi[p>=r]
            data = [tl, tk, phi]

        elif avg_acc:
            A = 2 * np.pi**3 / N**3
            I =  np.zeros((N,N,N)); Istd =  np.zeros((N,N,N))
            for i in tqdm(range(N), f"Modulated by Acceptance, i"):
                for j in range(N):
                    for k in range(N):
                        #Get average acceptance function in interval
                        V = tplquad(acceptance, edge[2][k],         edge[2][k+1], 
                                                np.cos(edge[1][j]), np.cos(edge[1][j+1]),
                                                np.cos(edge[0][i]), np.cos(edge[0][i+1]),
                                                epsabs = 1e-1, epsrel = 1e-1)[0]
                        avg_acc = V/A
                        # Divide
                        I[i,j,k]     = norm_binned[i,j,k] / avg_acc
                        Istd[i,j,k]  = norm_stds[i,j,k] / avg_acc
        else:
            I =  np.zeros((N,N,N)); Istd =  np.zeros((N,N,N))
            for i in tqdm(range(N), f"Modulated by Acceptance, i"):
                ctl_avg = (np.cos(edge[0][i+1]) - np.cos(edge[0][i])) / 2
                for j in range(N):
                    ctk_avg = (np.cos(edge[1][j+1]) - np.cos(edge[1][j])) / 2
                    for k in range(N):
                        phi_avg = (edge[2][k+1] - edge[2][k]) / 2
                        
                        avg_acc = acceptance(ctl_avg, ctk_avg, phi_avg)
                        # Divide
                        I[i,j,k]     = norm_binned[i,j,k] / avg_acc
                        Istd[i,j,k]  = norm_stds[i,j,k] / avg_acc
                        
    elif hasattr(acceptance, "__len__"):
        if avg_acc == "auto" or avg_acc =="Auto":
            ctl_acc = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,  0, 0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1, 1, 1, 1])
            ctk_acc = np.array([-1,-1,-1, 0, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 0, 1, 1, 1])
            phi_acc = np.array([-1, 0, 1,-1, 0, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 0, 1,-1, 0, 1])
            C = max(legendre_eval(acceptance, ctl_acc, ctk_acc, phi_acc * np.pi))
            p = legendre_eval(acceptance, np.cos(tl), np.cos(tk), phi)
            r = np.random.random(len(tl)) * C
            tl, tk, phi  = tl[p>=r], tk[p>=r], phi[p>=r]
            data = [tl, tk, phi]
            
        elif avg_acc:
            def myacceptance (ctl, ctk, phi):
                return legendre_eval(acceptance, ctl, ctk, phi)
            A = 2 * np.pi**3 / N**3
            I =  np.zeros((N,N,N)); Istd =  np.zeros((N,N,N))
            for i in tqdm(range(N), f"Modulated by Acceptance, i"):
                for j in range(N):
                    for k in range(N):
                        #Get average acceptance function in interval
                        V = tplquad(myacceptance, edge[2][k],        edge[2][k+1], 
                                                np.cos(edge[1][j]), np.cos(edge[1][j+1]),
                                                np.cos(edge[0][i]), np.cos(edge[0][i+1]),
                                                epsabs = 1e-1, epsrel = 1e-1)[0]
                        avg_acc = V/A
                        # Divide
                        I[i,j,k]     = norm_binned[i,j,k] / avg_acc
                        Istd[i,j,k]  = norm_stds[i,j,k] / avg_acc
        else:
            I =  np.zeros((N,N,N)); Istd =  np.zeros((N,N,N))
            for i in tqdm(range(N), f"Modulated by Acceptance, i"):
                ctl_avg = (np.cos(edge[0][i+1]) - np.cos(edge[0][i])) / 2
                for j in range(N):
                    ctk_avg = (np.cos(edge[1][j+1]) - np.cos(edge[1][j])) / 2
                    for k in range(N):
                        phi_avg = (edge[2][k+1] - edge[2][k]) / 2
                        
                        avg_acc = legendre_eval(acceptance, ctl_avg, ctk_avg, phi_avg)
                        # Divide
                        I[i,j,k]     = norm_binned[i,j,k] / avg_acc
                        Istd[i,j,k]  = norm_stds[i,j,k] / avg_acc
     
        
    ###########################################################################
    #Rearrange in our bins
    # bin_tl = np.array([0, 1/2, 1, 3/2, 2]) * np.pi (last 2 will be empty)
    # bin_tk = np.array([0, 1/2, 1], 3/2, 2) * np.pi (last 2 will be empty)
    # bin_phi = np.array([-1, -1/2, 0, 1/2, 1]) * np.pi
    if avg_acc == "auto" or avg_acc =="Auto":
        myrange = [ (0, 2*np.pi), (0, 2*np.pi), (-np.pi, np.pi) ]
        N = 4
        binned , edge = np.histogramdd(data, (N, N, N), myrange, False)
        stds = np.sqrt(binned)
        if norm:
            Ifinal   = np.histogramdd(data, (N, N, N), myrange, True)[0]
            A = Ifinal / binned
            Istds_final = stds*A
        else:
            Ifinal = binned
            Istds_final = stds
        
    else:
        Ifinal = np.zeros((4,4,4)); Istds_final = np.zeros((4,4,4))
        ntl = N/2
        ntk = N/2
        nph = N/4
        if nph != int(nph):
            raise Exception("nph != int(nph)")
        for i in range(2):
            for j in range(2):
                for k in range(4):#tqdm(range(4), "Put Together, i={i},j={j},k"):
                    # x0 = int((i)*ntl); x1 = int((i+1)*ntl)
                    # y0 = int((j)*ntk); y1 = int((j+1)*ntk)
                    # z0 = int((k)*nph); z1 = int((k+1)*nph)
                    # Ifinal[i,j,k] += np.nansum(I[x0:x1][y0:y1][z0:z1])
                    # Istds_final[i,j,k] += np.nansum(Istd[x0:x1][y0:y1][z0:z1])
                    for iprime in range(int(i*ntl), int((i+1)*ntl) ):
                        for jprime in range(int(j*ntk), int((j+1)*ntk) ):
                            for kprime in range(int(k*nph), int((k+1)*nph) ):
                                if not np.isnan(Istd[iprime, jprime, kprime]):
                                    a = ufloat(Ifinal[i,j,k], Istds_final[i,j,k])
                                    b = ufloat(I[iprime,jprime,kprime], 
                                               Istd[iprime,jprime,kprime])
                                    c = a + b
                                    Ifinal[i,j,k] = c.n
                                    Istds_final[i,j,k] = c.s
                                    # Istds_final[i,j,k] += Istd[iprime, jprime, kprime]
                                    # Ifinal[i,j,k]      += I[iprime, jprime, kprime]
                                # else:
                                #     Ifinal[i,j,k] += I[iprime, jprime, kprime]
                                    
        Ifinal = Ifinal * 16 / N**3
        Istds_final = Istds_final * 16 / N**3
        
    return Ifinal, Istds_final



def count_afb (I, Istds):
    Stot = np.nansum(I)
    Stot_std = np.nansum(Istds)
    
    Splus = np.nansum(I[0])
    Splus_std = np.nansum(Istds[0])
    Sminus = np.nansum(I[1])
    Sminus_std = np.nansum(Istds[1])
    Splus_u = ufloat(Splus, Splus_std)
    Sminus_u = ufloat(Sminus, Sminus_std)
    
    afb = (Splus_u - Sminus_u) / (Splus_u + Sminus_u)
    return afb.n, afb.s


def count_s4 (I, Istds):
    Stot = np.nansum(I)
    Stot_std = np.nansum(Istds)
    
    a         =  I[0][0][1]
    a_std =  Istds[0][0][1]
    b         =  I[0][0][2]
    b_std =  Istds[0][0][2]
    c          = I[0][1][0]
    c_std  = Istds[0][1][0]
    d          = I[0][1][3]
    d_std  = Istds[0][1][3]
    a = ufloat(a, a_std)
    b = ufloat(b, b_std)
    c = ufloat(c, c_std)
    d = ufloat(d, d_std)
    
    e         = I[1][0][0]
    e_std = Istds[1][0][0]
    f         = I[1][0][3]
    f_std = Istds[1][0][3]
    g         = I[1][1][1]
    g_std = Istds[1][1][1]
    h         = I[1][1][2]
    h_std = Istds[1][1][2]
    e = ufloat(e, e_std)
    f = ufloat(f, f_std)
    g = ufloat(g, g_std)
    h = ufloat(h, h_std)
    
    Stot_u = ufloat(Stot, Stot_std)
    Splus_u = a+b+c+d +e+f+g+h
    Sminus_u = Stot_u - Splus_u
    
    s4 = np.pi/2 * (Splus_u - Sminus_u) / Stot_u
    return s4.n, s4.s


def count_s5 (I, Istds):
    Stot = np.nansum(I)
    Stot_std = np.nansum(Istds)
    
    #For this one, integrate over ctl:
    I_tl = np.nansum(I, axis = 0)
    Istds_tl = np.nansum(Istds, axis = 0) 
    
    a         = I_tl[0][1]
    a_std = Istds_tl[0][1]
    b         = I_tl[0][2]
    b_std = Istds_tl[0][2]
    a = ufloat(a, a_std)
    b = ufloat(b, b_std)
    
    c         = I_tl[1][0]
    c_std = Istds_tl[1][0]
    d         = I_tl[1][3]
    d_std = Istds_tl[1][3]
    c = ufloat(c, c_std)
    d = ufloat(d, d_std)
    
    Stot_u = ufloat(Stot, Stot_std)
    Splus_u = a+b +c+d
    Sminus_u = Stot_u - Splus_u
    
    s5 = 4/3 *(Splus_u - Sminus_u) / Stot_u
    return s5.n, s5.s


def count_s7 (I, Istds):
    Stot = np.nansum(I)
    Stot_std = np.nansum(Istds)
    
    #For this one, integrate over ctl:
    I_tl = np.nansum(I, axis = 0)
    Istds_tl = np.nansum(Istds, axis = 0) 
    
    #Then compute
    a         = I_tl[0][2]
    a_std = Istds_tl[0][2]
    b         = I_tl[0][3]
    b_std = Istds_tl[0][3]
    a = ufloat(a, a_std)
    b = ufloat(b, b_std)
    
    c         = I_tl[1][0]
    c_std = Istds_tl[1][0]
    d         = I_tl[1][1]
    d_std = Istds_tl[1][1]
    c = ufloat(c, c_std)
    d = ufloat(d, d_std)
    
    Stot_u = ufloat(Stot, Stot_std)
    Splus_u = a+b +c+d
    Sminus_u = Stot_u - Splus_u

    s7 = 4/3 * (Splus_u - Sminus_u) / Stot_u
    return s7.n, s7.s


def count_s8 (I, Istds):
    Stot = np.nansum(I)
    Stot_std = np.nansum(Istds)
    Stot_u = ufloat(Stot, Stot_std)
    
    a     =     I[0][0][2]
    a_std = Istds[0][0][2]
    b     =     I[0][0][3]
    b_std = Istds[0][0][3]
    c     =     I[0][1][0]
    c_std = Istds[0][1][0]
    d     =     I[0][1][1]
    d_std = Istds[0][1][1]
    a = ufloat(a, a_std)
    b = ufloat(b, b_std)
    c = ufloat(c, c_std)
    d = ufloat(d, d_std)
    
    e     =     I[1][0][0]
    e_std = Istds[1][0][0]
    f     =     I[1][0][1]
    f_std = Istds[1][0][1]
    g     =     I[1][1][2]
    g_std = Istds[1][1][2]
    h     =     I[1][1][3]
    h_std = Istds[1][1][3]
    e = ufloat(e, e_std)
    f = ufloat(f, f_std)
    g = ufloat(g, g_std)
    h = ufloat(h, h_std)
    
    Stot_u = ufloat(Stot, Stot_std)
    Splus_u = a+b+c+d +e+f+g+h
    Sminus_u = Stot_u - Splus_u
    s8 = np.pi/2 *(Splus_u - Sminus_u) / Stot_u
    return s8.n, s8.s

# #%% GENERAL TEST

# ctl = [ 0.99,  0.99,  0.99, #3
#         0.80,  0.80,  0.80, #6
#         0.40,  0.45,        #8
#         0.02,  0.04,  0.05,  0.14,  0.16,  0.18,  0.19, 0.30,  0.35, #17   
#        -0.33, -0.35,                      #19
#        -0.50, -0.65, -0.61, -0.43, -0.49, #24
       
#        -0.99, -0.91, -0.90, -0.95, -0.95, -0.95] #30
# ctl = np.array(ctl)

# ctk = [ 0.99,  0.99,  0.99,  0.99,  0.99, #5
#         0.80,    #6
#         0.40,  0.45,  0.47,  0.49,  0.67, #11
#         0.08,  0.30,  0.35,               #14
#        -0.33, -0.35,                      #16
       
       
#        -0.99, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98]
# ctk = np.array(ctk)

# phi = [-0.99, -0.81, -0.76, -0.98, -0.83, #5
#        -0.72, -0.74, -0.67, -0.65, -0.61,  -0.60, -0.72, -0.57,  #13
#        -0.49, -0.33, -0.35,  #16
#        -0.08, -0.07, -0.09, -0.06, #20  
#         0.07,  0.03,  0.02,  0.16, #24
#         0.30,  0.35,  0.40,  0.45, #28
        
#         0.81,  0.94] #30

# phi = np.array(phi) * np.pi
# N = np.sqrt(len(phi))
# tl = np.arccos(ctl)
# tk = np.arccos(ctk)
# data = [tl, tk, phi]
# print(tl)
# print(tk)

# # Bin in "smaller bins", more accurate acceptance modulation
# n = 1
# n = 4*n
# myrange = [ [0, 2*np.pi], [0, 2*np.pi], [-np.pi, np.pi] ]
# N = 4#int( np.sqrt(len(tl)) / n) * int(n) #multiple of 4approximately sqrt(len(data))
# binned , edge = np.histogramdd(data, (N, N, N), myrange, False)
# norm_binned   = np.histogramdd(data, (N, N, N), myrange, True)[0]
# print(np.sum(binned))
# print(np.sum(binned, axis = 0))
# print(np.sum(binned, axis = 1))
# print(np.sum(binned, axis = 2))
# print(np.sum(np.sum(binned, axis = 0), axis = 0))
# V = 2* np.pi**3 / N**3 
# finalresult = norm_binned
# A = norm_binned / binned
# print(np.nansum(A))
# stds = np.sqrt(binned)
# norm_stds = stds*A
# finalstd = norm_stds
# finalresult = binned*1
# finalstd = stds*1

# n = 1
# n = 4*n
# myrange = [ [0, np.pi], [0, np.pi], [-np.pi, np.pi] ]
# N = 8#int( np.sqrt(len(tl)) / n) * int(n) #multiple of 4approximately sqrt(len(data))
# binned , edge = np.histogramdd(data, (N, N, N), myrange, False)
# norm_binned   = np.histogramdd(data, (N, N, N), myrange, True)[0]
# print(np.sum(binned))
# print(np.sum(binned, axis = 0))
# print(np.sum(binned, axis = 1))
# print(np.sum(binned, axis = 2))
# print(np.sum(np.sum(binned, axis = 0), axis = 0))
# V = 2* np.pi**3 / N**3 
# A = norm_binned / binned
# print(np.nansum(A))
# stds = np.sqrt(binned)
# norm_stds = stds*A

# # Modulate by acceptance function (smaller bins = more accurate)
# I =  np.zeros((N,N,N))
# Istd =  np.zeros((N,N,N))
# for i in range(N):
#     for j in range(N):
#         for k in range(N):
#             #Get average acceptance function in interval
#             # V = tplquad(acceptance, edge[0][i], edge[0][i+1], 
#             #                         edge[1][i], edge[1][i+1],
#             #                         edge[2][i], edge[2][i+1],
#             #                         epsabs = 1e-5, epsrel = 1e-5)
#             # A = 2 * np.pi**3 / N**3
#             # avg_acc = V/A
#             # Divide
#             I[i,j,k]     = norm_binned[i,j,k]# / avg_acc
#             Istd[i,j,k]  = norm_stds[i,j,k] #/ avg_acc
# #Rearrange in our bins
# bin_tl = np.array([0, 1/2, 1]) * np.pi
# bin_tk = np.array([0, 1/2, 1]) * np.pi
# bin_phi = np.array([-1, -1/2, 0, 1/2, 1]) * np.pi

# Ifinal = np.zeros((4,4,4))
# Istds_final = np.zeros((4,4,4))
# ntl = N/2
# ntk = N/2
# nph = N/4
# if nph != int(nph):
#     raise Exception("nph != int(nph)")
# for i in range(2):
#     for j in range(2):
#         for k in range(4):
#             for iprime in range(int(i*ntl), int((i+1)*ntl) ):
#                 for jprime in range(int(j*ntk), int((j+1)*ntk) ):
#                     for kprime in range(int(k*nph), int((k+1)*nph) ):
#                         Ifinal[i,j,k]      += I[iprime, jprime, kprime]
#                         if not np.isnan(Istd[iprime, jprime, kprime]):
#                             Istds_final[i,j,k] += Istd[iprime, jprime, kprime]
# Ifinal = Ifinal * 16 / N**3
# Istds_final = Istds_final * 16 / N**3
# #%%%
# def const(x):
#     return 1
# print(finalresult[1])
# print(np.nansum(finalresult[0][0][1:4]))
# print(count_afb([ctl, ctk, phi], const))















