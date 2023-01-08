# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:35:58 2022

@author: Stefano
"""

##############################################################################
# HERE SELECT THE INDEX OF CRBALL (2, 3, 4, 5)
# and some minimisation settings (recommend those here)
modify = 0
crball = "11" # 12, 13, 14 etc... till 39

###############################################################################
###############################################################################
#REQUIREMENTS
# 
#______________________________________________________________________________
#.PY FILES (saved in same folder)
#
# From fitting/fitting3.0:
#TBPS_bin_functions
#TBPS_pdfs1
#TBPS_pdfs2
#TBPS_pdf_full
#TBPS_fit_functions
#TBPS_counting
#
#From acceptance:
#find_coeff_numpy
#
#______________________________________________________________________________
#DATA FILES FROM GITHUB
# FROM CLASSIFICATION/FINAL_DATA_PROCESSED/CRYSTAL_BALL_SLECTIONS/
#f'total_dataset_classified{crball}.pkl'
# PUT YOUR PATH:
path = f'../total_dataset_classified{crball}.pkl'
# FROM ACCEPTANCE/OUTPUT
#"acceptance_legendre_coeffs.npz"
#"acceptance_legendre_std.npy"
# FROM FITTING/FITTING3.0/FITTING
#f"proj0_fls_scipy[0,2]_mod0.npy"
#f"proj0_afbs_scipy[0,2]_mod0.npy"
#f"proj0_s3s_scipy[0,2]_mod0.npy"
#f"proj0_aims_scipy[0,2]_mod0.npy"
#
#______________________________________________________________________________
#ADDITIONAL PACKAGE YOU MAY NOT HAVE
#conda install -c conda-forge uncertainties
###############################################################################

from TBPS_bin_functions import *
from TBPS_pdfs1 import *
from TBPS_pdfs2 import *
from TBPS_pdf_full import *
from TBPS_fit_functions import *
from TBPS_counting import *
from find_coeff_numpy import legendre_eval, legendre_eval_project_1D, acc_modify
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance


import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import iqr
import pandas as pd
from iminuit import Minuit

import matplotlib.pyplot as plt
from tqdm import tqdm
#from uncertainties import ufloat
#from uncertainties import umath as um

ps = {#"text.usetex": True,
      "font.size" : 16,
      "font.family" : "Times New Roman",
      "axes.labelsize": 15,
      "legend.fontsize": 10,
      "xtick.labelsize": 13,
      "ytick.labelsize": 13,
      "figure.figsize": [7.5, 6],
      "mathtext.default": "default"
       }
plt.rcParams.update(ps)
del ps


classdata = pd.read_pickle(path)
data = classdata.reset_index()

mybins = [( 0.1,   0.98),
          ( 1.1,  2.5),
          ( 2.5,  4.0),
          ( 4.0,  6.0),
          ( 6.0,  8.0),
          (15.0, 17.0),
          (17.0, 19.0),
          (11.0, 12.5),
          ( 1.0,  6.0),
          (15.0, 17.9)] 
bindata = q2bin(data, mybins)

path = "acceptance/acceptance_legendre_coeffs.npz"
acc_stuff = load_acceptance(path)
file = acc_stuff[-1]
acceptance_std = np.load("acceptance/acceptance_legendre_std.npy")
del path, classdata

# Use coeffciients, faster:
acceptance_all = []
for i in range(len(mybins)):
    acceptance_all.append(file[f"{i}"])
if modify:
    try:
        a = acceptance_std
        del a
        acceptance_all = acc_modify(acceptance_all, acceptance_std, modify)
    except NameError:
        modify = 0
    del i

 
# TRY LOADING PREVIOUS RESULTS
check1 = 0
check2 = 0
for i in [[0,1,2], [0,2], [0,1], [1,2],[2],[1],[0], 1, True, 0, False, "{want_scipy}"]:
    if not check1:
        try:
            ws = i
            fls,  fl_errs  = np.load(f"fitting/proj0_fls_scipy{ws}_mod0.npy")
            afbs, afb_errs = np.load(f"fitting/proj0_afbs_scipy{ws}_mod0.npy")
            check1 = 1
        except FileNotFoundError:
            pass 
    if not check2:
        try:
            ws = i
            s3s, s3_errs   = np.load(f"fitting/proj{crball}_s3s_scipy{ws}_mod{modify}.npy")
            aims, aim_errs = np.load(f"fitting/proj{crball}_aims_scipy{ws}_mod{modify}.npyy")
            check2 = 1
        except FileNotFoundError:
            continue
del check1, check2   
    
###############################################################################
# FIT CTL and CTK
###############################################################################
# Pick Function
f = [ctl_acc_PDF_NotNormal, ctk_acc_PDF_NotNormal]
args = []
for i in range(len(mybins)):
    args.append([ acceptance_all[i], acceptance_all[i] ])
try:
    guess = np.column_stack([fls, afbs])
except NameError:
    guess = get_FlAfb_guesses(bindata, acceptance_all, N = 20, plot = False)
#Set Limits
limits = [ [ (0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)] ]


# Minimisation Parameters
Nlayers = 2
want_scipy = [0]
if want_scipy:
    Nlayers += 1
guess_per_bin = False
limits_per_bin = False
if len(guess) == len(mybins):
    guess_per_bin = True
if hasattr(limits, "__len__"):
    if len(limits) == len(mybins):
        limits_per_bin = True   
(fls, afbs), (fl_errs, afb_errs), fits_c = fit_angs(bindata, f, guess, 
                                                guess_per_bin = guess_per_bin, 
                                                limits = limits,
                                                limits_per_bin = limits_per_bin,
                                                Nlayers = Nlayers, vary = 0.01,
                                                args = args, 
                                                want_scipy = want_scipy)
del limits, guess_per_bin, limits_per_bin


###############################################################################
# FIT PHI
###############################################################################
f = phi_acc_PDF_NotNormal
want_scipy = [0]
try:
    guess = np.column_stack([s3s, aims])
    want_scipy = 0
except NameError:
    guess = [0,0]
limits = 0
args = acceptance_all
Nlayers = 2
if want_scipy:
    Nlayers += 1
guess_per_bin = False
limits_per_bin = False
if len(guess) == len(mybins):
    guess_per_bin = True
if hasattr(limits, "__len__"):
    if len(limits) == len(mybins):
        limits_per_bin = True        
(s3s, aims), (s3_errs, aim_errs), fits_p = fit_angs(bindata, f, guess, 
                                                guess_per_bin = guess_per_bin, 
                                                limits = limits,
                                                limits_per_bin = limits_per_bin,
                                                Nlayers = Nlayers, vary = 0.01,
                                                args = args, 
                                                want_scipy = want_scipy)
del limits, guess_per_bin, limits_per_bin
  
###############################################################################
# COUNTING METHOD
###############################################################################

try: 
    specifics = ["auto", 1, 1]
    count_res = np.load(f"fitting/Counting{crball}_Min_auto_mod{modify}.npy")
    #plot_counting_results([count_res, specifics])
except FileNotFoundError:
    count_res, specifics = count(bindata, acceptance_all, avg_acc = "auto", 
                                 norm = 1, n = 0, test = False,
                                 crystalball = crball, modify = modify)
    #plot_counting_results([count_res, specifics], 
    #                      crystalball = crball, modify = modify)
afbs_c = count_res[0][0]; afbstd_c= count_res[0][1]
s4_c   = count_res[1][0]; s4std_c = count_res[1][1]
s5_c   = count_res[2][0]; s5std_c = count_res[2][1] 
s7_c   = count_res[3][0]; s7std_c = count_res[3][1] 
s8_c   = count_res[4][0]; s8std_c = count_res[4][1] 

###############################################################################
# FOLDING MIIMISATION
###############################################################################
# NOTE: all take parameters [fl, afb, s3, si]
# the first 2 to keep fixed

aloneguess = np.zeros(len(mybins))
sguess = np.column_stack([fls, afbs, s3s, aloneguess])
s9guess = np.column_stack([fls, afbs, s3s, aims])
limits = [[0,1], [-1, 1], [-1, 1], [-1, 1]]

want_scipy = 0
Nlayers = 2
if want_scipy:
    Nlayers += 1
guess_per_bin = False
limits_per_bin = False
if len(sguess) == len(mybins):
    guess_per_bin = True
if hasattr(limits, "__len__"):
    if len(limits) == len(mybins):
        limits_per_bin = True

s3_res = [s3s]
s3_errs = [s3_errs]
s4s = []
s4_errs = []
s5s = []
s5_errs = []
s6s = []
s6_errs = []
s7s = []
s7_errs = []
s8s = []
s8_errs = []
s9s = []
s9_errs = []
for i in [4, 5, 7, 8]:
    f = locals()[f"s{i}_acc_PDF_forfit"]
    print("Fitting", f.__name__ )
    # if i == 9:
    #     sguess = s9guess
    (useless1, useless2, s3s, sis), (useless3, useless4, s3s_errs, si_errs), fits =\
                                                fit_angs(bindata, f, sguess, 
                                                guess_per_bin = guess_per_bin, 
                                                limits = limits,
                                                limits_per_bin = limits_per_bin,
                                                Nlayers = Nlayers, vary = 0.01,
                                                args = acceptance_all,
                                                fixtrack = 0,
                                                want_scipy = want_scipy)
    s3_res.append(s3s)
    s3_errs.append(s3s_errs)
    locals()[f"s{i}s"], locals()[f"s{i}_errs"] = sis, si_errs   

del aloneguess, limits, guess_per_bin, limits_per_bin, Nlayers

results = [fls]
results.append(np.average([afbs, afbs_c], 
                           weights = 1/np.array([afb_errs, afbstd_c]),
                           axis = 0))
try:
    results.append( np.average(s3_res, weights = 1./np.array(s3_errs), axis = 0) )
except TypeError:
    s3_errs[0] = s3_errs[0][0]
    results.append( np.average(s3_res, weights = 1./np.array(s3_errs), axis = 0) )

for i in [4,5,7,8]:
    sis = locals()[f"s{i}s"];  si_errs = locals()[f"s{i}_errs"]
    sic = locals()[f"s{i}_c"]; sic_errs = locals()[f"s{i}std_c"]
    results.append( np.average([sis, sic], 
                               weights = 1/np.array([si_errs, sic_errs]),
                                axis = 0))
results.append(aims) 
np.save(f"fitting/results{crball}_mod{modify}.npy", results)
