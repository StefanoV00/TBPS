# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:35:58 2022

@author: Stefano
"""

###############################################################################
modify = 0 #0, 1, -1 and any other integer
crball = 0
want_scipy = 0
Nlayers = 2
###############################################################################

from TBPS_bin_functions import *
from TBPS_fit_functions import *
from TBPS_pdfs1 import *
from TBPS_pdfs2 import *
from TBPS_pdf_full import *
from TBPS_counting import *
from find_coeff_numpy import legendre_eval, legendre_eval_project_1D, acc_modify
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance

import numpy as np
import pandas as pd
from scipy.integrate import quad
from iminuit import Minuit

import matplotlib.pyplot as plt
from tqdm import tqdm
from uncertainties import ufloat
from uncertainties import umath as um

ps = {"text.usetex": True,
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

try:
    path = f'classification/final_data_processed/total_dataset_classified{crball}.pkl'
    classdata = pd.read_pickle(path)
except FileNotFoundError:
    path = 'classification/final_data_processed/total_dataset_classified.pkl'
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
del path, classdata


# Use coeffciients, faster:
acceptance_all = []
for i in range(len(mybins)):
    acceptance_all.append(file[f"{i}"])
try:
    a = acceptance_std
    del a
    acceptance_all = acc_modify(acceptance_all, acceptance_std, modify)
except NameError:
    modify = 0
del i






#%% COUNTING METHOD
# RECALL count_res looks like this
# [ [afb_count , afb_count_std],   \n
#   [s4_count  , s4_count_std ],   \n
#   [s5_count,   s5_count_std ],   \n
#   [s7_count,   s7_count_std ],   \n
#   [s8_count,   s8_count_std ] ]  \n

try: 
    specifics = [0, 1, 1]
    count_res = np.load(f"fitting/Counting{crball}_Min_auto_mod{modify}.npy")
    plot_counting_results([count_res, specifics])
except FileNotFoundError:
    count_res, specifics = count(bindata, acceptance_all, avg_acc = "auto", 
                                 norm = True, n = 0, test = False,
                                 crystalball = crball, modify = modify)
    plot_counting_results([count_res, specifics], 
                          crystalball = crball, modify = modify)
afbs_c = count_res[0][0]; afbstd_c= count_res[0][1]
s4_c   = count_res[1][0]; s4std_c = count_res[1][1]
s5_c   = count_res[2][0]; s5std_c = count_res[2][1] 
s7_c   = count_res[3][0]; s7std_c = count_res[3][1] 
s8_c   = count_res[4][0]; s8std_c = count_res[4][1] 

# LOAD PREVIOUS RESULTS
check1 = 0
check2 = 0
for i in [[0,1,2], [0,2], [0,1], [1,2],[2],[1],[0], 1, True, 0, False, "{want_scipy}"]:
    if not check1:
        try:
            ws = i
            fls,  fl_errs  = np.load(f"fitting/proj{crball}_fls_scipy{ws}_mod{modify}.npy")
            afbs, afb_errs = np.load(f"fitting/proj{crball}_afbs_scipy{ws}_mod{modify}.npy")
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
if not check1:
    try:
        ws = [0,2]
        fls,  fl_errs  = np.load(f"fitting/proj_fls_scipy{ws}.npy")
        afbs, afb_errs = np.load(f"fitting/proj_afbs_scipy{ws}.npy")
    except:
        pass
if not check2:
    try:
        s3s, s3_errs   = np.load("fitting/proj_s3s_scipy{want_scipy}.npy")
        aims, aim_errs = np.load("fitting/proj_aims_scipy{want_scipy}.npy")
        check2 = 1
    except:
        pass
print("Check1, Check2 = ", check1, check2)
del check1, check2, ws, i


for i in range(len(mybins)):
    columns = [f"Bin {i}:{mybins[i]}GeV^2","Projection Estimate", "Projection Uncertainty"]
    content = [ ["Fl" , round(fls[i], 4) ,  round(fl_errs[i], 4)],
                ["Afb", round(afbs[i],4) , round(afb_errs[i], 4)],
                ["S3" , round(s3s[i], 4) ,  round(s3_errs[i], 4)],
                ["AIm", round(aims[i],4) , round(aim_errs[i], 4)] ]
    table_i = pd.DataFrame(content, columns = columns)
    print("\n\n\nCrystalBall#:",crball,"      mod#:", modify, "\n",table_i)
    
    columns = [f"Bin {i}:{mybins[i]}GeV^2","Counting Estimates", "Counting Uncertainty"]
    content = [ ["S4" , round(s4_c[i], 4) ,  round(s4std_c[i], 4)],
                ["S5" , round(s5_c[i], 4) ,  round(s4std_c[i], 4)],
                ["Afb", round(afbs_c[i],4), round(afbstd_c[i], 4)],
                ["S7" , round(s7_c[i], 4) ,  round(s7std_c[i], 4)],
                ["S8" , round(s8_c[i],4) ,   round(s8std_c[i], 4)] ]
    table_i = pd.DataFrame(content, columns = columns)
    print("\n",table_i)
    
del i, columns, content, table_i  


#%% MIIMISE
# NOTE: all take parameters [fl, afb, s3, si]
# the first 2 to keep fixed

aloneguess = np.zeros(len(mybins))
sguess = np.column_stack([fls, afbs, s3s, aloneguess])
s9guess = np.column_stack([fls, afbs, s3s, aims])
limits = 0
# limits = [[0,1], [-1, 1], [-1, 1], [-1, 1]]


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
    
    print("\nChecking on the Minimisation\n")
    print(fls)
    print(useless1)
    print(fl_errs)
    print(useless3)
    print(afbs)
    print(useless2)
    print(afb_errs)
    print(useless4)
    
    print("\n")
    print(sguess[:,2])
    print(s3s)
    print(sguess[:,3])
    print(sis)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    plt.suptitle(r"Results of Fitting Folded S"+f"{i}")
    ns = np.linspace(0, len(bindata) - 1, len(bindata))
    ax1.errorbar(ns, s3s, yerr=s3s_errs, fmt = ".", capsize = 4)
    ax2.errorbar(ns, sis, yerr=si_errs, fmt = ".", capsize = 4)
    ax1.grid(alpha = 0.3)
    ax2.grid(alpha = 0.3)
    ax1.set_ylabel(r'$S_3$')
    ax2.set_ylabel(f'$S_{i}$')
    ax1.set_xlabel(r'Bin number')
    ax2.set_xlabel(r'Bin number')
    ax1.set_ylim( [min(ax1.get_ylim()[0],-0.4), max(ax1.get_ylim()[1], 0.4)] )
    ax2.set_ylim( [min(ax2.get_ylim()[0],-0.4), max(ax2.get_ylim()[1], 0.4)] )
    plt.tight_layout()
    plt.savefig(f"MinPlots/S3_S{i}_Coeffs")
    plt.show()
    
    s3_res.append(s3s)
    s3_errs.append(s3s_errs)
    np.save(f"fitting/All{crball}_s{i}_scipy{want_scipy}_mod{modify}.npy", [sis, si_errs])
    locals()[f"s{i}s"], locals()[f"s{i}_errs"] = sis, si_errs
    
    del fig, ax1, ax2, ns
    
np.save(f"fitting/All{crball}_s3_scipy{want_scipy}_mod{modify}.npy", [s3_res, s3_errs])   
del aloneguess, limits, guess_per_bin, limits_per_bin, Nlayers


s3s_all_mean = np.average(s3_res, weights = 1/np.array(s3_errs), axis = 0)
s3s_mean_std = np.mean(s3_errs, axis = 0)
s3s_all_std = np.std(s3_res, axis = 0)
print(s3s_all_mean)
print(s3s_mean_std)
print(s3s_all_std)

#%%

for i in range(len(mybins)):
    columns = [f"Bin{i}:{mybins[i]}GeV^2","NLL Est", "NLL 1\u03C3 ", 
               "Counting Est", "Count 1\u03C3",
               "Mean", "Final 1\u03C3"]
    columns = [f"Bin {i}","NLL Est", "NLL 1\u03C3 ", 
                "Counting Est", "Count 1\u03C3",
                "Mean", "Final 1\u03C3"]
    content = []
    
    row = ["Fl",fls[i],        fl_errs[i],    "/",     "/", fls[i], fl_errs[i]]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    
    row = ["S3", s3s_all_mean[i], s3s_mean_std[i], "/",     "/",  s3s_all_mean[i], s3s_all_std[i]]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    s4_mean = np.average([s4s[i], s4_c[i]], weights = 1/np.array([s4_errs[i], s4std_c[i]]))
    s4_std = np.std([s4s[i], s4_c[i]])
    row = ["S4",    s4s[i],       s4_errs[i],   s4_c[i], s4std_c[i], s4_mean, s4_std]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    s5_mean = np.average([s5s[i], s5_c[i]], weights = 1/np.array([s5_errs[i], s5std_c[i]]))
    s5_std = np.std([s5s[i], s5_c[i]])
    row = ["S5",    s5s[i],       s5_errs[i],   s5_c[i], s5std_c[i], s5_mean, s5_std]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    afb_mean = np.average([afbs[i], afbs_c[i]], weights = 1/np.array([afb_errs[i], afbstd_c[i]]))
    afb_std = np.std([afbs[i], afbs_c[i]])
    row = ["Afb",    afbs[i],       afb_errs[i],   afbs_c[i], afbstd_c[i], afb_mean, afb_std]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    s7_mean = np.average([s7s[i], s7_c[i]], weights = 1/np.array([s7_errs[i], s7std_c[i]]))
    s7_std = np.std([s7s[i], s7_c[i]])
    row = ["S7",    s7s[i],       s7_errs[i],   s7_c[i], s7std_c[i], s7_mean, s7_std]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    s8_mean = np.average([s8s[i], s8_c[i]], weights = 1/np.array([s8_errs[i], s8std_c[i]]))
    s8_std = np.std([s8s[i], s8_c[i]])
    row = ["S8",    s8s[i],       s8_errs[i],   s8_c[i], s8std_c[i], s8_mean, s8_std]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    row = ["Aim",aims[i],        aim_errs[i],    "/",     "/", aims[i], aim_errs[i]]
    for j in range(1, len(row)):
        if isinstance(row[j], float):
            row[j] = round(row[j], 4)
    content.append(row*1)
    
    try:
        row = ["S9",    s9s[i],       s9_errs[i],   "/",     "/", s9s[i],       s9_errs[i]]
        for j in range(1, len(row)):
            if isinstance(row[j], float):
                row[j] = round(row[j], 4)
        content.append(row*1)
    except IndexError:
        pass
    except NameError:
        pass
    
    table_i = pd.DataFrame(content, columns = columns)
    table_i.to_csv(f"fitting/AAA{crball}_Bin{i}_res_mod{modify}")
    table_i.to_pickle(f"fitting/AAA{crball}_Bin{i}_res_mod{modify}")
    print("\n\n\nCrystalBall#:",crball,"      mod#:", modify)
    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 4,):
        print(table_i)
    
    ##############################################
    columns =  [f"Bin{i}", "P-Estimate", "Uncertainty"]
    content = []
    
    flu = ufloat(fls[i], fl_errs[i])
    Cflu = ufloat(1 - fls[i], fl_errs[i])
    s3u = ufloat(s3s_all_mean[i], max(s3s_all_std[i], s3s_mean_std[i]))
    
    P1 = 2 * s3u/Cflu
    row = ["P1", P1.n, P1.s]
    content.append(row)
    P2 = 2/3 * ufloat(afbs[i], max(afb_errs[i], afb_std)) / Cflu
    row = ["P2", P2.n, P2.s]
    content.append(row)
    try:
        P3 = - ufloat(s9s[i], s9_errs[i]) / Cflu
        row = ["P3", P3.n, P3.s]
        content.append(row)
    except IndexError:
        pass
    except NameError:
        pass
    
    Pp4 = ufloat(s4_mean, max(s4_errs[i], s4_std))/um.sqrt(flu*Cflu)
    row = ["P'4", Pp4.n, Pp4.s]
    content.append(row)
    Pp5 = ufloat(s5_mean, max(s5_errs[i], s5_std))/um.sqrt(flu*Cflu)
    row = ["P'5", Pp5.n, Pp5.s]
    content.append(row)
    Pp6 = ufloat(s7_mean, max(s7_errs[i], s7_std))/um.sqrt(flu*Cflu)
    row = ["P'6", Pp6.n, Pp6.s]
    content.append(row)
    Pp8 = ufloat(s8_mean, max(s8_errs[i], s8_std))/um.sqrt(flu*Cflu)
    row = ["P'8", Pp8.n, Pp8.s]
    content.append(row)
    
    table_i = pd.DataFrame(content, columns = columns)
    table_i.to_csv(f"fitting/AAA{crball}_Bin{i}_resP_mod{modify}")
    table_i.to_pickle(f"fitting/AAA{crball}_Bin{i}_resP_mod{modify}")
    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 4,):
        print("\n",table_i)
    
    
    
    
    
