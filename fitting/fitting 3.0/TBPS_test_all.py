# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 19:48:04 2022

@author: Stefano
"""

from TBPS_bin_functions import *
from TBPS_pdfs1 import *
from TBPS_pdfs2 import *
from TBPS_pdf_full import *
from TBPS_fit_functions import *
from classification.data_loader import load_classified_total, load_classified_acceptance
from TBPS_counting import count, plot_counting_results
from find_coeff_numpy import legendre_eval, legendre_eval_project_1D
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance

import pickle
import numpy as np
from scipy.integrate import quad
from scipy.stats import iqr, rv_continuous
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm

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


# Using literature values from
# LHCb collaboration. Angular analysis of the B0→ K∗0µ+µ− decay using 3 fb−1 
# of integrated luminosity. Available at https://arxiv.org/pdf/1512.04442.pdf. 
testbins = [( 0.1,   0.98),
            ( 1.1,  2.5),
            ( 2.5,  4.0),
            ( 4.0,  6.0),
            ( 6.0,  8.0),
            (11.0, 12.5),
            (15.0, 17.0),
            (17.0, 19.0),
            ( 1.0,  6.0)] 
 
Fl  = [  0.263,  0.660,  0.876,  0.611,  0.579,     0.493,  0.349,  0.354,  0.690]
S3  = [ -0.036, -0.077,  0.035,  0.035, -0.142,    -0.189, -0.142, -0.188,  0.012]
S4  = [  0.082, -0.077, -0.234, -0.219, -0.296,    -0.283, -0.321, -0.266, -0.155]
S5  = [  0.170,  0.137, -0.022, -0.146, -0.249,    -0.327, -0.316, -0.323, -0.023]
Afb = [ -0.003, -0.191, -0.118,  0.025,  0.152,     0.318,  0.411,  0.305, -0.075]
S7  = [  0.015, -0.219,  0.068, -0.016, -0.047,    -0.141,  0.061,  0.044, -0.077]
S8  = [  0.079, -0.098,  0.030,  0.167, -0.085,    -0.007,  0.003,  0.013,  0.028]
S9  = [ -0.083, -0.119, -0.092, -0.032, -0.024,    -0.004, -0.019, -0.094, -0.064]
coeffs = np.column_stack((Fl, Afb, S3, S4, S5, S7, S8, S9))
count_coeffs = [Afb, S4, S5, S7, S8]

path = "acceptance/acceptance_legendre_coeffs.npz"
acceptance, acceptance_ctl, acceptance_ctk, acceptance_phi, file = load_acceptance(path)
acc, acc_ctl, acc_ctk, acc_phi, file = load_acceptance(path)

# Use coeffciients, faster:
acceptance = []
for i in range(len(testbins)):
    acceptance.append(file[f"{i}"])
#Match Test and "Real" Bins
acceptance[6],     acceptance_ctl[6] =     acc[5], acc_ctl[5]                                             
acceptance_ctk[6], acceptance_phi[6] = acc_ctk[5], acc_phi[5]
acceptance[7],     acceptance_ctl[7] =     acc[6], acc_ctl[6]                                            
acceptance_ctk[7], acceptance_phi[7] = acc_ctk[6], acc_phi[6]
acceptance[5],     acceptance_ctl[5] =     acc[7], acc_ctl[7]                                             
acceptance_ctk[5], acceptance_phi[5] = acc_ctk[7], acc_phi[7]
acceptance_ctl = acceptance*1
acceptance_ctk = acceptance*1
acceptance_phi = acceptance*1
del path, file, acc, acc_ctl, acc_ctk, acc_phi

try:
    a = acceptance
    del a
except NameError:
    def trivial1 (x):
        return 1
    def trivial3 (x, y, z):
        return 1
    acceptance = [1 for i in Fl]
    acceptance_ctl = [trivial1 for i in Fl] 
    acceptance_ctk = [trivial1 for i in Fl] 
    acceptance_phi = [trivial1 for i in Fl] 

#%%
yn = input("Do you want to simulate new data or load old? [new/old]")

M = int(1e3)
acc = 1

if yn == "new":
    #GENERATE A TESTDATA WITH KNOWN ACC_PDF, AND KNOW COEFFICIENTS
    ctl = []
    ctk = []
    phi = []
    q2 = []
    testdata = []
    for i in tqdm(range(len(testbins)), "Creating TestData"):
        x_l = np.linspace(-1, 1, 5)
        x_k = np.linspace(-1, 1, 5)
        x_p = np.linspace(-1, 1, 5) * np.pi
        angs = np.meshgrid(x_l, x_k, x_p)
        #C = 1.1*np.amax(full_acc_PDF(angs, coeffs[i], acceptance[i]))
        C = 1.1*np.amax(full_acc_PDF(angs, coeffs[i], 1)) 
        ctl_acc = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,  0, 0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1, 1, 1, 1])
        ctk_acc = np.array([-1,-1,-1, 0, 0, 0, 1, 1, 1, -1,-1,-1, 0, 0, 0, 1, 1, 1,  -1,-1,-1, 0, 0, 0, 1, 1, 1])
        phi_acc = np.array([-1, 0, 1,-1, 0, 1,-1, 0, 1, -1, 0, 1,-1, 0, 1,-1, 0, 1,  -1, 0, 1,-1, 0, 1,-1, 0, 1])
        C *= max(acceptance[i](ctl_acc, ctk_acc, phi_acc))
        C = max(C, 0.20)
        print("\nC = ", C)
        if np.isnan(C):
            print("C was none, should worry about it?")
            C = 2
        size = 0
        ctli = []
        ctki = []
        phii = []
        q2i = []
        while size < M:
            y_l = np.random.uniform(-1, 1)
            y_k = np.random.uniform(-1, 1)
            y_p = np.random.uniform(-1, 1) * np.pi
            p   = np.random.uniform(0, C)
            #if full_acc_PDF([y_l, y_k, y_p], coeffs[i], acceptance[i]) >= p:
            P = full_acc_PDF([y_l, y_k, y_p], coeffs[i], 1) * acceptance[i](y_l, y_k, y_p)
            if P >= p:
                size += 1
                ctli.append(y_l); ctl.append(y_l)
                ctki.append(y_k); ctk.append(y_k)
                phii.append(y_p); phi.append(y_p)
                y_q = np.random.uniform(testbins[i][0], testbins[i][1])
                q2i.append(y_q)
                q2.append(y_q)
        testdatai = pd.DataFrame(np.column_stack([q2i, ctli, ctki, phii]), 
                                 columns = ["q2","costhetal", "costhetak", "phi"])
        try:
            if acceptance[0] == 1:
                name = f"TestData_FullPDF(NoAcc)_Bin{testbins[i]}"
                try:
                    old_testdatai = pd.read_pickle(f"TestPlots/{name}.pkl")
                    new = pd.concat([old_testdatai, testdatai])
                    new.to_pickle(f"TestPlots/{name}.pkl")
                    new.to_pickle(f"TestPlots/{name}.csv")
                except:
                    testdatai.to_pickle(f"TestPlots/{name}.pkl")
                    testdatai.to_pickle(f"TestPlots/{name}.csv")
        except:
            name = f"TestData_FullPDF_Bin{testbins[i]}"
            try:
                old_testdatai = pd.read_pickle(f"TestPlots/{name}.pkl")
                new = pd.concat([old_testdatai, testdatai])
                new.to_pickle(f"TestPlots/{name}.pkl")
                new.to_pickle(f"TestPlots/{name}.csv")
            except:
                testdatai.to_pickle(f"TestPlots/{name}.pkl")
                testdatai.to_pickle(f"TestPlots/{name}.csv")
            testdata.append(testdatai)
        
            testdata.append(testdatai)
    del i, x_l, x_k, x_p, angs, C, 
    del size, ctli, ctki, phii
    del y_l, y_k, y_p, p
    del testdatai, q2, ctl, ctk, phi
    
elif yn == "old":
    testdata = []
    for i in range(len(testbins)):
        if acc == 1:
            try:
                if acceptance[0] != 1:
                    try:
                        file = f"TestPlots/TestData_FullPDF_Bin{testbins[i]}.pkl"
                        testdata_i = pd.read_pickle(file)
                        testdata.append(testdata_i[:M])
                    except:
                        pass
            except:
                try:
                    file = f"TestPlots/TestData_FullPDF_Bin{testbins[i]}.pkl"
                    testdata_i = pd.read_pickle(file)
                    testdata.append(testdata_i[:M])
                except:
                    pass
        else:
            file = f"TestPlots/TestData_FullPDF(NoAcc)_Bin{testbins[i]}.pkl"
            testdata_i = pd.read_pickle(file)
            testdata.append(testdata_i[:M])
    del testdata_i, i, file
    
else:
    print("Did nothing, as answer was neither <new> nor <old>")

#%% TEST COUNTING METHOD
###############################################################################
if acc:
    count_res = count(testdata, acceptance, avg_acc = "auto", n = 1)
    plot_counting_results(count_res, true = count_coeffs, M = M)
else:
    count_res = count(testdata, np.zeros(len(testdata))+1, avg_acc = "auto", n = 1)
    plot_counting_results(count_res, true = count_coeffs, M = M)


##############################################################################
#%% TEST COSTHETAS
###############################################################################
# Pick Function
if acc:
    f = [ctl_acc_PDF_NotNormal, ctk_acc_PDF_NotNormal]
else:
    f = [ctl_PDF, ctk_PDF]
count = 2
#f = phi_acc_PDF
#f = ctl_acc_PDF
if hasattr(f, "__len__"):
    if len(f) == 2:
        if "ctl" in f[0].__name__ and "ctk" in f[1].__name__:
            # Get Guesses "close, more or less" to True Values
            rands = np.random.uniform(0.75, 1.25, size = (len(coeffs), 2))
            guess = rands * np.column_stack((coeffs[:,0], coeffs[:,1]))
            #Set Limits
            limits = [ [ (0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)],
                        [(0, 1), (-1, 1)]]
            if acc:
                args = []
                for i in range(len(mybins)):
                    args.append([ acceptance_all[i], acceptance_all[i] ])
            else:
                args = None
    elif len(f) == 3:
        # Get Guesses "close, more or less" to True Values
        rands = np.random.uniform(0.75, 1.25, size = (len(coeffs), 4))
        guess = rands * np.column_stack(
                        (coeffs[:,0], coeffs[:,1], coeffs[:,2], coeffs[:,7]))
        #Set Limits    Fl      Afb       S3       Aim
        limits = [  [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)],
                    [(0, 1), (-1, 1), (-1, 1), (-1, 1)]]
        if "acc" in f[0].__name__:
            args = np.column_stack([acceptance_ctl, acceptance_ctk, acceptance_phi])
        else:
            args = None
elif "ctl" in f.__name__ :
    if acc:
        args = acceptance_ctl *   1
    else:
        args = None
    # Get Guesses "close, more or less" to True Values
    rands = np.random.uniform(0.75, 1.25, size = (len(coeffs), 2))
    guess = rands * np.column_stack((coeffs[:,0], coeffs[:,1]))
    limits = 0
    
    
# Minimisation Parameters
Nlayers = 3
want_scipy = [0,1,2]
if want_scipy:
    Nlayers += 1

guess_per_bin = False
limits_per_bin = False
if len(guess) == len(testbins):
    guess_per_bin = True
if hasattr(limits, "__len__"):
    if len(limits) == len(testbins):
        limits_per_bin = True

(Fls, Afbs), (Fl_errs, Afbs_errs), fits = fit_angs(testdata, f, guess, 
                                guess_per_bin = guess_per_bin, 
                                limits = limits,
                                limits_per_bin = limits_per_bin,
                                Nlayers = Nlayers, vary = 0.01,
                                args = args, 
                                want_scipy = want_scipy)
del limits, guess_per_bin, limits_per_bin, Nlayers

##############################################################################
# FIT PHI
##############################################################################
try:
    count = count
except:
    count = 0
if acc:
    f = phi_acc_PDF_NotNormal
else:
    f = phi_PDF
count += 1
rands = np.random.uniform(0.75, 1.25, size = (len(coeffs), 2))
guess = rands * np.column_stack((coeffs[:,2], coeffs[:,7]))
limits = 0
if acc:
    args = acceptance_phi *   1
else:
    args = None
    
# Minimisation Parameters
Nlayers = 3
want_scipy = [0,1,2]
if want_scipy:
    Nlayers += 1

guess_per_bin = False
limits_per_bin = False
if len(guess) == len(testbins):
    guess_per_bin = True
if hasattr(limits, "__len__"):
    if len(limits) == len(testbins):
        limits_per_bin = True

(S3s, Aims), (S3_errs, Aims_errs), fits = fit_angs(testdata, f, guess, 
                                guess_per_bin = guess_per_bin, 
                                limits = limits,
                                limits_per_bin = limits_per_bin,
                                Nlayers = Nlayers, vary = 0.01,
                                args = args, 
                                want_scipy = want_scipy)
del limits, guess_per_bin, limits_per_bin, Nlayers


##############################################################################
# TAKE RESULTS
##############################################################################
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
if acc:
    plt.suptitle("Testing Results of Projections Fit")
else:
    plt.suptitle("Testing Results of Projections Fit")
ns = np.linspace(0, len(testdata) - 1, len(testdata))
ax1.errorbar(ns, Fls, yerr=Fl_errs, fmt = ".", capsize = 4,c ="r")
ax1.plot(ns, Fl, "x", label='True Value', c = "b", zorder = 4)
ax2.errorbar(ns, Afbs, yerr=Afb_errs, fmt = ".", capsize = 4,c ="r")
ax2.plot(ns, Afb, "x", label='True Value', c = "b", zorder = 4)
ax3.errorbar(ns, S3s, yerr=S3_errs, fmt = ".", capsize = 4,c ="r")
ax3.plot(ns, S3, "x", label='True Value', c = "b", zorder = 4)
ax4.errorbar(ns, Aims, yerr=Aim_errs, fmt = ".", capsize = 4,c ="r")
ax4.plot(ns, S9, "x", label='True Values (of S9)', c = "b", zorder = 4)
ax1.grid(alpha = 0.3)
ax2.grid(alpha = 0.3)
ax3.grid(alpha = 0.3)
ax4.grid(alpha = 0.3)
ax1.set_ylabel(r'$F_L$')
ax2.set_ylabel(r'$A_{FB}$')
ax3.set_ylabel(r'$S_3$')
ax4.set_ylabel(r'$A_{Im}$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
ax3.set_xlabel(r'Bin number')
ax4.set_xlabel(r'Bin number')
ax2.set_ylim(min(ax2.get_ylim()[0], -0.5), max(ax2.get_ylim()[1], 0.5))
ax3.set_ylim(min(ax3.get_ylim()[0], -0.5), max(ax3.get_ylim()[1], 0.5))
ax4.set_ylim(min(ax4.get_ylim()[0], -0.5), max(ax4.get_ylim()[1], 0.5))
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.tight_layout()
if acc:
    name = f"TestProj_scipy{want_scipy}_{M}Datapoints"
else:
    name = f"TestProj(noAcc)_scipy{want_scipy}_{M}Datapoints"
#plt.savefig(f"TestPlots/{name}")
plt.show()
del fig, ax1, ax2, ns



#%% RESULTS of TESTDATA
#%%% NLL PLOTS

N = 100
delta = 0.2
for i in range(len(testdata)):

    fig, axes = plt.subplots(2,1, figsize = (7, 10), 
                            gridspec_kw={'height_ratios': [1.3, 1]})
    plt.suptitle(r"NLL of $cos(\theta_l)$;"+f" Bin {i}: {testbins[i]} $GeV^2$")
    ax0 = axes[0]; ax1 = axes[1]
    
    fl = np.linspace(0, 1, N); afb = np.linspace(-1, 1, N)
    X,Y = np.meshgrid(fl,afb)
    if acc:
        Z = nLL_ctl(testdata, acceptance_l, [X, Y], i)
    else:
        Z = nLL_ctl(testdata, np.zeros(len(testbins))+1, [X, Y], i)
    cntr0 = ax0.contourf(X, Y, Z, 100, cmap = "nipy_spectral")
    ax0.errorbar(Fls[i], Afbs[i], yerr = Afb_errs[i], xerr = Fl_errs[i], 
                 fmt = ".", capsize = 5, color = "red")
    ax0.plot(Fl[i], Afb[i], "x", c="y")
    ax0.set_xlabel(r"$F_l$"); ax0.set_ylabel(r"$A_fb$")
    ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
    
    plt.subplot(2,1,2)
    afbmin_i, flmin_i = np.unravel_index(np.nanargmin(Z), Z.shape)
    flmin = fl[flmin_i]; afbmin = afb[afbmin_i]
    fl0 = max(0, flmin - delta);    fl1 = min(1, flmin + delta)
    afb0 = max(-1, afbmin - delta); afb1 = min(1, afbmin + delta)
    fl = np.linspace(fl0, fl1, N); afb = np.linspace(afb0, afb1, N)
    X,Y = np.meshgrid(fl,afb)
    if acc:
        Z = nLL_ctl(testdata, acceptance_l, [X, Y], i)
    else:
        Z = nLL_ctl(testdata, np.zeros(len(testbins))+1, [X, Y], i)
    ax1.errorbar(Fls[i], Afbs[i], yerr = Afb_errs[i], xerr = Fl_errs[i], 
                 fmt = ".", capsize = 5, color = "red")
    ax1.plot(Fl[i], Afb[i], "x", c="y")
    cntr1 = ax1.contourf(X, Y, Z, 100, cmap = "nipy_spectral")
    ax1.set_xlabel(r"$F_l$"); ax1.set_ylabel(r"$A_fb$")
    ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")
    
    plt.subplots_adjust(hspace=0.2)
    fig.colorbar(cntr0, ax = ax0)
    fig.colorbar(cntr1, ax = ax1)
    fig.tight_layout()
    plt.savefig( f"TestPlots/Test_NLL_ctl_bin{i}_{M}Datapoints.pdf" )
    plt.show()
    
    
    # HERE FL AND AFB ARE ACTUALLY S3 AND AIM, BUT TOO MUCH EFFORT
    fig, axes = plt.subplots(2,1, figsize = (7, 10), 
                            gridspec_kw={'height_ratios': [1.3, 1]})
    plt.suptitle(r"NLL of $\phi$;"+f" Bin {i}: {testbins[i]} $GeV^2$")
    ax0 = axes[0]; ax1 = axes[1]
    
    fl = np.linspace(-1, 1, N); afb = np.linspace(-1, 1, N)
    X,Y = np.meshgrid(fl,afb)
    if acc:
        Z = nLL_phi(testdata, acceptance_phi, [X, Y], i)
    else:
        Z = nLL_phi(testdata, np.zeros(len(testbins))+1, [X, Y], i)
    cntr0 = ax0.contourf(X, Y, Z, 200, cmap = "nipy_spectral")
    ax0.errorbar(S3s[i], Aims[i], yerr = Aim_errs[i], xerr = S3_errs[i], 
                  fmt = ".", capsize = 5, color = "red")
    ax0.plot(S3[i], S9[i], "x", c="y")
    ax0.set_xlabel(r"$S_3$"); ax0.set_ylabel(r"$A_{Im}$")
    ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
    
    plt.subplot(2,1,2)
    afbmin_i, flmin_i = np.unravel_index(np.nanargmin(Z), Z.shape)
    flmin = fl[flmin_i]; afbmin = afb[afbmin_i]
    fl0 = max(0, flmin - delta);    fl1 = min(1, flmin + delta)
    afb0 = max(-1, afbmin - delta); afb1 = min(1, afbmin + delta)
    fl = np.linspace(fl0, fl1, N); afb = np.linspace(afb0, afb1, N)
    X,Y = np.meshgrid(fl,afb)
    if acc:
        Z = nLL_phi(testdata, acceptance_phi, [X, Y], i)
    else:
        Z = nLL_phi(testdata, np.zeros(len(testbins))+1, [X, Y], i)
    ax1.errorbar(S3s[i], Aims[i], yerr = Aim_errs[i], xerr = S3_errs[i], 
                  fmt = ".", capsize = 5, color = "red")
    ax1.plot(S3[i], S9[i], "x", c="y")
    cntr1 = ax1.contourf(X, Y, Z, 200, cmap = "nipy_spectral")
    ax1.set_xlabel(r"$S_3$"); ax1.set_ylabel(r"$A_{Im}$")
    ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")

    plt.subplots_adjust(hspace=0.2)
    fig.colorbar(cntr0, ax = ax0)
    fig.colorbar(cntr1, ax = ax1)
    plt.savefig( f"TestPlots/Test_NLL_phi_bin{i}_{M}Datapoints.pdf" )
    plt.show()

del fig, axes, ax0, ax1, cntr0, cntr1
del N, delta, fl, afb, X, Y, Z 
del flmin_i, afbmin_i, flmin, afbmin, fl0, fl1, afb0, afb1

#%%% BINS PLOTS
# Prepare for plotting
rc = len(testdata)
c = int(np.sqrt(rc))
r = int( np.ceil( rc / c) )
figsize = (c * 5, r * 2.25)

# SEE RESULTS FOR COSTHETAL
plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(testdata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetal vs PDF")
    ctli = testdata[i]['costhetal']
    ctli_binned, edges = np.histogram(ctli, bins = "auto", range = (-1, 1))
    centres = (edges[:-1] + edges[1:])/2.
    bwidth = edges[1] - edges[0]
    err = np.sqrt(ctli_binned)
    heights = ctli_binned / (bwidth * len(ctli))
    err = err /(bwidth * len(ctli))
    plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
    x = np.linspace(-1, 1, 100)
    if acc:
        y1 = ctl_acc_PDF(x, [Fl[i], Afb[i]], acceptance_ctl[i], accept = True)
        y2 = ctl_acc_PDF(x, [Fls[i], Afbs[i]], acceptance_ctl[i], accept = True)
    else:
        y1 = ctl_PDF(x, [Fl[i], Afb[i]])
        y2 = ctl_PDF(x, [Fls[i], Afbs[i]])
    title = f" Bin {i}: {testbins[i]} $GeV^2$;   "
    values = "$F_l$="+f"{round(Fls[i], 3)},"+r"$A_{fb}$="+f"{round(Afbs[i],3)}"
    plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                 ha = "left", size = 10)
    plt.plot(x, y2, label = "Acc_PDF")
    plt.plot(x, y1, label = "True")
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'PDF')
    plt.ylim(0, max(plt.ylim()[1], 1))
    plt.xlim(-1, 1)
    plt.legend()
    plt.grid()
plt.savefig(f'TestPlots/Ctl_TestFit_Datapoints{M}')
plt.show()

# SEE RESULTS FOR COSTHETA_K
plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(testdata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetak vs PDF")
    ctki = testdata[i]['costhetak']
    ctki_binned, edges = np.histogram(ctki, bins = "auto", range = (-1, 1))
    centres = (edges[:-1] + edges[1:])/2.
    bwidth = edges[1] - edges[0]
    err = np.sqrt(ctki_binned)
    heights = ctki_binned / (bwidth * len(ctki))
    err = err /(bwidth * len(ctki))
    plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
    x = np.linspace(-1, 1, 100)
    if acc:
        y2 = ctk_acc_PDF(x, Fls[i], acceptance_ctk[i], accept = True)
        y2 = ctk_acc_PDF(x, Fls[i], acceptance_ctk[i], accept = True)
    else:
        y1 = ctk_PDF(x, Fls[i])
        y2 = ctk_PDF(x, Fls[i])
    plt.plot(x, y2, label = "Acc_PDF")
    plt.plot(x, y1, label = "True")
    title = f" Bin {i}: {testbins[i]} $GeV^2$;   "
    values = "$F_l$="+f"{round(Fls[i], 3)},"+r"$A_{fb}$="+f"{round(Afbs[i],3)}"
    plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                 ha = "left", size = 10)
    plt.xlabel(r'$cos(\theta_k)$')
    plt.ylabel(r'PDF')
    plt.ylim(0, max(plt.ylim()[1], 1))
    plt.xlim(-1, 1)
    plt.legend()
    plt.grid()
plt.savefig(f'TestPlots/Ctk_TestFit_Datapoints{M}')
plt.show()

# SEE RESULTS FOR PHI
plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(testdata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Phi vs PDF")
    phii = testdata[i]['phi']
    phii_binned, edges = np.histogram(phii, bins = "auto", range = (-np.pi, np.pi))
    centres = (edges[:-1] + edges[1:])/2.
    bwidth = edges[1] - edges[0]
    err = np.sqrt(phii_binned)
    heights = phii_binned / (bwidth * len(phii))
    err = err /(bwidth * len(phii))
    plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
    x = np.linspace(-np.pi, np.pi, 100)
    if acc:
        y1 = phi_acc_PDF(x, [S3s[i], Aims[i]], acceptance_phi[i], accept = True)
        y2 = phi_acc_PDF(x, [S3s[i], Aims[i]], acceptance_phi[i], accept = True)
    else:
        y1 = phi_PDF(x, [S3s[i], Aims[i]])
        y2 = phi_PDF(x, [S3s[i], Aims[i]])
    plt.plot(x, y2, label = "Acc_PDF")
    plt.plot(x, y1, label = "True")
    title = f" Bin {i}: {testbins[i]} $GeV^2$;   "
    values = "$S_3$="+f"{round(S3s[i], 3)},"+r"$A_{Im}$="+f"{round(Aims[i],3)}"
    plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                 ha = "left", size = 10)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'PDF')
    plt.ylim(0, max(plt.ylim()[1], 1))
    plt.xlim(-np.pi, np.pi)
    plt.legend()
    plt.grid()
plt.savefig(f'TestPlots/Phi_TestFit_Datapoints{M}')
plt.show()
#%%
del rc, r, c, figsize
del i, ctli, ctli_binned, ctki, ctki_binned, phii, phii_binned
del bwidth, heights, edges, centres, err
del x, y2
del title, values



#%% CLEAN MEMORY
yn = input("Do you want to delete Test variables)? [y/n]")
if yn:
    del Fl, Afb, coeffs, M, acceptance_l
    del Fls, Afbs, Fl_errs, Afb_errs, Fits
    del test, testbins, testdata, yn
    
# if count == 2:
#     if "ctl" in f[0].__name__ and "ctk" in f[1].__name__:  
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
#         plt.suptitle(r"Fitting Test")
#         ns = np.linspace(0, len(testdata) - 1, len(testdata))
#         ax1.errorbar(ns, Fls, yerr=Fl_errs, fmt = ".", capsize = 4, label="Fit",c="r")
#         ax1.plot(ns, Fl, "x", label='True Value', c = "b", zorder = 4)
#         ax1.plot(ns, np.array(guess)[:,0], "o", label='Guess', c = "y", zorder = 4)
#         ax2.errorbar(ns, Afbs, yerr=Afb_errs, fmt = ".", capsize = 4, label='Fit',c="r")
#         ax2.plot(ns, Afb, "x", label='True Value', c = "b", zorder = 4)
#         ax2.plot(ns, np.array(guess)[:,1], "o", label='Guess', c = "y", zorder = 4)
#         ax1.grid(alpha = 0.3)
#         ax2.grid(alpha = 0.3)
#         ax1.set_ylabel(r'$F_L$')
#         ax2.set_ylabel(r'$A_{FB}$')
#         ax1.set_xlabel(r'Bin number')
#         ax2.set_xlabel(r'Bin number')
#         ax1.legend()
#         ax2.legend()
#         plt.tight_layout()
#         plt.savefig(f"TestPlots/Test_Proj_Coeffs_scipy{want_scipy}_{M}Datapoints")
#         plt.show()
#         del guess, fig, ax1, ax2, ns
        
# elif count == 3:

# else:
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
#     plt.suptitle(r"Fitting Test")
#     ns = np.linspace(0, len(testdata) - 1, len(testdata))
#     ax1.errorbar(ns, Fls, yerr=Fl_errs, fmt = ".", capsize = 4, label="Fit",c="r")
#     ax1.plot(ns, Fl, "x", label='True Value', c = "b", zorder = 4)
#     #ax1.plot(ns, np.array(guess)[:,0], "o", label='Guess', c = "y", zorder = 4)
#     ax2.errorbar(ns, Afbs, yerr=Afb_errs, fmt = ".", capsize = 4, label='Fit',c="r")
#     ax2.plot(ns, Afb, "x", label='True Value', c = "b", zorder = 4)
#     #ax2.plot(ns, np.array(guess)[:,1], "o", label='Guess', c = "y", zorder = 4)
#     ax1.grid(alpha = 0.3)
#     ax2.grid(alpha = 0.3)
#     ax1.set_ylabel(r'$F_L$')
#     ax2.set_ylabel(r'$A_{FB}$')
#     ax1.set_xlabel(r'Bin number')
#     ax2.set_xlabel(r'Bin number')
#     ax1.legend()
#     ax2.legend()
#     plt.tight_layout()
#     plt.savefig(f"TestPlots/Test_Proj_Coeffs_scipy{want_scipy}_{M}Datapoints")
#     plt.show()
#     del guess, fig, ax1, ax2, ns