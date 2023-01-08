# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:35:58 2022

@author: Stefano
"""
###############################################################################
modify = 0 #0, 1, -1 and any other integer
crball = 0
###############################################################################

from TBPS_bin_functions import *
from TBPS_pdfs1 import *
from TBPS_pdfs2 import *
from TBPS_pdf_full import *
from TBPS_fit_functions import *
from TBPS_counting import *

import numpy.polynomial.legendre as npl
from functools import reduce
from find_coeff_numpy import legendre_eval, legendre_eval_project_1D
from find_coeff_numpy import legendre_eval_project_2D, load_acceptance, acc_modify

import pickle
import numpy as np
from scipy.integrate import quad
from scipy.stats import iqr
import pandas as pd
from iminuit import Minuit

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

path = f'classification/final_data_processed/total_dataset_classified{crball}.pkl'
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
acceptance, acceptance_ctl, acceptance_ctk, acceptance_phi, file \
    = load_acceptance(path)
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


#%% FIT ANGLES 

###############################################################################
#% FIT CTL and CTK
###############################################################################
# Pick Function
f = [ctl_acc_PDF_NotNormal, ctk_acc_PDF_NotNormal]
f = [ctl_acc_PDF, ctk_acc_PDF]
args = []
for i in range(len(mybins)):
    args.append([ acceptance_all[i], acceptance_all[i] ])
#args = np.column_stack([acceptance_all, acceptance_all])
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
want_scipy = [0,2]
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
del limits, guess_per_bin, limits_per_bin, Nlayers


###############################################################################
# FIT PHI
###############################################################################

f = phi_acc_PDF_NotNormal
guess = [0,0]
limits = 0
args = acceptance_all

Nlayers = 2
want_scipy = [0,2]
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
del limits, guess_per_bin, limits_per_bin, Nlayers



###############################################################################
# TAKE RESULTS
###############################################################################
counter = 3
if counter == 2:
    if "ctl" in f[0].__name__ and "ctk" in f[1].__name__:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        plt.suptitle(r"Results of Fitting $cos(\theta_l)$ & $cos(\theta_k)$")
        ns = np.linspace(0, len(bindata) - 1, len(bindata))
        ax1.errorbar(ns, fls, yerr=fl_errs, fmt = ".", capsize = 4, label=r'$F_L$')
        ax2.errorbar(ns, afbs, yerr=afb_errs, fmt = ".", capsize = 4, label=r'$A_{FB}$')
        ax1.grid(alpha = 0.3)
        ax2.grid(alpha = 0.3)
        ax1.set_ylabel(r'$F_L$')
        ax2.set_ylabel(r'$A_{FB}$')
        ax1.set_xlabel(r'Bin number')
        ax2.set_xlabel(r'Bin number')
        plt.tight_layout()
        plt.savefig("MinPlots/Fl_Afb_Coeffs")
        plt.show()
        del fig, ax1, ax2, ns
        
        np.save(f"fitting/coss{crball}_fls_scipy{want_scipy}_mod{modify}.npy", 
                np.array([fls,  fl_errs]), 
                allow_pickle = True)
        np.save(f"fitting/coss{crball}_afbs_scipy{want_scipy}_mod{modify}.npy", 
                np.array([afbs, afb_errs]), 
                allow_pickle = True)
#          
if counter == 3:
        fig, axs = plt.subplots(2, 2, figsize=(8, 5))
        ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
        plt.suptitle(r"Results of Fitting Projections")
        ns = np.linspace(0, len(bindata) - 1, len(bindata))
        ax1.errorbar(ns, fls,  yerr= fl_errs, fmt = ".", capsize = 4)
        ax2.errorbar(ns, afbs, yerr=afb_errs, fmt = ".", capsize = 4)
        ax3.errorbar(ns, s3s,  yerr= s3_errs, fmt = ".", capsize = 4)
        ax4.errorbar(ns, aims, yerr=aim_errs, fmt = ".", capsize = 4)
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
        ax3.set_ylim(min(ax3.get_ylim()[0], -0.4), max(ax3.get_ylim()[1], 0.4))
        ax4.set_ylim(min(ax4.get_ylim()[0], -0.4), max(ax4.get_ylim()[1], 0.4))
        plt.tight_layout()
        plt.savefig(f"MinPlots/Projection{crball}_Coeffs_scipy{want_scipy}_mod{modify}")
        plt.show()
        del fig, ax1, ax2, ns
        
        np.save(f"fitting/proj{crball}_fls_scipy{want_scipy}_mod{modify}.npy", 
                np.array([fls,  fl_errs]), 
                allow_pickle = True)
        np.save(f"fitting/proj{crball}_afbs_scipy{want_scipy}_mod{modify}.npy", 
                np.array([afbs, afb_errs]), 
                allow_pickle = True)
        np.save(f"fitting/proj{crball}_s3s_scipy{want_scipy}_mod{modify}.npy", 
                np.array([s3s,  s3_errs]), 
                allow_pickle = True)
        np.save(f"fitting/proj{crball}_aims_scipy{want_scipy}_mod{modify}.npy", 
                np.array([aims, aim_errs]), 
                allow_pickle = True)

    
#%% PROJECTION HISTS RESULTS
if not modify:
    # Prepare for plotting
    rc = len(bindata)
    c = int(np.ceil(np.sqrt(rc)))
    r = int(np.ceil( rc / c) )
    figsize = (c * 4, r * 3)
    
    
    plt.figure(figsize = figsize, tight_layout = True)
    for i in range(len(bindata)):
        plt.subplot(r, c, i + 1)
        plt.suptitle(r"Observed cos$(\theta_\ell)$ Distribution vs PDF")
        ctli = bindata[i]['costhetal']
        ctli_binned, edges = np.histogram(ctli, bins = "auto", range = (-1, 1))
        centres = (edges[:-1] + edges[1:])/2.
        bwidth = edges[1] - edges[0]
        err = np.sqrt(ctli_binned)
        heights = ctli_binned / (bwidth * len(ctli))
        err = err /(bwidth * len(ctli))
        # bwidth = 2 * iqr(ctli) / np.cbrt(len(ctli))
        # Nbins = int(2 / bwidth)
        # heights,edges = np.histogram(ctli, Nbins)
        # err = np.sqrt(heights)
        plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
        x = np.linspace(-1, 1, 100)
        #y1 = ctl_PDF(x, [fls[i], afbs[i]])
        y2 = ctl_acc_PDF(x, [fls[i], afbs[i]], acceptance_all[i], accept = True)
        title = f" Bin {i}: {mybins[i]} $GeV^2$;   "
        values = "$F_l$="+f"{round(fls[i], 3)},"+r"$A_{fb}$="+f"{round(afbs[i],3)}"
        plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                     ha = "left", size = 10)
        #plt.plot(x, y1, label = "PDF")
        plt.plot(x, y2, label = "Acc_PDF")
        plt.xlabel(r'$cos(\theta_l)$')
        plt.ylabel(r'PDF')
        plt.ylim(0, max(plt.ylim()[1], 1))
        plt.xlim(-1, 1)
        plt.legend()
        plt.grid()
    plt.savefig(f'MinPlots/Ctl{crball}_fit_mod{modify}')
    plt.show()
    
    # SEE RESULTS FOR COSTHETA_K
    plt.figure(figsize = figsize, tight_layout = True)
    for i in range(len(bindata)):
        plt.subplot(r, c, i + 1)
        plt.suptitle(r"Observed cos$(\theta_K)$ Distribution vs PDF")
        ctki = bindata[i]['costhetak']
        ctki_binned, edges = np.histogram(ctki, bins = "auto", range = (-1, 1))
        centres = (edges[:-1] + edges[1:])/2.
        bwidth = edges[1] - edges[0]
        err = np.sqrt(ctki_binned)
        heights = ctki_binned / (bwidth * len(ctki))
        err = err /(bwidth * len(ctki))
        plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
        
        x = np.linspace(-1, 1, 100)
        y2 = ctk_acc_PDF(x, fls[i], acceptance_all[i], accept = True)
        plt.plot(x, y2, label = "Acc_PDF")
        title = f" Bin {i}: {mybins[i]} $GeV^2$;   "
        values = "$F_l$="+f"{round(fls[i], 3)},"+r"$A_{fb}$="+f"{round(afbs[i],3)}"
        plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                     ha = "left", size = 10)
        plt.xlabel(r'$cos(\theta_k)$')
        plt.ylabel(r'PDF')
        plt.ylim(0, max(plt.ylim()[1], 1))
        plt.xlim(-1, 1)
        plt.legend()
        plt.grid()
    plt.savefig(f'MinPlots/Ctk{crball}_fit_mod{modify}')
    plt.show()
    
    # SEE RESULTS FOR PHI
    plt.figure(figsize = figsize, tight_layout = True)
    for i in range(len(bindata)):
        plt.subplot(r, c, i + 1)
        plt.suptitle(r"Observed $\phi$ Distribution vs PDF")
        
        phii = bindata[i]['phi']
        phii_binned, edges = np.histogram(phii, bins = "auto", range = (-np.pi, np.pi))
        centres = (edges[:-1] + edges[1:])/2.
        bwidth = edges[1] - edges[0]
        err = np.sqrt(phii_binned)
        heights = phii_binned / (bwidth * len(phii))
        err = err /(bwidth * len(phii))
        plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
        
        x = np.linspace(-np.pi, np.pi, 100)
        y2 = phi_acc_PDF(x, [s3s[i], aims[i]], acceptance_all[i], accept = True)
        plt.plot(x, y2, label = "Acc_PDF")
        title = f" Bin {i}: {mybins[i]} $GeV^2$;   "
        values = "$S_3$="+f"{round(s3s[i], 3)},"+r"$A_{Im}$="+f"{round(aims[i],3)}"
        plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                     ha = "left", size = 10)
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'PDF')
        plt.ylim(0, max(plt.ylim()[1], 0.4))
        plt.xlim(-np.pi, np.pi)
        plt.legend()
        plt.grid()
    plt.savefig(f'MinPlots/Phi{crball}_Fit_mod{modify}')
    plt.show()
    
    del rc, r, c, figsize
    del i, ctli, ctli_binned, ctki, ctki_binned, phii, phii_binned
    del bwidth, heights, edges, centres, err
    del x, y2
    del title, values
    

#%% NLLS RESULTS
#yn = input("Do you want to do the many NLL subplots (takes a bit)? [y/n]")
if yn:
    N = 200
    delta = 0.2
    for i in range(len(bindata)):
    
        fig, axes = plt.subplots(2,1, figsize = (7, 10), 
                                gridspec_kw={'height_ratios': [1.3, 1]})
        plt.suptitle(r"NLL of $cos(\theta_l)$;"+f" Bin {i}: {mybins[i]} $GeV^2$")
        ax0 = axes[0]; ax1 = axes[1]
        
        fl = np.linspace(0, 1, N); afb = np.linspace(-1, 1, N)
        X,Y = np.meshgrid(fl,afb)
        Z = nLL_ctl(bindata, acceptance_all, [X, Y], i)
        cntr0 = ax0.contourf(X, Y, Z, 200, cmap = "nipy_spectral")
        ax0.errorbar(fls[i], afbs[i], yerr = afb_errs[i], xerr = fl_errs[i], 
                     fmt = ".", capsize = 5, color = "red")
        ax0.set_xlabel(r"$F_l$"); ax0.set_ylabel(r"$A_fb$")
        ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
        
        plt.subplot(2,1,2)
        afbmin_i, flmin_i = np.unravel_index(np.nanargmin(Z), Z.shape)
        flmin = fl[flmin_i]; afbmin = afb[afbmin_i]
        fl0 = max(0, flmin - delta);    fl1 = min(1, flmin + delta)
        afb0 = max(-1, afbmin - delta); afb1 = min(1, afbmin + delta)
        fl = np.linspace(fl0, fl1, N); afb = np.linspace(afb0, afb1, N)
        X,Y = np.meshgrid(fl,afb)
        Z = nLL_ctl(bindata, acceptance_all, [X, Y], i)
        ax1.errorbar(fls[i], afbs[i], yerr = afb_errs[i], xerr = fl_errs[i], 
                     fmt = ".", capsize = 5, color = "red")
        cntr1 = ax1.contourf(X, Y, Z, 200, cmap = "nipy_spectral")
        ax1.set_xlabel(r"$F_l$"); ax1.set_ylabel(r"$A_fb$")
        ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")
    
        plt.subplots_adjust(hspace=0.2)
        fig.colorbar(cntr0, ax = ax0)
        fig.colorbar(cntr1, ax = ax1)
        plt.savefig( f"MinPlots/Fit_NLL_ctl_bin{i}.pdf" )
        plt.show()
        
        # HERE FL AND AFB ARE ACTUALLY S3 AND AIM, BUT TO0 MUCH EFFORT
        fig, axes = plt.subplots(2,1, figsize = (7, 10), 
                                gridspec_kw={'height_ratios': [1.3, 1]})
        plt.suptitle(r"NLL of $\phi$;"+f" Bin {i}: {mybins[i]} $GeV^2$")
        ax0 = axes[0]; ax1 = axes[1]
        
        fl = np.linspace(-1, 1, N); afb = np.linspace(-1, 1, N)
        X,Y = np.meshgrid(fl,afb)
        Z = nLL_phi(bindata, acceptance_all, [X, Y], i)
        cntr0 = ax0.contourf(X, Y, Z, 200, cmap = "nipy_spectral")
        ax0.errorbar(s3s[i], aims[i], yerr = aim_errs[i], xerr = s3_errs[i], 
                     fmt = ".", capsize = 5, color = "red")
        ax0.set_xlabel(r"$S_3$"); ax0.set_ylabel(r"$A_{Im}$")
        ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
        
        plt.subplot(2,1,2)
        afbmin_i, flmin_i = np.unravel_index(np.nanargmin(Z), Z.shape)
        flmin = fl[flmin_i]; afbmin = afb[afbmin_i]
        fl0 = max(0, flmin - delta);    fl1 = min(1, flmin + delta)
        afb0 = max(-1, afbmin - delta); afb1 = min(1, afbmin + delta)
        fl = np.linspace(fl0, fl1, N); afb = np.linspace(afb0, afb1, N)
        X,Y = np.meshgrid(fl,afb)
        Z = nLL_phi(bindata, acceptance_all, [X, Y], i)
        ax1.errorbar(s3s[i], aims[i], yerr = aim_errs[i], xerr = s3_errs[i], 
                     fmt = ".", capsize = 5, color = "red")
        cntr1 = ax1.contourf(X, Y, Z, 200, cmap = "nipy_spectral")
        ax1.set_xlabel(r"$S_3$"); ax1.set_ylabel(r"$A_{Im}$")
        ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")
    
        plt.subplots_adjust(hspace=0.2)
        fig.colorbar(cntr0, ax = ax0)
        fig.colorbar(cntr1, ax = ax1)
        plt.savefig( f"MinPlots/Fit_NLL_phi_bin{i}.pdf" )
        plt.show()

    del fig, axes, ax0, ax1, cntr0, cntr1
    del N, delta, fl, afb, X, Y, Z 
    del flmin_i, afbmin_i, flmin, afbmin, fl0, fl1, afb0, afb1