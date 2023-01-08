# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:00:50 2022

@author: Stefano
"""

from TBPS_bin_functions import *
from TBPS_pdfs1 import *
from TBPS_pdfs2 import *
from TBPS_pdf_full import *
from TBPS_fit_functions import *
from classification.data_loader import load_classified_total, load_classified_acceptance

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

path = 'classification/data_processed/Hard_selected_dataset_q2cut.pkl'
classdata = pd.read_pickle(path)
data = classdata.reset_index()

# Create acceptance_l: a list of np.polynomial, one for each bin. 
with open("data/acceptance_function_coefs.pkl", "rb") as file:
    acceptance_coefs = pickle.load(file)
acceptance_l = []
for coeffsi in acceptance_coefs:
    acceptance_l.append(np.poly1d(coeffsi))

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

del path, classdata, file, coeffsi, acceptance_coefs

#%% GIVE A LOOK TO THE NLLs
N = 100
fl = np.linspace(0, 1, N); afb = np.linspace(-1, 1, N)
X,Y = np.meshgrid(fl,afb)
Z = []
guess = []
for i in range(len(bindata)):
    Z.append(nLL_ctl(bindata, acceptance_l, [X, Y], i))
    afbmin_i, flmin_i = np.unravel_index(np.nanargmin(Z[-1]), Z[-1].shape)
    flmin = fl[flmin_i]; afbmin = afb[afbmin_i]
    guess.append([flmin, afbmin])
    
# yn = input("Do you want to do the many NLL subplots (takes a bit)? [y/n]")
# if yn:
#     N = 200
#     delta = 0.1
#     for i in range(len(bindata)):
    
#         fig, axes = plt.subplots(2,1, figsize = (7, 10), 
#                                 gridspec_kw={'height_ratios': [1.3, 1]})
#         plt.suptitle(r"NLL of $cos(\theta_l)$;"+f" Bin {i}: {mybins[i]} $GeV^2$")
#         ax0 = axes[0]; ax1 = axes[1]
        
#         cntr0 = ax0.contourf(X, Y, Z[i], 200, cmap = "nipy_spectral")
#         ax0.set_xlabel(r"$F_l$"); ax0.set_ylabel(r"$A_fb$")
#         ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
        
#         plt.subplot(2,1,2)
#         flmin = guess[i][0]; afbmin = guess[i][1]
#         fl0 = max(0, flmin - delta);    fl1 = min(1, flmin + delta)
#         afb0 = max(-1, afbmin - delta); afb1 = min(1, afbmin + delta)
#         fl = np.linspace(fl0, fl1, N); afb = np.linspace(afb0, afb1, N)
#         Xmin,Ymin = np.meshgrid(fl,afb)
#         Zmin = nLL_ctl(bindata, acceptance_l, [X, Y], i)
#         cntr1 = ax1.contourf(Xmin, Ymin, Zmin, 200, cmap = "nipy_spectral")
#         ax1.set_xlabel(r"$F_l$"); ax1.set_ylabel(r"$A_fb$")
#         ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")
    
#         plt.subplots_adjust(hspace=0.2)
#         fig.colorbar(cntr0, ax = ax0)
#         fig.colorbar(cntr1, ax = ax1)
#         #plt.savefig( f"MinPlots/NLL_ctl_bin{i}.pdf" )
#         plt.show()

#     del fig, axes, ax0, ax1, cntr0, cntr1
#     del N, delta 
#     del Xmin, Ymin, Zmin, fl0, fl1, afb0, afb1
del i, N, fl, afb, X, Y, Z
del flmin_i, afbmin_i, flmin, afbmin
    
#%% MINIMISE WITH MINUIT
f = ctl_acc_PDF
limits = 0
Nlayers = 7 #should take less than 60s per bin, triplicating for Nlayers = 5.
# guess = [0,0]
# guess = [[0.99, -0.075],
#           [0.95, 0],
#           [0.95, 0],
#           [0.70, 0],
#           [0.70, 0],
#           [0.50, 0],
#           [0.50, 0],
#           [0.60, 0],
#           [0.90, 0],
#           [0.50, 0]]
limits = [ [(0.98, 1),(-0.1, -0.05)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)],
            [(0, 1), (-1, 1)] ]

guess_per_bin = False
limits_per_bin = False
if len(guess) == len(mybins):
    guess_per_bin = True
if hasattr(limits, "__len__"):
    if len(limits) == len(mybins):
        limits_per_bin = True
(fls, afbs), (fl_errs, afb_errs), fits = fit_angs(bindata, f, guess, 
                                           guess_per_bin = guess_per_bin, 
                                           limits = limits,
                                           limits_per_bin = limits_per_bin,
                                           Nlayers = Nlayers, vary = 0,
                                           args = acceptance_l, 
                                           flexible = True)
del guess, limits, guess_per_bin, limits_per_bin, Nlayers

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
plt.suptitle(r"Results of Fitting $cos(\theta_l)$")
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

#%%
yn = input("Do you want to do the many NLL subplots (takes a bit)? [y/n]")
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
        Z = nLL_ctl(bindata, acceptance_l, [X, Y], i)
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
        Z = nLL_ctl(bindata, acceptance_l, [X, Y], i)
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

    del fig, axes, ax0, ax1, cntr0, cntr1
    del N, delta, fl, afb, X, Y, Z 
    del flmin_i, afbmin_i, flmin, afbmin, fl0, fl1, afb0, afb1
    
#%% SEE RESULTS FOR COSTHETA_L
# Prepare for plotting
rc = len(bindata)
c = int(np.sqrt(rc))
r = int( np.ceil( rc / c) )
figsize = (c * 5, r * 2.25)

plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(bindata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetal vs PDF")
    ctli = bindata[i]['costhetal']
    bwidth = 2 * iqr(ctli) / np.cbrt(len(ctli))
    Nbins = int(2 / bwidth)
    heights,edges = np.histogram(ctli, Nbins)
    centres = (edges[:-1] + edges[1:])/2.
    err = np.sqrt(heights)
    heights = heights / (bwidth * len(ctli))
    #ERRORS MAY BE WRONG
    err = err /(bwidth * len(ctli))
    plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
    x = np.linspace(-1, 1, 100)
    y1 = ctl_PDF(x, [fls[i], afbs[i]])
    y2 = ctl_acc_PDF(x, [fls[i], afbs[i]], acceptance_l[i])
    title = f" Bin {i}: {mybins[i]} $GeV^2$;   "
    values = "$F_l$="+f"{round(fls[i], 3)},"+r"$A_{fb}$="+f"{round(afbs[i],3)}"
    plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                 ha = "left", size = 10)
    plt.plot(x, y1, label = "PDF")
    plt.plot(x, y2, label = "Acc - PDF")
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'PDF')
    plt.ylim(0, max(plt.ylim()[1], 1))
    plt.xlim(-1, 1)
    plt.legend()
    plt.grid()
plt.savefig(f'MinPlots/Ctl_fit')
plt.show()

# plt.figure(figsize = figsize, tight_layout = True)
# plt.suptitle("Observed Costhetal vs Acceptance-modulated PDF")
# for i in range(len(bindata)):
#     plt.subplot(r, c, i + 1)
#     ctli = bindata[i]['costhetal']
#     bwidth = 2 * iqr(ctli) / np.cbrt(len(ctli))
#     Nbins = int(2 / bwidth)
#     heights,edges = np.histogram(ctli, Nbins, density)
#     centres = (edges[:-1] + edges[1:])/2.
#     err = np.sqrt(heights)
#     plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
#     x = np.linspace(-1, 1, 100)
#     y = ctl_acc_PDF(x, [fls[i], afbs[i]], acceptance_l[i], 
#                     label = "Acceptance Modulated")
#     plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
# Fl={round(fls[i], 3)},Afb={round(afbs[i],3)}')
#     plt.xlabel(r'$cos(\theta_l)$')
#     plt.ylabel(r'PDF')
#     plt.legend()
#     plt.grid()
# plt.savefig(f'MinPlots/Ctl_Acc_fit')
# plt.show()

# SEE RESULTS FOR COSTHETA_K
plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(bindata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetak vs PDF")
    ctki = bindata[i]['costhetak']
    bwidth = 2 * iqr(ctki) / np.cbrt(len(ctki))
    Nbins = int(2 / bwidth)
    heights,edges = np.histogram(ctki, Nbins)
    centres = (edges[:-1] + edges[1:])/2.
    err = np.sqrt(heights)
    heights = heights / (bwidth * len(ctki))
    #ERRORS MAYBE WRONG
    err = err /(bwidth * len(ctki))
    plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
    x = np.linspace(-1, 1, 100)
    y1 = ctk_PDF(x, fls[i])
    #y2 = ctk_acc_PDF(x, fls[i], acceptance_k[i])
    plt.plot(x, y1, label = "PDF")
    #plt.plot(x, y, label = "Acc - PDF")
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
plt.savefig(f'MinPlots/Ctk_fit')
plt.show()

# plt.figure(figsize = figsize, tight_layout = True)
# for i in range(len(bindata)):
#     plt.subplot(r, c, i + 1)
#     plt.suptitle("Observed Costhetak vs Acceptance-Modulated PDF")
#     ctki = bindata[i]['costhetak']
#     bwidth = 2 * iqr(ctki) / np.cbrt(len(ctki))
#     Nbins = int(2 / bwidth)
#     hist, _bins, _ = plt.hist(ctki, bins = Nbins, density = True)
#     x = np.linspace(-1, 1, 4 * Nbins)
#     y = ctk_PDF(x, fls[i]) #* pdf_multiplier
#     plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
# Fl={round(fls[i], 3)}')
#     plt.xlabel(r'$cos(\theta_k)$')
#     plt.ylabel(r'PDF')
#     plt.legend()
#     plt.grid()
# plt.savefig(f'MinPlots/Ctk_Acc_fit')
# plt.show()

del figsize, r, rc, c
del ctli, ctki, i, Nbins, centres, edges
del x, y1, y2, err



