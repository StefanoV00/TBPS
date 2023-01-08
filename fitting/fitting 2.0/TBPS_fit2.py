# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:00:50 2022

@author: Stefano
"""

from TBPS_bin_functions import *
from TBPS_pdf_functions2 import *
from TBPS_fit_function2 import *
from classification.data_loader import load_classified_total, load_classified_acceptance

import pickle
import numpy as np
from scipy.integrate import quad
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm

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

path = 'classification/data_processed/Classified total Dataset.bz2'
classdata = pd.read_pickle(path)
data = classdata.reset_index()

# Create acceptance_l: a list of np.polynomial, one for each bin. 
with open("data/acceptance_function_coefs.pkl", "rb") as f:
    acceptance_coefs = pickle.load(f)
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

del path, classdata, f, coeffsi, acceptance_coefs

#%% GIVE A LOOK TO THE NLLs
yn = input("Do you want to do the many LONG subplots?")
if yn:
    N = 500
    delta = 0.1
    for i in range(len(bindata)):
    
        fig, axes = plt.subplots(2,1, figsize = (7, 10), 
                                gridspec_kw={'height_ratios': [1.3, 1]})
        plt.suptitle(r"NLL of $cos(\theta_l)$;"+f" Bin {i}: {mybins[i]} $GeV^2$")
        ax0 = axes[0]; ax1 = axes[1]
        
        fl = np.linspace(-1, 1, N); afb = np.linspace(-1, 1, N)
        X,Y = np.meshgrid(fl,afb)
        Z = nLL_ctl(bindata, acceptance_l, [X, Y], i)
        cntr0 = ax0.contourf(X, Y, Z, 300, cmap = "nipy_spectral")
        ax0.set_xlabel(r"$F_l$"); ax0.set_ylabel(r"$A_fb$")
        ax0.annotate ("a)", (-0.15, 1.00), xycoords = "axes fraction")
        
        plt.subplot(2,1,2)
        afbmin_i, flmin_i = np.unravel_index(np.nanargmin(Z), Z.shape)
        flmin = fl[flmin_i]; afbmin = afb[afbmin_i]
        fl0 = max(-1, flmin - delta);    fl1 = min(1, flmin + delta)
        afb0 = max(-1, afbmin - delta); afb1 = min(1, afbmin + delta)
        fl = np.linspace(fl0, fl1, N); afb = np.linspace(afb0, afb1, N)
        X,Y = np.meshgrid(fl,afb)
        Z = nLL_ctl(bindata, acceptance_l, [X, Y], i)
        cntr1 = ax1.contourf(X, Y, Z, 300, cmap = "nipy_spectral")
        ax1.set_xlabel(r"$F_l$"); ax1.set_ylabel(r"$A_fb$")
        ax1.annotate ("b)", (-0.15, 1.00), xycoords = "axes fraction")
    
        plt.subplots_adjust(hspace=0.2)
        fig.colorbar(cntr0, ax = ax0)
        fig.colorbar(cntr1, ax = ax1)
        plt.savefig( f"MinPlots/NLL_ctl_bin{i}.pdf" )
        plt.show()
    
    del fig, axes, ax0, ax1, cntr0, cntr1
    del N, delta, fl, afb, X, Y, Z 
    del flmin_i, afbmin_i, flmin, afbmin, fl0, fl1, afb0, afb1
#
#%% MINIMISE WITH MINUIT
f = ctl_acc_PDF
Nlayers = 4 #should take less than 60s per bin, triplicating for Nlayers = 5.
guess = [[0.95, 0],
         [0.95, 0],
         [0.95, 0],
         [0.70, 0],
         [0.70, 0],
         [0.50, 0],
         [0.50, 0],
         [0.60, 0],
         [0.90, 0],
         [0.50, 0]]
limits = [ [(0.8, 1),(-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)],
           [(-1, 1), (-1, 1)] ]

guess_per_bin = False
if len(guess) == len(mybins):
    guess_per_bin = True
if len(limits) == len(mybins):
    limits_per_bin = True
(fls, afbs), (fl_errs, afb_errs) = fit_cts(bindata, f, guess, 
                                           guess_per_bin = guess_per_bin, 
                                           limits = limits,
                                           limits_per_bin = limits_per_bin,
                                           Nlayers = Nlayers, vary = 0.1,
                                           args = acceptance_l, 
                                           flexible = True)
del guess, guess_per_bin
# plt.figure(figsize=(8, 5))
# plt.subplot(221)
# results_to_check.draw_mnprofile('fl', bound=3)
# plt.subplot(222)
# results_to_check.draw_mnprofile('afb', bound=3)
# plt.tight_layout()
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
plt.suptitle("Results of Fitting Accepteance-Modulated Costhetal PDF")
ns = np.linspace(0, len(bindata) - 1, len(bindata))
ax1.errorbar(ns, fls, yerr=fl_errs, fmt = ".", capsize = 4, label=r'$F_L$')
ax2.errorbar(ns, afbs, yerr=afb_errs, fmt = ".", capsize = 4, label=r'$A_{FB}$')
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$F_L$')
ax2.set_ylabel(r'$A_{FB}$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.savefig("MinPlots/Fl_Afb_Coeffs")
plt.show()

print("Checking Normalisation of PDFs")
A_ctl, A_ctk, A_ctl_acc = [], [], []
for i in range(len(bindata)):
    A_ctl.append(quad(ctl_PDF, -1, 1, args = ([fls[i], afbs[i]]))[0])
    A_ctk.append(quad(ctk_PDF, -1, 1, args = fls[i])[0])
    A_ctl_acc.append(quad(ctl_acc_PDF, -1, 1,
                                args = ([fls[i], afbs[i]], acceptance_l[i]))[0])
    print("Area of ctl-pdf:", A_ctl[-1])
    print("Area of ctk-pdf:", A_ctk[-1])
    print("Area of ctl-acc-pdf", A_ctl_acc[-1],"\n")

#%% SEE RESULTS FOR COSTHETA_L
# Prepare for plotting
rc = len(bindata)
c = int(np.sqrt(rc))
r = int( np.ceil( rc / c) )
figsize = (c * 4, r * 2.2)

plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(bindata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetal vs PDF")
    Nbins = 25
    ctli = bindata[i]['costhetal']
    hist, _bins, _ = plt.hist(ctli, bins = Nbins, density = True)
    x = np.linspace(-1, 1, 4 * Nbins)
    #pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
    y = ctl_PDF(x, [fls[i], afbs[i]]) #* pdf_multiplier
    plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
Fl={round(fls[i], 3)},Afb={round(afbs[i],3)}')
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'PDF')
    plt.legend()
    plt.grid()
plt.savefig(f'MinPlots/Ctl_fit')
plt.show()

plt.figure(figsize = figsize, tight_layout = True)
plt.suptitle("Observed Costhetal vs Acceptance-modulated PDF")
for i in range(len(bindata)):
    plt.subplot(r, c, i + 1)
    Nbins = 25
    ctli = bindata[i]['costhetal']
    hist, _bins, _ = plt.hist(ctli, bins = Nbins, density = True)
    x = np.linspace(-1, 1, 4 * Nbins)
    #pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
    y = ctl_acc_PDF(x, [fls[i], afbs[i]], acceptance_l[i]) / A_ctl_acc[i]
    plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
Fl={round(fls[i], 3)},Afb={round(afbs[i],3)}')
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'PDF')
    plt.legend()
    plt.grid()
plt.savefig(f'MinPlots/Ctl_Acc_fit')
plt.show()

# SEE RESULTS FOR COSTHETA_K
plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(bindata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetak vs PDF")
    Nbins = 25
    ctki = bindata[i]['costhetak']
    hist, _bins, _ = plt.hist(ctki, bins = Nbins, density = True)
    x = np.linspace(-1, 1, 4 * Nbins)
    #pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
    y = ctk_PDF(x, fls[i]) #* pdf_multiplier
    plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
Fl={round(fls[i], 3)}')
    plt.xlabel(r'$cos(\theta_k)$')
    plt.ylabel(r'PDF')
    plt.legend()
    plt.grid()
plt.savefig(f'MinPlots/Ctk_fit')
plt.show()

plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(bindata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetak vs Acceptance-Modulated PDF")
    Nbins = 25
    ctki = bindata[i]['costhetak']
    hist, _bins, _ = plt.hist(ctki, bins = Nbins, density = True)
    x = np.linspace(-1, 1, 4 * Nbins)
    #pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
    y = ctk_PDF(x, fls[i]) #* pdf_multiplier
    plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
Fl={round(fls[i], 3)}')
    plt.xlabel(r'$cos(\theta_k)$')
    plt.ylabel(r'PDF')
    plt.legend()
    plt.grid()
plt.savefig(f'MinPlots/Ctk_Acc_fit')
plt.show()

del figsize, r, rc, c
del ctli, ctki, hist, _bins, i, Nbins, x, y
del fig, ax1, ax2, ns



