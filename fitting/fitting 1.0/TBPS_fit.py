# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:00:50 2022

@author: Stefano
"""

from TBPS_bin_functions import *
from TBPS_pdf_functions import *
from TBPS_fit_functions import *

import numpy as np
from scipy.integrate import quad
import pandas as pd

import matplotlib.pyplot as plt

#%% MINIMISE WITH MINUIT

guess = [0.5, 0]
(fls, afbs), (fl_errs, afb_errs), results_to_check = fit_cts(nLL_ctl,
    guess = guess, bin_number_to_check = 5)

plt.figure(figsize=(8, 5))
plt.subplot(221)
results_to_check.draw_mnprofile('afb', bound=3)
plt.subplot(222)
results_to_check.draw_mnprofile('fl', bound=3)
plt.tight_layout()
plt.show()


rc = len(data)
c = int(np.sqrt(rc))
r = int( np.ceil( rc / c) )
figsize = (c * 4, r * 2.2)

#%%% SEE RESULTS FOR COSTHETA_L
plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(data)):
    plt.subplot(r, c, i + 1)
    Nbins = 25
    ctli = data[i]['costhetal']
    hist, _bins, _ = plt.hist(ctli, bins = Nbins, density = True)
    x = np.linspace(-1, 1, 4 * Nbins)
    #pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
    y = ctl_PDF(x, fls[i], afbs[i]) #* pdf_multiplier
    plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
Fl={round(fls[i], 3)},Afb={round(afbs[i],3)}')
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'PDF')
    plt.legend()
    plt.grid()
plt.savefig(f'MinPlots/Ctl_fit')
plt.show()

del Nbins, ctli, x

#%%% SEE RESULTS FOR COSTHETA_K
plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(data)):
    plt.subplot(r, c, i + 1)
    Nbins = 25
    ctki = data[i]['costhetak']
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
ns = np.linspace(0, len(data) - 1, len(data))
ax1.errorbar(ns, fls, yerr=fl_errs, fmt = ".", capsize = 4, label=r'$F_L$')
ax2.errorbar(ns, afbs, yerr=afb_errs, fmt = ".", capsize = 4, label=r'$A_{FB}$')
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$F_L$')
ax2.set_ylabel(r'$A_{FB}$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()

for i in range(len(data)):
    print("Area of ctl-pdf:", quad(ctl_PDF, -1, 1, args = (fls[i], afbs[i])))
    print("Area of ctk-pdf:", quad(ctk_PDF, -1, 1, args = fls[i]))

del figsize, guess, r, rc, c, results_to_check
del ctli, ctki, hist, _bins, i, Nbins, x, y
del fig, ax1, ax2, ns



