# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:15:10 2022

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

# Create acceptance_l: a list of np.polynomial, one for each bin. 
with open("data/acceptance_function_coefs.pkl", "rb") as file:
    acceptance_coefs = pickle.load(file)
acceptance_l = []
for coeffsi in acceptance_coefs:
    acceptance_l.append(np.poly1d(coeffsi))

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
Fl  = [  0.263,  0.660,  0.876,  0.611,  0.579,  0.493,  0.349,  0.354,  0.690]
Afb = [ -0.003, -0.191, -0.118,  0.025,  0.152,  0.318,  0.411,  0.305, -0.075]

coeffs = np.column_stack((Fl, Afb))
del file, coeffsi, acceptance_coefs

#%%
M = 500
#GENERATE A TESTDATA WITH KNOWN ACC_PDF, AND KNOW COEFFICIENTS
ctl = []
q2 = []
testdata = []
for i in tqdm(range(len(testbins)), "Creating TestData"):
    x_l = np.linspace(-1, 1, 10000)
    C_l = max(ctl_acc_PDF(x_l, coeffs[i], acceptance_l[i]))
    size = 0
    ctli = []
    while size < M:
        y_l = np.random.uniform(-1, 1)
        p_l = np.random.uniform(0, C_l)
        if ctl_acc_PDF(y_l, coeffs[i], acceptance_l[i]) >= p_l:
            size += 1
            ctli.append(y_l)
            ctl.append(y_l)
            q2.append(np.random.uniform(testbins[i][0], testbins[i][1]))
    testdatai = pd.DataFrame(ctli, columns = ["costhetal"])
    testdata.append(testdatai)
test = pd.DataFrame(np.column_stack([q2, ctl]), columns = ["q2", "costhetal"])
test.to_pickle(f"TestPlots/TestingData_All{M}.pkl") 
del i, x_l, C_l, size, ctli
del y_l, p_l, testdatai, q2, ctl

#%% FIT TESTDATA
f = ctl_acc_PDF
limits = 0
Nlayers = 7 #should take 3 minutes 
guess = [0,0]

guess_per_bin = False
limits_per_bin = False
if len(guess) == len(testbins):
    guess_per_bin = True
if hasattr(limits, "__len__"):
    if len(limits) == len(testbins):
        limits_per_bin = True
(Fls, Afbs), (Fl_errs, Afb_errs), Fits = fit_angs(testdata, f, guess, 
                                           guess_per_bin = guess_per_bin, 
                                           limits = limits,
                                           limits_per_bin = limits_per_bin,
                                           Nlayers = Nlayers, vary = 0,
                                           args = acceptance_l, 
                                           flexible = True)
del guess, limits, guess_per_bin, limits_per_bin, Nlayers

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
plt.suptitle(r"Fitting Test")
ns = np.linspace(0, len(testdata) - 1, len(testdata))
ax1.errorbar(ns, Fls, yerr=Fl_errs, fmt = ".", capsize = 4, label="Fit",c="r")
ax1.plot(ns, Fl, "x", label='True Value', c = "b", zorder = 4)
ax2.errorbar(ns, Afbs, yerr=Afb_errs, fmt = ".", capsize = 4, label='Fit',c="r")
ax2.plot(ns, Afb, "x", label='True Value', c = "b", zorder = 4)
ax1.grid(alpha = 0.3)
ax2.grid(alpha = 0.3)
ax1.set_ylabel(r'$F_L$')
ax2.set_ylabel(r'$A_{FB}$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig(f"TestPlots/Test_Coeffs_{M}Datapoint")
plt.show()
del fig, ax1, ax2, ns

# RESULTS of TESTDATA
if "ctl" in f.__name__:
    N = 100
    delta = 0.2
    for i in range(len(testdata)):
    
        fig, axes = plt.subplots(2,1, figsize = (7, 10), 
                                gridspec_kw={'height_ratios': [1.3, 1]})
        plt.suptitle(r"NLL of $cos(\theta_l)$;"+f" Bin {i}: {testbins[i]} $GeV^2$")
        ax0 = axes[0]; ax1 = axes[1]
        
        fl = np.linspace(0, 1, N); afb = np.linspace(-1, 1, N)
        X,Y = np.meshgrid(fl,afb)
        Z = nLL_ctl(testdata, acceptance_l, [X, Y], i)
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
        Z = nLL_ctl(testdata, acceptance_l, [X, Y], i)
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

    del fig, axes, ax0, ax1, cntr0, cntr1
    del N, delta, fl, afb, X, Y, Z 
    del flmin_i, afbmin_i, flmin, afbmin, fl0, fl1, afb0, afb1

#SHOWING RESULT FOR COSTHETAL
# Prepare for plotting
rc = len(testdata)
c = int(np.sqrt(rc))
r = int( np.ceil( rc / c) )
figsize = (c * 5, r * 2.25)

plt.figure(figsize = figsize, tight_layout = True)
for i in range(len(testdata)):
    plt.subplot(r, c, i + 1)
    plt.suptitle("Observed Costhetal vs PDF")
    ctli = testdata[i]['costhetal']
    bwidth = 2 * iqr(ctli) / np.cbrt(len(ctli))
    Nbins = int(2 / bwidth)
    heights,edges = np.histogram(ctli, Nbins)
    centres = (edges[:-1] + edges[1:])/2.
    err = np.sqrt(heights)
    heights = heights / (bwidth * len(ctli))
    err = err /(bwidth * len(ctli))
    plt.errorbar(centres, heights, yerr=err, fmt='.', capsize = 4)
    x = np.linspace(-1, 1, 100)
    y1 = ctl_acc_PDF(x, [Fl[i], Afb[i]], acceptance_l[i])
    y2 = ctl_acc_PDF(x, [Fls[i], Afb[i]], acceptance_l[i])
    title = f" Bin {i}: {testbins[i]} $GeV^2$;   "
    values = f"{M} Datapoints per Bin"
    plt.annotate(title+values, (0, 1), xycoords = "axes fraction", va='bottom',
                 ha = "left", size = 10)
    plt.plot(x, y1, label = "True PDF")
    plt.plot(x, y2, label = "Fit PDF")
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'PDF')
    plt.ylim(0, max(plt.ylim()[1], 1))
    plt.xlim(-1, 1)
    plt.legend()
    plt.grid()
plt.savefig(f'TestPlots/Test_Ctl_fit_{M}Datapoints')
plt.show()

del rc, r, c, figsize
del i, ctli, Nbins, bwidth, heights, edges, centres, err
del x, y1, y2
del title, values
#%% CLEAN MEMORY
yn = input("Do you want to delete Test variables)? [y/n]")
if yn:
    del Fl, Afb, coeffs, M, acceptance_l
    del Fls, Afbs, Fl_errs, Afb_errs, Fits
    del test, testbins, testdata, yn
            
        
    


