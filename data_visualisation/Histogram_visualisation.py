# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:34:15 2022

@author: Zhihao
"""
# %%
import numpy as np
import scipy as sp
#import scipy.integrate as spi
#import scipy.optimize as spo
#import math as m
import pandas as pd
import matplotlib.pyplot as plt


#rom scipy import stats
#from scipy.optimize import curve_fit

params = {
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 8,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   } 
plt.rcParams.update(params)
# %%
"""
Importing dataset
"""
data_path = '../data/'

dataset_path = f'{data_path}total_dataset.pkl' 
# "real" data from LHCb to analyse, with signal decay and background
signal_path = f'{data_path}signal.pkl' 
# simulated signal decay, as per Standard Model

jpsi_path = f'{data_path}jpsi.pkl' 
# B0 -> J/psi K*0, with J/psi -> mumu
psi2S_path = f'{data_path}psi2S.pkl' 
# B0 -> psi(2S) K*0 with psi(2S) -> mumu

jpsi_mu_k_swap_path = f'{data_path}jpsi_mu_k_swap.pkl' 
# B0 -> J/psiK*0 with mu reconstructed as K, and K as mu
jpsi_mu_pi_swap_path = f'{data_path}jpsi_mu_pi_swap.pkl' 
# B0 -> J/psiK*0 with mu reconstructed as pi, and pi as mu
k_pi_swap_path = f'{data_path}k_pi_swap.pkl' 
# signal decay with K reconstructed as pi, and pi as K

phimumu_path = f'{data_path}phimumu.pkl' 
# B0_s -> phimumu with phi -> KK and a K reconstructed as pi
pKmumu_piTok_kTop_path = f'{data_path}pKmumu_piTok_kTop.pkl' 
# Lambda0_b -> pKmumu with p reconstructed as K, K as pi
pKmumu_piTop_path = f'{data_path}pKmumu_piTop.pkl' 
# Lambda0_b -> pKmumu with p reconstructed as pi

acceptance_mc_path = f'{data_path}acceptance_mc.pkl' 
# flat in three angular variables and q^2

# load dataset
signal             = pd.read_pickle(signal_path)
dataset            = pd.read_pickle(dataset_path)
jpsi               = pd.read_pickle(jpsi_path)
psi2S              = pd.read_pickle(psi2S_path)
jpsi_mu_k_swap     = pd.read_pickle(jpsi_mu_k_swap_path)
jpsi_mu_pi_swap    = pd.read_pickle(jpsi_mu_pi_swap_path)
k_pi_swap          = pd.read_pickle(k_pi_swap_path)
phimumu            = pd.read_pickle(phimumu_path)
pKmumu_piTok_kTop  = pd.read_pickle(pKmumu_piTok_kTop_path)
pKmumu_piTop       = pd.read_pickle(pKmumu_piTop_path)
acceptance_mc      = pd.read_pickle(acceptance_mc_path)

Datalist = [dataset, signal, acceptance_mc, jpsi, jpsi_mu_k_swap, jpsi_mu_pi_swap,
            psi2S, k_pi_swap, phimumu, pKmumu_piTok_kTop, pKmumu_piTop]
Datatitle = ['dataset','signal', 'acceptance_mc','jpsi','jpsi_mu_k_swap',
             'jpsi_mu_pi_swap','psi2S', 'k_pi_swap','phimumu','pKmumu_piTok_kTop',
             'pKmumu_piTop']
#%%
"""
Add Probability of correct reconstruction
"""
for data in Datalist:
    P_correct = data['mu_plus_MC15TuneV1_ProbNNmu'] *\
                data['mu_minus_MC15TuneV1_ProbNNmu'] *\
                data['K_MC15TuneV1_ProbNNk'] *\
                data['Pi_MC15TuneV1_ProbNNpi']
    data['P_correct'] = P_correct


#%%
"""
Histograms of general quantities
"""

generals = ["P_correct", "phi", "costhetak", "costhetal", "q2", 'B0_MM']

rc = len(Datalist)
c = int(np.sqrt(rc))
r = int( np.ceil( rc / c) )
figsize = (c * 4, r * 2.1)

for para in generals:
    fig = plt.figure(figsize = figsize, tight_layout = True)
    fig.suptitle(para, size = "15")
    for i in range(len(Datalist)):
       plt.subplot(r, c, i + 1)
       plt.hist(Datalist[i][para], bins = 50, density=True)
       plt.xlabel(r'$q^2$')
       plt.ylabel(r'Number of events')
       plt.title(Datatitle[i])
       if Datatitle[i] == "dataset":
           xlim = plt.xlim()
       plt.xlim(xlim)
       plt.grid()
    plt.show()

# %%
# """
# Historgram of q^2
# """
# para = 'q2'
# xlim = [0,25]

# # simulated data
# plt.hist(signal[para], bins=50, density=True)
# plt.xlabel(r'$q^2$')
# plt.ylabel(r'Number of events')
# plt.title('signal')
# plt.xlim(xlim)
# plt.grid()
# plt.show()


# fig = plt.figure(figsize=[9,16],tight_layout=True)
# for i in range(len(Datalist)):
#    plt.subplot(4, 3, i+1)
#    plt.hist(Datalist[i][para], bins=50, density=True)
#    plt.xlabel(r'$q^2$')
#    plt.ylabel(r'Number of events')
#    plt.title(Datatitle[i])
#    plt.xlim(xlim)
#    plt.grid()
# plt.show()

# plt.hist(acceptance_mc[para], bins=50, density=True)
# plt.xlabel(r'$q^2$')
# plt.ylabel(r'Number of events')
# plt.title('acceptance_mc')
# plt.xlim(xlim)
# plt.grid()
# plt.show()

# # real dataset
# plt.hist(dataset[para], bins=50, density=True)
# plt.xlabel(r'$q^2$')
# plt.ylabel(r'Number of events')
# plt.title('dataset')
# plt.xlim(xlim)
# plt.grid()
# plt.show()



# # %%
# """
# Historgram of cos_theta_l
# """
# para = 'costhetal'
# xlim = [-1,1]

# # simulated data
# plt.hist(signal[para], bins=50, density=True)
# plt.xlabel(r'$\cos{{\theta}_l}$')
# plt.ylabel(r'Number of events')
# plt.title('signal')
# plt.xlim(xlim)
# plt.grid()
# plt.show()

# Datalist = [jpsi,psi2S,jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap,phimumu,pKmumu_piTok_kTop,pKmumu_piTop]
# Datatitle = ['jpsi','psi2S','jpsi_mu_k_swap','jpsi_mu_pi_swap','k_pi_swap','phimumu','pKmumu_piTok_kTop','pKmumu_piTop']
# fig = plt.figure(figsize=[9,16],tight_layout=True)
# for i in range(len(Datalist)):
#    plt.subplot(5,2,i+1)
#    plt.hist(Datalist[i][para], bins=50, density=True)
#    plt.xlabel(r'$\cos{{\theta}_l}$')
#    plt.ylabel(r'Number of events')
#    plt.title(Datatitle[i])
#    plt.xlim(xlim)
#    plt.grid()
# plt.show()

# plt.hist(acceptance_mc[para], bins=50, density=True)
# plt.xlabel(r'$\cos{{\theta}_l}$')
# plt.ylabel(r'Number of events')
# plt.title('acceptance_mc')
# plt.xlim(xlim)
# plt.grid()
# plt.show()

# # real dataset
# plt.hist(dataset[para], bins=50, density=True)
# plt.xlabel(r'$\cos{{\theta}_l}$')
# plt.ylabel(r'Number of events')
# plt.title('dataset')
# plt.xlim(xlim)
# plt.grid()
# plt.show()
# # %%
# """
# Historgram of m_B0
# """
# para = 'B0_MM' # bins of dataframe
# xlim = [5000,5800]

# # simulated data
# plt.hist(signal[para], bins=50, density=True)
# plt.xlabel(r'$m_{B^0}$')
# plt.ylabel(r'Number of events')
# plt.title('signal')
# plt.xlim(xlim)
# plt.grid()
# plt.show()

# Datalist = [jpsi,psi2S,jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap,phimumu,pKmumu_piTok_kTop,pKmumu_piTop]
# Datatitle = ['jpsi','psi2S','jpsi_mu_k_swap','jpsi_mu_pi_swap','k_pi_swap','phimumu','pKmumu_piTok_kTop','pKmumu_piTop']
# fig = plt.figure(figsize=[9,16],tight_layout=True)
# for i in range(len(Datalist)):
#    plt.subplot(5,2,i+1)
#    plt.hist(Datalist[i][para], bins=50, density=True)
#    plt.xlabel(r'$m_{B^0}$')
#    plt.ylabel(r'Number of events')
#    plt.title(Datatitle[i])
#    plt.xlim(xlim)
#    plt.grid()
# plt.show()

# plt.hist(acceptance_mc[para], bins=50, density=True)
# plt.xlabel(r'$m_{B^0}$')
# plt.ylabel(r'Number of events')
# plt.title('acceptance_mc')
# plt.xlim(xlim)
# plt.grid()
# plt.show()

# # real dataset
# plt.hist(dataset[para], bins=50, density=True)
# plt.xlabel(r'$m_{B^0}$')
# plt.ylabel(r'Number of events')
# plt.title('dataset')
# plt.xlim(xlim)
# plt.grid()
# plt.show()
