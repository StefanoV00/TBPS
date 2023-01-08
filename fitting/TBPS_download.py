# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:17:35 2022

@author: Stefano
"""
import pandas as pd

"""
Importing dataset
"""
data_path = '../data/'

# load dataset
dataset            = pd.read_pickle(f'{data_path}total_dataset.pkl')
# "real" data from LHCb to analyse, with signal decay and background
signal             = pd.read_pickle(f'{data_path}signal.pkl')
# simulated signal decay, as per Standard Model
acceptance_mc      = pd.read_pickle(f'{data_path}acceptance_mc.pkl' )
# flat in three angular variables and q^2

jpsi               = pd.read_pickle(f'{data_path}jpsi.pkl')
# B0 -> J/psi K*0, with J/psi -> mumu
jpsi_mu_k_swap     = pd.read_pickle(f'{data_path}jpsi_mu_k_swap.pkl' )
# B0 -> J/psiK*0 with mu reconstructed as K, and K as mu
jpsi_mu_pi_swap    = pd.read_pickle(f'{data_path}jpsi_mu_pi_swap.pkl')
# B0 -> J/psiK*0 with mu reconstructed as pi, and pi as mu

psi2S              = pd.read_pickle(f'{data_path}psi2S.pkl')
# B0 -> psi(2S) K*0 with psi(2S) -> mumu
k_pi_swap          = pd.read_pickle(f'{data_path}k_pi_swap.pkl')
# signal decay with K reconstructed as pi, and pi as K

phimumu            = pd.read_pickle(f'{data_path}phimumu.pkl')
# B0_s -> phimumu with phi -> KK and a K reconstructed as pi
pKmumu_piTok_kTop  = pd.read_pickle(f'{data_path}pKmumu_piTok_kTop.pkl')
# Lambda0_b -> pKmumu with p reconstructed as K, K as pi
pKmumu_piTop       = pd.read_pickle(f'{data_path}pKmumu_piTop.pkl')
# Lambda0_b -> pKmumu with p reconstructed as pi


Datalist = [dataset, signal, acceptance_mc, jpsi, jpsi_mu_k_swap, jpsi_mu_pi_swap,
            psi2S, k_pi_swap, phimumu, pKmumu_piTok_kTop, pKmumu_piTop]
Datatitle = ['dataset','signal', 'acceptance_mc','jpsi','jpsi_mu_k_swap',
             'jpsi_mu_pi_swap','psi2S', 'k_pi_swap','phimumu','pKmumu_piTok_kTop',
             'pKmumu_piTop']

#%% P_correct
"""
Add Probability of correct reconstruction
"""
for data in Datalist:
    P_correct = data['mu_plus_MC15TuneV1_ProbNNmu'] *\
                data['mu_minus_MC15TuneV1_ProbNNmu'] *\
                data['K_MC15TuneV1_ProbNNk'] *\
                data['Pi_MC15TuneV1_ProbNNpi']
    data['P_correct'] = P_correct