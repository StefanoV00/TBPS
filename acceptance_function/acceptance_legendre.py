#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:12:32 2022

@author: shiweiyuan
"""

"""
This script calculates the acceptance function for a given q^2 bin.
To use, type:
    from acceptance_function import acceptance
which will return a list of normalised functions, one for each bin
"""
from dbm import ndbm
import numpy as np
from data_loader import *
import scipy as sp
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from scipy.optimize import curve_fit

from numpy.linalg import LinAlgError
from find_coeff_numpy import find_coeff
from find_coeff_numpy import legendre_eval

# This function will be provided by the filtering group
# for now, leave this as a placeholder
filter_data = lambda x: x

# Polynomial degree to which the angular distribution is fitted.
# Check manually if it looks reasonable by plotting it. 
# Also check if it looks more like another functon like a sinusoidal
# and if so contact me so I can change it
poly_degree = 4

# Range of dctk to invetigate in each j iteration
N_dctk = 10
ctk_bins = np.linspace(-1, 1, N_dctk+1)

def chi_err(amps, centres, f):
    chi = np.sum((amps - f(centres))**2 / f(centres))
    dof = amps.shape[0] - poly_degree - 1
    p_value = chi2.sf(chi, dof)
    return p_value

def single_fit(data_arr, f=None):
    # Find the frequency density at each cos(theta_l)
    amps, edges = np.histogram(data_arr, bins = 25)#, density = True)
    centres = (edges[: -1] + edges[1: ]) / 2

    # Low values of bin height misestimate errors, so don't use them for fitting
    indx = np.where(amps >= 50, True, False)
    amps = amps[indx]
    centres = centres[indx]

    if f is not None:
        p_value = chi_err(amps, centres, f)
    else:
        p_value = None
    
    # Fit the amplitudes to a polynomial
    # The weight is chosen assuming a poisson distribution for the
    # frequency of each angle
    try:
        p_opt = np.polyfit(centres, amps, poly_degree) #w = 1 / np.sqrt(amps))
    except LinAlgError:
        p_opt = np.zeros(poly_degree, dtype=float)

    return p_opt, p_value

# Implemented by Andres Perez Fadon, verified by Zhenghe Xuan
def acceptance(plot=False):
    # Load the data
    acceptance_data = load_acceptance_dataset(data_dir='./data')
    
    # Apply the filering 
    filtered_data = filter_data(acceptance_data)
    
    # Initialise a list of functions and coefficients
    run = []
    acc_funcs = []
    acc_coefs = []
    p_values = []
    
    # Iterate through each bin to calculate its acceptance function
    for i in range(len(q2_ranges)):
        # Extract ith bin - Implemented by Zhenghe Xuan
        current_bin = extract_bin_number(filtered_data, i)

        if plot:
            plt.figure(figsize=(16, 6))
            plt.suptitle(f'{i}th bin')
        
        #print(current_bin)
        # make data array of only ctk, ctl, phi for find_coeff
        n = len(current_bin['phi'])
        data_lkphi = np.zeros((n, 3))
        data_lkphi[:,0] = current_bin["costhetak"]
        data_lkphi[:,1] = current_bin["costhetal"]
        data_lkphi[:,2] = current_bin["phi"]
        #print(data_lkphi.shape)
        
        C = find_coeff(data_lkphi, order = 4)
        
        phi_ex = np.zeros(50)
        
        for i in range(10):
            ctk = np.arange(-0.9, 1, 0.2)[i]
            
            if plot:
                ctk_ex = np.zeros(50) + ctk
                
                amps, edges = np.histogram(current_bin["costhetal"], bins = 25)#, density = True)
                centres = (edges[: -1] + edges[1: ]) / 2
                
                
                #### NOT SURE HOW TO SCALE FIT
                #scale = np.mean(amps)*0.05
                
                fitcurve = legendre_eval(C, ctk_ex, np.linspace(-1, 1, 50), phi_ex)
                
                centrebin = 12
                scale = amps[centrebin]/fitcurve[centrebin] #this is very very dodgy
                
                
                # Plot - to check polynomial degree
                print(f'plot ctk = {ctk}')
                plt.subplot(2, 5, i+1)
                plt.plot(centres, amps)
                plt.plot(np.linspace(-1, 1, 50), scale*fitcurve)
    
                plt.title(f'ctk = {ctk}')
                plt.title('test')
                plt.legend(["Data", "Fit"])
                plt.grid()
                # plt.ylim([0, 1])
        
    if plot:
        plt.tight_layout()
        plt.show()
            
        """
        #overall_p_opt, _ = single_fit(current_bin["costhetal"])
        #overall_poly_func = np.poly1d(overall_p_opt)
        #legendre_coeffs = find_coeff()

        def overall_to_fit(x, A):
            #return A * overall_poly_func(x)
            return(A * legendre_func(x))
            
        # Only use data for which a <= ctk < a + dctk
        for j in range(N_dctk):
            print(f'{j}th iteration')

            # Find indices for which a <= ctk < a + dctk
            a_min = ctk_bins[j]
            a_max = ctk_bins[j+1]
            indx = np.where(np.logical_and((a_min <= current_bin["costhetak"]),
                (current_bin["costhetak"] < a_max)), True, False)

            # This new array will have the ctl distribution but only for a
            # specific range of ctk values
            new_arr = current_bin["costhetal"][indx]

            # Find the frequency density at each cos(theta_l)
            amps, edges = np.histogram(new_arr, bins = 25)#, density = True)
            centres = (edges[: -1] + edges[1: ]) / 2

            # Low values of bin height misestimate errors, so don't use them for fitting
            indx = np.where(amps >= 50, True, False)
            amps = amps[indx]
            centres = centres[indx]

            sf_opt, _ = curve_fit(overall_to_fit, centres, amps, p0 = [1.])
            p_value = chi_err(amps, centres, lambda x: overall_to_fit(x, sf_opt))
            p_values.append(p_value)

            # p_opt, p_value = single_fit(new_arr, overall_poly_func)
            s_cur = f"q^2 bin: {i}\tctk bin: {j}\tchi^2 p-value: {p_value}"
            print(s_cur)
            run.append(s_cur)
            
            # Find the fit function
            poly_func = lambda x: overall_to_fit(x, sf_opt)#np.poly1d(p_opt)
            
            if plot:
                amps, edges = np.histogram(current_bin["costhetal"], bins = 25)#, density = True)
                centres = (edges[: -1] + edges[1: ]) / 2

                # Plot - to check polynomial degree
                print(f'plot {j}th plot')
                plt.subplot(2, 5, j+1)
                plt.plot(centres, amps)
                plt.plot(np.linspace(-1, 1, 50), poly_func(np.linspace(-1, 1, 50)))

                plt.title(f'ctk: {a_min:.1f} to {a_max:.1f}, {len(new_arr)} points')
                plt.legend(["Data", "Fit"])
                plt.grid()
                # plt.ylim([0, 1])
            
            # We are going to be dividing by the fit
            acc_func = lambda x: 1/poly_func(x)
            
            # Normalise - we want the average value we multiply by to be 1
            mean = spi.quad(acc_func, -1, 1)[0] / 2
            norm_func = lambda x: acc_func(x) / mean
            
            # Append function and coefficients to list
            acc_funcs.append(norm_func)
            acc_coefs.append(sf_opt) """
        
        
    return run, acc_funcs, acc_coefs, p_values

if __name__ == "__main__":
    run, acc_funcs, acc_coefs, p_values = acceptance(plot=True)
    for a, b in zip(run, p_values):
        print(a)

    # save acceptance coefficients into a pickle file 
    # import pickle
    # fn = './acceptance function/output/acceptance_function_coefs.pkl'
    # with open(fn, 'wb') as f:
    #     pickle.dump(acc_coefs, f)