#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 18:22:51 2022

@author: shiweiyuan
"""

"""
This script calculates the acceptance function for a given q^2 bin.
To use, type:
    from acceptance_function import acceptance
which will return a list of normalised functions, one for each bin
"""
import numpy as np
from data_loader import *
import scipy as sp
import scipy.integrate as spi
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2

# This function will be provided by the filtering group
# for now, leave this as a placeholder
filter_data = lambda x: x

# Polynomial degree to which the angular distribution is fitted.
# Check manually if it looks reasonable by plotting it. 
# Also check if it looks more like another functon like a sinusoidal
# and if so contact me so I can change it
poly_degree = 4

# Implemented by Andres Perez Fadon, verified by Zhenghe Xuan
def acceptance(plot=False):
    # Load the data
    acceptance_data = load_acceptance_dataset(data_dir='./data')
    
    # Apply the filering 
    filtered_data = filter_data(acceptance_data)
    
    # Initialise a list of functions and coefficients
    acc_funcs = []
    acc_coefs = []
    MSE_bins = []
    p_vals = []

    if plot:
        plt.figure(figsize=(16, 6))
    
    # Iterate through each bin to calculate its acceptance function
    for i in range(len(q2_ranges)):
        # Extract ith bin - Implemented by Zhenghe Xuan
        current_bin = extract_bin_number(filtered_data, i)
        
        # Find the frequency density at each cos(theta_l)
        #amps, edges = np.histogram(current_bin["costhetal"], bins = 25, density = True)
        amps, edges = np.histogram(current_bin["costhetal"], bins = 25, density = False)
        centres = (edges[: -1] + edges[1: ]) / 2
        #print("No. points in bin %i" % i, len(centres))
        indx = np.where(amps >= 500, True, False)
        amps = amps[indx]
        centres = centres[indx]
        # Fit the amplitudes to a polynomial
        # The weight is chosen assuming a poisson distribution for the
        # frequency of each angle
        p_opt = np.polyfit(centres, amps, poly_degree, w = 1 / np.sqrt(amps))
        
        # Find the fit function
        poly_func = np.poly1d(p_opt)
        
        # Calculate the Mean Squared Error of the fit in bin i NEEDS CHECKING
        diffs = amps - poly_func(centres)
        MSE = np.sqrt(sum(diffs**2)/(len(amps) - 2))
        #print("MSE bin %i = %.5g" % (i, MSE))
        MSE_bins.append(MSE)
        
        if plot:
            # Plot - to check polynomial degree
            plt.subplot(2, 5, i+1)
            plt.plot(centres, amps)
            plt.plot(np.linspace(-1, 1, 50), poly_func(np.linspace(-1, 1, 50)))

            plt.title(f'Bin #: {i}')
            plt.legend(["Data", "Fit"])
            plt.grid()
            if i == 0 or i == 1:
                plt.ylim(bottom=0)
            #plt.ylim([0, 1])
        
        # We are going to be dividing by the fit
        acc_func = lambda x: 1/poly_func(x)

        # Find the p-value of the fit - implemented by Andres Perez Fadon
        # if i == 0 or i == 1:
        #     print("Amps: ", amps)
        #     bound = 5
        #     chi = np.sum((amps[bound:-bound] - poly_func(centres[bound:-bound]))**2 / poly_func(centres[bound:-bound]))
        #     #print("no. of events in bin %i : " % i, len(current_bin["costhetal"]))
        #     #print("chi2 value of bin %i : " % i, chi)
        #     #print(amps)
        #     #print(poly_func(centres))
        #     dof = amps.shape[0] - poly_degree - 1 - 2 * bound
        #     #print("DOF in bin %i: " % i, dof)
        #     p_value = chi2.sf(chi, dof)
        #     #print("p-value bin %i" % i, p_value)
        
        # else:
        #     chi = np.sum((amps - poly_func(centres))**2 / poly_func(centres))
        #     #print("no. of events in bin %i : " % i, len(current_bin["costhetal"]))
        #     #print("chi2 value of bin %i : " % i, chi)
        #     #print(amps)
        #     #print(poly_func(centres))
        #     dof = amps.shape[0] - poly_degree - 1
        #     #print("DOF in bin %i: " % i, dof)
        #     p_value = chi2.sf(chi, dof)
        #     #print("p-value bin %i" % i, p_value)
        
        # indx = np.where(amps >= 500, True, False)
        # temp_amps = amps[indx]
        # temp_centres = centres[indx]
        chi = np.sum((amps - poly_func(centres))**2 / poly_func(centres))
        #print("no. of events in bin %i : " % i, len(current_bin["costhetal"]))
        #print("chi2 value of bin %i : " % i, chi)
        #print(amps)
        #print(poly_func(centres))
        dof = amps.shape[0] - poly_degree - 1
        #print("DOF in bin %i: " % i, dof)
        p_value = chi2.sf(chi, dof)
        #print("p-value bin %i" % i, p_value)
        
        # Normalise - we want the average value we multiply by to be 1
        mean = spi.quad(acc_func, -1, 1)[0] / 2
        norm_func = lambda x: acc_func(x) / mean
        
        # Append function and coefficients to list
        acc_funcs.append(norm_func)
        acc_coefs.append(p_opt)
        p_vals.append(p_value)

    if plot:
        plt.tight_layout()
        plt.show()
        
    return acc_funcs, acc_coefs, MSE_bins, p_vals

if __name__ == "__main__":
    acc_funcs, acc_coefs, MSEs, p_vals = acceptance(plot=True)
    print(acc_coefs)

    # save acceptance coefficients into a pickle file 
    #import pickle
    #fn = './acceptance function/output/acceptance_function_coefs.pkl'
    #with open(fn, 'wb') as f:
    #    pickle.dump(acc_coefs, f)