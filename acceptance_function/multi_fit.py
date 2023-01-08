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

from numpy.linalg import LinAlgError

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

# Implemented by Andres Perez Fadon, verified by Zhenghe Xuan
def acceptance(plot=False):
    # Load the data
    acceptance_data = load_acceptance_dataset(data_dir='./data')
    
    # Apply the filering 
    filtered_data = filter_data(acceptance_data)
    
    # Initialise a list of functions and coefficients
    acc_funcs = []
    acc_coefs = []
    
    # Iterate through each bin to calculate its acceptance function
    for i in range(len(q2_ranges)):
        # Extract ith bin - Implemented by Zhenghe Xuan
        current_bin = extract_bin_number(filtered_data, i)

        if plot:
            plt.figure(figsize=(16, 6))
            plt.suptitle(f'{i}th bin')

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
            amps, edges = np.histogram(new_arr, bins = 25, density = True)
            centres = (edges[: -1] + edges[1: ]) / 2
            
            # Fit the amplitudes to a polynomial
            # The weight is chosen assuming a poisson distribution for the
            # frequency of each angle
            try:
                p_opt = np.polyfit(centres, amps, poly_degree) #w = 1 / np.sqrt(amps))
            except LinAlgError:
                p_opt = np.zeros(poly_degree, dtype=float)
            
            # Find the fit function
            poly_func = np.poly1d(p_opt)
            
            if plot:
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
            acc_coefs.append(p_opt)

        if plot:
            plt.tight_layout()
            plt.show()
        
    return acc_funcs, acc_coefs

if __name__ == "__main__":
    acc_funcs, acc_coefs = acceptance(plot=True)
    print(acc_coefs)

    # save acceptance coefficients into a pickle file 
    # import pickle
    # fn = './acceptance function/output/acceptance_function_coefs.pkl'
    # with open(fn, 'wb') as f:
    #     pickle.dump(acc_coefs, f)
