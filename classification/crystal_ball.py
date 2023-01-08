# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 06:12:15 2022

@author: kangyuchenterry
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#input all_mass_list is a list of that contains all invariant mass that we use to plot the distribution, for example, we can have a list of mass [1,1,2,3,4,4,7]. We need to count the repetition here.
'''
all_mass_list = []
sum_of_all_mass = 0
for item in all_mass_list:
    sum_of_all_mass += item
initial_guess_miu = sum_of_all_mass/len(all_mass_list)
'''

#mass_list_for_distribution is the list for the mass that we have a number to plot the distribution. Here we don't have repetition. For the list of mass example in input all_mass_list, we now take [1,2,3,4,7]
# mass_list_for_distribution = []

#number is the list of numbers for distribution corresponding to the mass_list_for_distribution
# number = []


def exponential_background(m,coe,tau):
    return coe*np.exp(-tau*(m-5170))
'''
def crystal_ball(m,A,mu,sigma,alpha,n,coe,tau):
  
    a = (n/abs(alpha))**n*np.exp(-0.5*alpha**2)
    b = n/abs(alpha)-abs(alpha)

    compare_to_alpha = (m-mu)/sigma
    
    ind = (compare_to_alpha > -1*abs(alpha))  
    
    return A*(np.exp(-0.5*(compare_to_alpha)**2)*(ind) - (ind-1)*(a/(b-(compare_to_alpha))**n))  #+ exponential_background(m,coe,tau))
'''

def crystal_ball(m,A,mu,sigma,alpha,n,coe,tau):
  
    a = (n/abs(alpha))**n*np.exp(-0.5*alpha**2)
    b = n/abs(alpha)-abs(alpha)

    x = (m-mu)/sigma
    c = -1*abs(alpha)
    
    return np.piecewise(x, [x > c, x <= c], 
                [lambda x: A*(np.exp(-0.5*(x)**2)) 
                    + exponential_background(x*sigma+mu,coe,tau), 
                lambda x: A*((a/(b-(x))**n)) 
                    + exponential_background(x*sigma+mu,coe,tau)])

def crystal_ball_nb(m,A,mu,sigma,alpha,n,coe,tau):
  
    a = (n/abs(alpha))**n*np.exp(-0.5*alpha**2)
    b = n/abs(alpha)-abs(alpha)

    x = (m-mu)/sigma
    c = -1*abs(alpha)
    
    return np.piecewise(x, [x > c, x <= c], [lambda x: A*(np.exp(-0.5*(x)**2)), lambda x: A*((a/(b-(x))**n))])

def crystal_ball_np(m,A,mu,sigma,coe,tau):
  
    

    x = (m-mu)/sigma
    
    
    return A*(np.exp(-0.5*(x)**2)) + exponential_background(x*sigma+mu,coe,tau)
 
def fit_signal_according_to_formula_six_point_three(m,mu,sigma1,sigma2,alpha,n,fcore):
    return fcore*crystal_ball(m, mu,sigma1, alpha, n)+(1-fcore)*crystal_ball(m,mu, sigma2, alpha, n)
    





    
# popt,pcov = curve_fit(crystal_ball,mass_list_for_distribution,number)
# popt,pcov = curve_fit(fit_signal_according_to_formula_six_point_three,mass_list_for_distribution,number)













