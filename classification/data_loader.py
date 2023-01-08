# Helper functions to load classified dataset for other groups 
# To use, do `from classification.data_loader import *`

import pandas as pd

def load_classified_total(fp_classification='./'):
    # `fp_classification` is the path to the classification folder
    # this might change depending on your working directory 

    return pd.read_pickle(fp_classification + '/Classified total Dataset.bz2')

def load_classified_acceptance(fp_classification='./'):
    # `fp_classification` is the path to the classification folder
    # this might change depending on your working directory 
    
    return pd.read_pickle(fp_classification + '/Classified_acceptance_mc.bz2')







#### COPIED FROM ACCEPTANCE FOR CONVENIENCE ####


import numpy as np
import matplotlib.pyplot as plt


## CONSTANTS ##

# q2_ranges contains bins as defined in 
# https://mesmith75.github.io/ic-teach-kstmumu-public/predictions/
q2_bin_defs = '''0	0.1 - 0.98
1	1.1 - 2.5
2	2.5 - 4.0
3	4.0 - 6.0
4	6.0 - 8.0
5	15.0 - 17.0
6	17.0 - 19.0
7	11.0 - 12.5
8	1.0 - 6.0
9	15.0 - 17.9'''.split('\n')
q2_ranges = [tuple(map(float, x[2:].split(' - '))) for x in q2_bin_defs]
del q2_bin_defs


## FUNCTIONS ##


def load_acceptance_dataset(data_dir='../data'):
    acceptance_path = f'{data_dir}/acceptance_mc.pkl'
    return pd.read_pickle(acceptance_path)


def plot_ang_distr(df):
    'Takes one dataframe and plots the costhetal distribution'
    df['costhetal'].plot.hist(density=True)
    plt.xlabel(r'$\cos\theta_l$')


def calc_bin_number(q2, ranges, range_labels):
    for i, r in zip(range_labels, ranges):
        r_low, r_high = r
        if q2 > r_low and q2 < r_high:
            return i

    return np.nan # not included in the bins 


def create_column_bin_number(df):
    '''Adds two new columns to df which identifies which q2 bin 
       each row belongs to.
       
       Bin #'s 0-7 under `bin_number_1`
       Bin #'s 8-9 under `bin_number_2`'''

    q2_ranges_1 = q2_ranges[:8]
    q2_ranges_2 = q2_ranges[8:]

    df['bin_number_1'] = df['q2'].apply(calc_bin_number, 
                ranges=q2_ranges_1, range_labels=range(8)).astype('category')

    df['bin_number_2'] = df['q2'].apply(calc_bin_number, 
                ranges=q2_ranges_2, range_labels=[8, 9]).astype('category')


def extract_q2_range(df, q2_low, q2_high):
    '''Returns dataframe where column q2 satisfies q2_low < q2 < q2_high'''
    return df.query('@q2_low < q2 & q2 < @q2_high')


def extract_bin_number(df, bin_number):
    '''Extracts events which fall into bin_number from df'''
    q2_low, q2_high = q2_ranges[bin_number]
    return extract_q2_range(df, q2_low, q2_high)