'''
Helper functions to load acceptance dataset (perhaps after processing) 
and then extract data based on bin numbers or q ranges 

NOTE: bin numbers are defined here:
https://mesmith75.github.io/ic-teach-kstmumu-public/predictions/
However, these bins overlap, which is why there are two columns: 
    bin_number_1 and bin_number_2

NOTE: when working with dataframes, it is faster to discard unneeded columns
i.e. work with `df[['q2', 'costhetal']]` instead of whole `df`

To import, use: `from data_loader import load_acceptance_dataset`

Example of functions in action at the end of file. 
'''

import numpy as np
import pandas as pd
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



if __name__ == '__main__':
    # Following code splits up acceptance dataset by bin, and plots the results

    data_dir = './data' # <-- this might need to be changed depending on your working dir
    acceptance = load_acceptance_dataset(data_dir)
    acceptance = acceptance[['q2', 'costhetal']] # discard unecessary columns 

    plt.figure(figsize=(16, 5))
    plt.suptitle(r'Acceptance dataset $\cos\theta_l$ distributions')
    for i in range(len(q2_ranges)):
        plt.subplot(2, 5, i+1)
        plt.title('q2: {} - {}'.format(*q2_ranges[i]))

        # extract slice of dataset and plots it 
        df_slice = extract_bin_number(acceptance, i)
        plot_ang_distr(df_slice)

    plt.tight_layout()
    plt.show()
