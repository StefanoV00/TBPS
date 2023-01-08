# Acceptance function sub-group code

## Information for fit group 

We fit the filtered acceptance dataset along the different q2 bins as 
defined by the problem statement (code in acceptance_function.py). 
The resulting polynomial coefficients are stored as a pickle file in 
the output folder. 

The pickle file can be loaded with `pickle.load`. It contains a list 
where the index is the q2 bin number and the contents is a numpy array 
of polynomial coefficients for the fitted function. For now, can use 
`np.poly1d` to generate the polynomial function from the coefficients. 