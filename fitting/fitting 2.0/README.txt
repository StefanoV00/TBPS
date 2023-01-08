TBPS_fit.py is the main file, where fitting takes place using functions from TBPS_pdf_functions and TBPS_fit_functions.py. 

TBPS_pdf_functions defines defines the PDFs and the PDFs modulated by an acceptance function.

TBPS_functions.py:
- first imports data and acceptance functions (which are hardcoded in many of the minimisation functions
due to the fact Minuit doesn't handle extra DataFrames-like arguments)
- define the Likelihoods, and Minimisation function

TBPS_bin_functions.py contains the functions used for binning in TBPS_fit_functions.py.

Then, there are plots.
