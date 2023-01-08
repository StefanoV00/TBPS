###############################################################################################à
MAIN RESULTS
_________________________________________________________________________________________________
Main results are saved as

               All0_Bin{i}_res_mod0.pkl (or csv)

which is a pickle containing a Dataframe for results of bin numbr i.
The 0s stand for:
- dataset c (0 = classified, other = calssififed with random cuts for uncertainty)
- acceptance modification (0 = no modification, other = modified for uncertainty)



###############################################################################################
PROJECTION RESULTS
Numpy array with [values, uncertainties], where values and uncertainties have 10 elements, 
one per bin:
                proj0_{var}_scipy[0, 2]_mod0.npy

and scipy [0, 2] indicate functons of scipy labelled 0 (brute) and 2 (difefrential_evolution)
were also used.


###############################################################################################
OTEHR RESULTS
Numpy arrays [fls, afbs, s3, s4, s5, s7, s8] with each being a ndarray of 10 elements,
one per bin:
                  results{c}_mod{m}.npy

with c and m beig integers:
- dataset c (0 = classified, other = calssififed with random cuts for uncertainty)
- acceptance modification (0 = no modification, other = modified for uncertainty)
Used for systematic uncertainty.
