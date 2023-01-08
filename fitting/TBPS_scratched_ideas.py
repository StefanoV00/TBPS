# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:14:51 2022

@author: Stefano
"""

def ctk_PDF2(costhetak, costhetal, fl):
    """
    Parameters
    ----------
    costhetak : ndarray
        Tuple of experimentally observed cos(theta_k)
    costhetal : ndarray or int
        Experimentally observed cos(theta_l) corresponding to the cos(theta_k)
    fl : float
        The factor F_L in the distribution.

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_k
    """
    ctk = costhetak; ctl = costhetal
    P = 3/2 * fl * ctk**2 + 3/4 *(1-fl)*(1-ctl**2) 
    return P


def ctk_acc_PDF2(costhetak, costhetal, fl, acc_poly):
    """
    Parameters
    ----------
    costhetak : ndarray
        Tuple of experimentally observed cos(theta_k), for a bin of q2.
    costhetal : ndarray or int
        Experimentally observed cos(theta_l) corresponding to the cos(theta_k),
        for a bin of q2.
    fl : float
        The factor F_L in the distribution, for a bin of q2.
    acc_poly: numpy 4th order polynomial
        The acceptance function for costhetak, for a bin of q2.

    Returns
    -------
    normalised_P : ndarray
        Probability Distribution for observing costheta_k
    """
    return ctk_PDF(costhetak, costhetal, fl) * acc_poly(costhetak, costhetal)

def nLL_ctk2 (fl, afb, _bin):
    """
    Parameters
    ----------
    fl : float
        The factor F_L in the distribution.
    afb : float
        The factor A_fb in the distribution. It doesn't contribute, at all, to
        ctk_PDF, but is needed as an argument to let nLL_ctk and other nLLs
        have same number of arguments.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed cos(theta_k) distribution
    """
    _bin = int(_bin)
    ctl = data[_bin]['costhetal']
    ctk = data[_bin]['costhetak']
    normalised_P = ctk_PDF2(costhetak = ctk, costhetal = ctl, fl=fl)
    if min(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL
    
    
    
    
#---------------------------------------------------------------------------- 
############################################################################
#% Fitting Alternatives, in the case the one above shouldn't work
############################################################################
#----------------------------------------------------------------------------

def fit_ctl (guess, bin_number_to_check = 5):
    """
    Minimise the NLL associated to the cos(theta_l) distribution using
    Minuit class from iminuit, hence enjoying of its functionalities,
    with its scipy function, allowing to set constraints.
    
    Parameters
    ----------
    guess : list or tuple
        Initial guess for Fl and Afb. A different pair of guesses can be
        provided for each bin, then guess is a ndarray of shape (N, 2), where
        N is the number of bins.
    bin_number_to_check : int, optional
        If an int between 0 and len(data), then return bin_results_to_check
        for further analysis. The default is 5.

    Returns
    -------
    estimates: tuple of 2 lists of floats:
        One list of estimates of F_L for each bin.
        One list of estimates of A_fb for each bin.
    errors: tuple of 2 lists of floats:
        One list of errors of F_L for each bin.
        One list of errors of A_fb for each bin.
    bin_results_to_check : minuit estimate
        Return iff bin_number_to_check is an int between 0 and len(data) for 
        further anlysis.
    """
    nLL_ctl.errordef = Minuit.LIKELIHOOD
    decimal_places = 3
    fls, fl_errs = [], []
    afbs, afb_errs = [], []
    
    #Define constraining function: positive PDFs
    def PDF_con (fl, afb, i):
        i = int(i)
        ndarray_ctl = ctl_PDF( fl, afb, data[i]['costhetal'] )
        ndarray_ctk = ctk_PDF( fl, data[i]['costhetak'] )
        return min([min(ndarray_ctl), min(ndarray_ctk)]) 
    
    bin_results_to_check = 0             
    if len(guess) == len(data):
        for i in tqdm( range (len(data)), "Bins" ):
           
            m = Minuit(nLL_ctl, 
                        fl = guess[i][0], 
                        afb = guess[i][1], 
                        _bin = i)
            
            # First run a simplex minimisation
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            m.simplex()
            fl = m.values[0]
            afb = m.values[1]
            
            #Then a scipy-constrained minimisation from simplex's solution
            m = Minuit(nLL_ctl, fl = fl, afb = afb, _bin = i)
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            con = NonlinearConstraint(PDF_con, lb = 0, ub = np.inf)
            m.scipy(constraints = con)
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
            else:
                #Apply only positiveness constraint
                print("Only positiveness constraint applied, due to failure")
                print("in minimisation with all constraints.")
                m = Minuit(nLL_ctl, fl = fl, afb = afb, _bin = i)
                m.fixed['_bin'] = True
                m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
                con = NonlinearConstraint(PDF_con, lb = 0, ub = np.inf)
                m.scipy(constraints = con[0])
                m.hesse()
            
            Finally a migrad minimisation starting from scipy's solution
            mainly to obtain errors on estimates
            m = Minuit(nLL_ctl, fl = fl, afb = afb, _bin = i)
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            m.migrad()
            m.hesse()
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
            
            if i == bin_number_to_check:
                bin_results_to_check = m
                
            fls.append(fl);      afbs.append(afb)
            fl_errs.append(m.errors[0]);  afb_errs.append(m.errors[1])
    else:
        for i in tqdm( range (len(data)), "Bins" ):

            m = Minuit(nLL_ctl, fl = guess[0], afb = guess[1], _bin = i)
            
            # First run a simplex minimisation
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            m.simplex()
            fl = m.values[0]
            afb = m.values[1]
            
            #Then a scipy-constrained minimisation from simplex's solution
            m = Minuit(nLL_ctl, fl = fl, afb = afb, _bin = i)
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            con = NonlinearConstraint(PDF_con, lb = 0, ub = np.inf)
            m.scipy(constraints = con)
            m.hesse()
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
            else:
                #Apply only positiveness constraint
                print("Only positiveness constraint applied, due to failure")
                print("in minimisation with all constraints.")
                m = Minuit(nLL_ctl, fl = fl, afb = afb, _bin = i)
                m.fixed['_bin'] = True
                m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
                con = NonlinearConstraint(PDF_con, lb = 0, ub = np.inf)
                m.scipy(constraints = con[0])
                m.hesse()
            
            Finally a migrad minimisation starting from scipy's solution
            mainly to obtain errors on estimates
            m = Minuit(nLL_ctl, fl = fl, afb = afb, _bin = i)
            m.fixed['_bin'] = True
            m.limits=( (-1.0, 1.0), (-1.0, 1.0), None)
            m.migrad()
            m.hesse()
            if not ( np.isnan(m.values[0]) and np.isnan(m.values[1]) ):
                fl = m.values[0]
                afb = m.values[1]
            
            if i == bin_number_to_check:
                bin_results_to_check = m
                
            fls.append(fl);      afbs.append(afb)
            fl_errs.append(m.errors[0]);  afb_errs.append(m.errors[1])
    if bin_results_to_check:
        return (fls, afbs), (fl_errs, afb_errs), bin_results_to_check  
    else:
        return (fls, afbs), (fl_errs, afb_errs)

def nLL_ctl_sci (x, _bin):
    """
    Parameters
    ----------
    x: list
        It is [fl, afb], where:
        - fl is the factor F_L in the distribution.
        - afb is the factor A_FB in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihood of the observed cos(theta_l) distribution
    """
    fl, afb = x
    _bin = int(_bin)
    ctl = data[_bin]['costhetal']
    normalised_P = ctl_PDF(fl=fl, afb=afb, costhetal = ctl)
    if min(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL

def nLL_phi_sci (x, _bin):
    """
    Parameters
    ----------
    x : 1D list, with 3 entries:
        The factor F_L in the distribution.
        The factor A_t in the distribution.
        The factor A_I_m in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed phi distribution
    """
    fl, at, aim = x
    _bin = int(_bin)
    phi = data[_bin]['phi']
    normalised_P = phi_PDF(fl, at, aim, phi = phi)
    if min(normalised_P) <= 0:
        return np.float("nan")
    else:
        nLL = - np.sum(np.log(normalised_P))
        return nLL

def nLL_cts_sci (x, _bin):
    """
    Parameters
    ----------
    x : 1D list, with 3 entries:
        The factor F_L in the distribution.
        The factor A_fb in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed distributions of costheta_l
        and costheta_k.
    """
    fl, afb = x
    return nLL_ctl(fl, afb, _bin) + nLL_ctk(fl, afb, _bin)


def nLL_ang_sci (x, _bin):
    """
    Parameters
    ----------
    x : 1D list, with 3 entries:
        The factor F_L in the distribution.
        The factor A_fb in the distribution.
        The factor A_t in the distribution.
        The factor A_I_m in the distribution.
    _bin : int
        We analyse the bin number binth in the data (data[binth]).

    Returns
    -------
    nLL : ndarray
        Negative Log Likelihhod of the observed angular distribution.
    """
    fl, afb, at, aIm = x
    return nLL_ctl(fl, afb, _bin)+nLL_ctk(fl, afb, _bin)+nLL_phi(fl, at, aIm, _bin)

def fit_ctl_scipy (guess):
    """
    Minimise the NLL associated to the cos(theta_l) distribution using
    minimize from SciPy.optimize, with constraints on positive PDF.
    
    Parameters
    ----------
    guess : list or tuple
        Initial guess for Fl and Afb. A different pair of guesses can be
        provided for each bin, then guess is a ndarray of shape (N, 2), where
        N is the number of bins.

    Returns
    -------
    fls : list of floats
        One estimate of Fl for each bin.
    afbs : list of floats
        One estimate of Fl for each bin.
    """
    fls = []
    afbs = []
    
    def PDF_con (x, i):
        ndarray = ctl_PDF( x[0], x[1], data[i]['costhetal'] )
        return min(ndarray)

    if len(guess) == len(data): #one set of guesses per bin
        for i in tqdm( range (len(data)), "Bins" ):
            
            con = {'type': 'ineq', 'fun': PDF_con, "args": (i)}
            
            res = minimize(nLL_ctl_sci, guess[i], args = (i), 
                           bounds = ((-1, 1), (-1, 1),),
                           constraints = con)
            
            if not res.success:
                res = minimize(nLL_ctl_sci, guess[i], args = (i), 
                           bounds = ((-1, 1), (-1, 1),), 
                           constraints = con, 
                           method = "cobyla" )
            elif not res.success:
                res = minimize(nLL_ctl_sci, guess[i], args = (i), 
                           bounds = ((-1, 1), (-1, 1),), 
                           constraints = con, 
                           method = "SLSQP" )
            elif not res.success:
                res = minimize(nLL_ctl_sci, guess[i], args = (i), 
                           bounds = ((-1, 1), (-1, 1),), 
                           constraints = con, 
                           method = "trust-constr" )
            fls.append(res.x[0])
            afbs.append(res.x[1])
            
    else:
        for i in tqdm( range (len(data)), "Bins" ):
            
            con = {'type': 'ineq', 'fun': PDF_con, "args": [i]}
            
            res = minimize(nLL_ctl_sci, guess, args = (i), 
                            bounds = ((-1, 1), (-1, 1),),
                            constraints = con)
            if not res.success:
                res = minimize(nLL_ctl_sci, guess, args = (i), 
                           bounds = ((-1, 1), (-1, 1),), 
                           constraints = con, 
                           method = "cobyla" )
            elif not res.success:
                res = minimize(nLL_ctl_sci, guess, args = (i), 
                           bounds = ((-1, 1), (-1, 1),), 
                           constraints = con, 
                           method = "SLSQP" )
            elif not res.success:
                res = minimize(nLL_ctl_sci, guess, args = (i), 
                           bounds = ((-1, 1), (-1, 1),), 
                           constraints = con, 
                           method = "trust-constr" )
            fls.append(res.x[0])
            afbs.append(res.x[1])
    
    return fls, afbs


def fit_ctl_hopping (guess, niter = 100, step = 0.1):
    """
    Minimise the NLL associated to the cos(theta_l) distribution using
    basinhopping from SciPy.optimize, with constraints on positive PDF.
    
    Parameters
    ----------
    guess : list or tuple
        Initial guess for Fl and Afb. 

    Returns
    -------
    fls : list of floats
        One estimate of Fl for each bin.
    afbs : list of floats
        One estimate of Fl for each bin.
    """
    fls = []
    afbs = []
    
    def PDF_con (x, i):
        ndarray = ctl_PDF( x[0], x[1], data[i]['costhetal'] )
        return min(ndarray)
    
    for i in tqdm( range (len(data)), "Bins" ):
        
        con = {'type': 'ineq', 'fun': PDF_con, "args": (i,)}
        minimizer_kwargs = {"args": (i), "bounds": ((-1, 1), (-1, 1),),
                            "constraints": con}
        
        res = basinhopping(nLL_ctl_sci, guess, 
                           minimizer_kwargs = minimizer_kwargs,
                           niter = niter, stepsize = step, disp = True)
        
        fls.append(res.x[0])
        afbs.append(res.x[1])
    
    return fls, afbs


def fit_ctl_mymetro (iters = 5e3, kT0 = 100, anneal = 0.5, scan = 1e3, 
                step = 0.2, close_factor = 2, halving = 0.5,
                params = float("nan") ):
    """
    Parameters
    ----------
    See metrpolis function.

    Returns
    -------
    fls : list of floats
        One estimate of Fl for each bin.
    afbs : list of floats
        One estimate of Fl for each bin.
    """

    fls = []
    afbs = []
    
    interval = [[-1, -1], [1, 1]] #min,min -> max,max
    for i in tqdm( range (len(data)), "Bins" ):
        (fl, afb), E = metropolis(nLL_ctl_sci, interval, iters = iters, 
                                    kT0 = kT0, anneal = anneal, scan = scan, 
                                    step = step, close_factor = close_factor,
                                    halving = halving, params = i)
        
        fls.append(fl); afbs.append(afb)
    return fls, afbs
#Stefano's Metropolis Algorithm, 
# could be useful to have a Non-blackbox stochastic algorithm
def metropolis (f, interval, iters = 5e3, kT0 = 100, anneal = 0.5, scan = 1e3, 
                step = 0.2, close_factor = 2, track = False, halving = 0.5,
                params = float("nan")):
    """
    Perform a Montecarlo Metropolis (Annealing) minimisation of N-dimensional 
    function f. The probability of jumping around even if not minimum is based 
    on P = exp(-deltaf/kT). \n

    Parameters
    ----------
    f : callable
        Function to be minimised.
    interval : ndarray (2, N)
        Interval defines the scanned region [[xmin, ymin,...],[xmax, ymax...]].
    iters : int, optional
        Number of iterations. 
        The default is 5e3.
    kT0 : float, optional
        Initial temperature: the bigger kT, the more probable to "jump" 
        around. 
        The default is 100.
    anneal : float in (0, 1], optional
        If anneal is given, lT will then be reduced at every iteration up to,
        at the end, kT*(1-anneal) amount. 
        The default is 0.5.
    scan : int, optional
        Scan defines the number of evaluation at the very beginning, to get a 
        fast image of the function's landscape.
        The default is 1e3.
    step : float, optional
        Each step is taken randomly with gaussian pdf, centred at latest point,
        with sigma being, at the most, step * size of interval (0 < step <=1). 
        Note, the point will be rejected if outside the interval, but another 
        will be taken, in that same direction, with uniform distribution.
        The default is 0.2.
    close_factor : float, optional
        Close_factor modifies sigma accoridng to how close we are to the 
        one we think, at the moment, is the minimum, such that: 
            
     sigma = step*size*close_factor**(abs(position_now - minimum)/size - 1) 
      
         hence sigma goes from step*size/close_factor to step*size.
         The default is 2.
    track : bool, optional
        If track, also return a list of all positions the algorithm visited.
        The default is False.
    halving : float, optional
        If halving not 0, after halving*iters iterations (0 <= halving <= 1), 
        go to the one we think is the global minimum, half the interval around 
        it, repeat procedure, recharging the temperature.
        The default is 0.5.
    params : whatever, optional
        IMPORTANT: If function takes one extra numerical parameter, you have to 
        insert it even if the default value is wanted, otherwise returns NaN, 
        sorry. The default is float("nan").

    Returns
    -------
    umin: ndarray
        Coordinates of minimum.
    Emin: float
        Value of f at minimum.
    utrack: ndarray, only if track is True.
        List of coordinates "visited" by algorithm in non-scanning mode.
    """
    
    interval = np.array(interval, float)
    size = abs(interval[1] - interval[0])
    step = step * size
    
    #starting point
    i = 0
    while i < scan:
        i += 1
        uvec = np.random.uniform(interval[0], interval[1])
        try:    E = f(uvec, params)
        except: E = f(uvec)
        if i == 1:
            Emin = E * 1.0
            umin = np.array(uvec, float)
        else:
            if np.isnan(E) or np.isnan(Emin):
                continue
            elif E <= Emin:
                Emin = E * 1.0
                umin = np.array(uvec, float)
            else:
                continue
            
    reduc= anneal / iters # to anneal kT, see end of for loop
    uveclist = [umin]     # start keeping track
    
    if halving:
        times = int(1 // halving)
        sets = [(iters*halving)]*times + [iters - int(iters*halving)*times]
        sets = [int(setj) for setj in sets]
    else:
        sets = [iters]
        
    for setj in sets:
        uvec0 = np.array(umin, float) 
        E0 = Emin * 1.                
        kT   = kT0 * 1.
        for i in range(setj):
            # find new point
            sigma = step * close_factor ** (abs(uvec0 - umin)/size - 1)
            uvec  = np.random.normal(uvec0, sigma)
            
            # put new point's coordinates in the interval if outside
            for j in range(len(uvec)):
                if uvec[j] <= interval[0][j]:
                    uvec[j] = np.random.uniform(interval[0][j], uvec0[j])
                elif uvec[j] >= interval[1][j]:
                    uvec[j] = np.random.uniform(uvec0[j], interval[1][j])
            
            try:    E = f(uvec, params)
            except: E = f(uvec)
            deltaE = E - E0
            
            if np.isnan(deltaE):
                continue #without annealing, as it represents a miss
            
            if deltaE <= 0: #we found a smaller energy: change point
                uvec0 = np.array(uvec, float)
                E0    = E * 1.
                uveclist.append(uvec0)
                if E < Emin: 
                    # set new total minimum
                    umin = np.array(uvec, float)
                    Emin = E * 1.
                if anneal:
                    kT -= reduc * kT0
            else:
                # smaybe change starting point for next iteration
                if np.exp(- deltaE / kT) >= np.random.uniform(0, 1):
                    uvec0 = np.array(uvec, float)
                    E0    = E * 1.
                    uveclist.append(uvec0)
                else:
                    pass
                if anneal:
                    kT -= reduc * kT0
            
            
        #Do the halving
        step /= 2
        delta = umin - interval
        interval = umin - delta/2
        
    if track:
        return umin, Emin, np.array(uveclist)
    
    else:
        return umin, Emin
    
    
#%% MINIMISE WITH SCIPY

# guess = [-0.5, 0.0]
# fls, afbs = fit_ctl_scipy(guess)

# rc = len(data)
# c = int(np.sqrt(rc))
# r = int( np.ceil( rc / c) )
# figsize = (c * 4, r * 2.2)
# # for i in range(len(data)):
# #     
# #     #fig.suptitle("Subtitle", size = "15")
# fig = plt.figure(figsize = figsize, tight_layout = True)
# for i in range(len(data)):
#     plt.subplot(r, c, i + 1)
#     Nbins = 25
#     ctli = data[i]['costhetal']
#     hist, _bins, _ = plt.hist(ctli, bins = Nbins, density = True)
#     x = np.linspace(-1, 1, Nbins)
#     #pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
#     y = costhetal_PDF(fl=ftls[i], afb=afbs[i], costhetal = x) #* pdf_multiplier
#     plt.plot(x, y, label=f'Bin {i}: {mybins[i]}, \n\
# Fl={round(fls[i], 3)},Afb={round(afbs[i],3)}')
#     plt.xlabel(r'$cos(\theta_l)$')
#     plt.ylabel(r'PDF')
#     plt.legend()
#     plt.grid()
    
#     fls.append(ress[i].x[0])
#     afbs.append(ress[i].x[1])
# plt.savefig(f'MinPlots/Ctl_Scipy')
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
# ns = np.linspace(0, len(data) - 1, len(data))
# ax1.errorbar(ns, fls, fmt = ".", capsize = 4, label=r'$F_L$')
# ax2.errorbar(ns, afbs, fmt = ".", capsize = 4, label=r'$A_{FB}$')
# ax1.grid()
# ax2.grid()
# ax1.set_ylabel(r'$F_L$')
# ax2.set_ylabel(r'$A_{FB}$')
# ax1.set_xlabel(r'Bin number')
# ax2.set_xlabel(r'Bin number')
# plt.tight_layout()
# plt.show()