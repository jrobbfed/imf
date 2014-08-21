import sys
import numpy as np
"""
Functions for generating a synthetic pulsar population. 
Adapted from PsrPopPy (Bates et al. 2014)
"""
def lum(n, lumdisttype='lognorm', lumdistpars=[-1.1, 0.9]):
    if lumdisttype == 'lognorm':
        lum = 10. ** np.random.normal(lumdistpars[0], lumdistpars[1], n)

    return lum

def period(n, pdisttype='lor12', pdistpars=[]):
    """
    Assign periods to n pulsars. The included period
    distributions are as follows:

    lor12: Discrete empirical distribution defined by Lorimer et al. 2012, as
    described in PsrPopPy populate.py (Bates et al. 2014), for millisecond
    pulsars.
    """
    if pdisttype == 'lor12':
        # In ms
        dist = [1.,3.,5.,16.,9.,5.,5.,3.,2.]
        logpmin = 0.
        logpmax = 1.5
        distcdf = np.cumsum(dist) / np.sum(dist)
        bin_num = np.digitize(np.random.random_sample(n), distcdf)
        
        #Assume a linear distribution within each bin.
        logp = logpmin + (logpmax - logpmin) * (bin_num +
                np.random.random_sample(n)) / len(dist)
        p = 10. ** logp 
    
    return p


def 
