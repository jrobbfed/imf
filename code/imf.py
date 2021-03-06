"""
Implement various forms of the initial mass function, and 
draw samples of stars from them to form star clusters.
*** ADAPTED FROM ADAM GINSBURG @ https://code.google.com/p/agpy/ ***
"""

import numpy as np
import types

class IMF(object):
    """
    Initial mass function class
    """
    def dndm(self, m, **kwargs):
        "Returns differential form of the function"
        return self(m, integ_form=False, **kwargs)

    def dndlogm(self, m, **kwargs):
        "Returns the dn/d(logm) form, to compare to log-binned samples"
        return m * self(m, integ_form=False, **kwargs)

    def n_of_m(self, m, **kwargs):
        "Return the integral form of the function"
        return self(m, integ_form=True, **kwargs)

    def integrate(self, m1, m2, **kwargs):
        "Integrate the function between upper and lower limits m1,m2."
        import scipy.integrate
        return scipy.integrate.quad(self, m1, m2, **kwargs)


###SUBCLASSES OF IMF####
class Salp(IMF):

    def __init__(self, alpha=-2.35, mmin=0.01, mmax=100, **kwargs):
        """
        Create a Salpeter IMF, using linear solar mass units,
        with a default slope of alpha=-2.35,
        from Salpeter (1955). dN/dm = m^alpha
        """
        self.alpha = alpha
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, m, integ_form=False):
        if integ_form:
            return m ** (self.alpha + 1)
        else:
            return m ** (self.alpha)

class Larson(IMF):
    def __init__(self, alpha=-1.35, ms=0.4, mmin=0.09, mmax=20):
        """
        Create a Larson (1998) IMF, in linear solar mass units, with a default
        slope of -1.35, scale mass of 0.4 solar masses, and lower/upper limits
        of 0.09/20 solar masses. These default parameters follow the treatment
        of Hernandez (2012).
        """
        self.alpha = alpha
        self.ms = ms
        self.mmin = mmin
        self.mmax = mmax

    def __call__(self, m, integ_form=False):
        if integ_form:
            pass
        else:
            return (1 + m / self.ms) ** (self.alpha) / m

class Chabrier(IMF):
    def __init__(self, ):
        """
        """

    def __call__(self, m, integ_form=False):
        if integ_form:
            pass
        else:
            return 

class Kroupa(IMF):
    def __init__(self, mmin=0.03, mmax=120.):
        """
        Create a Kroupa (2001) IMF, in linear solar mass units, with default
        lower/upper mass limits of 0.03/120 solar masses. The precise form of
        the IMF is defined in the __call__ parameters. 
        """
        self.mmin = mmin
        self.mmax = mmax
    def __call__(self, m, p1=0.3, p2=1.3, p3=2.3, b1=0.08, b2=0.5,
            integ_form=False):
        """
        a parameters are the three power-law slopes of the Kroupa IMF
        b parameters are the breaks in the IMF in solar masses
        """
        binv = ((b1**(-(p1-1)) - self.mmin**(-(p1-1)))/(1-p1) +
                (b2**(-(p2-1)) - b1**(-(p2-1))) * (b1**(p2-p1))/(1-p2) +
                (- b2**(-(p3-1))) * (b1**(p2-p1)) * (b2**(p3-p2))/(1-p3))
        b = 1./binv
        c = b * b1**(p2-p1)
        d = c * b2**(p3-p2)

        zeta = (b*(m**(-(p1))) * (m<b1) +
                c*(m**(-(p2))) * (m>=b1) * (m<b2) +
                d*(m**(-(p3))) * (m>=b2))

        if integ_form:
            return zeta * m
        else:
            return zeta



mf_dict = {'salpeter':Salp(), 'larson':Larson(), 'kroupa':Kroupa()}
reverse_mf_dict = {i:j for j,i in mf_dict.iteritems()}
mostcommonmass = {'salpeter':Salp().mmin, 'larson':Larson().mmin,
'kroupa':max([0.08, Kroupa().mmin])}

def get_massfunc(massfunc):
    """
    Pass through any user-defined custom mass function, or if given a string
    return the IMF class corresponding to that name
    """
    
    if type(massfunc) is types.FunctionType or hasattr(massfunc,'__call__'):
        return massfunc
    elif type(massfunc) is str:
        return mf_dict[massfunc]
    else:
        raise ValueError("massfunc must either be a string in %s or a function"
                % (",".join(mf_dict.keys())))


# class 


#Functions to sample from IMFs.
def inverse_sample(n, nbins=10000, massfunc='salpeter', **kwargs):
    """
    Invert a given imf (really the cdf of that imf) and return n masses where
    the probability that a star is below that mass is prob. I.e. 
    prob(mass) = P(mmin < m < mass), where m is the random variable.

    massfunc can be a string from mf_dict or a user-defined custom function.
    """
    mmin = get_massfunc(massfunc).mmin
    mmax = get_massfunc(massfunc).mmax
    prob = np.random.random(n)
    masses = np.logspace(np.log10(mmin), np.log10(mmax), nbins)
    #Construct normalized cdf from the integral form of the mass function
    mf_integ = get_massfunc(massfunc)(masses, integ_form=True, **kwargs)
    cdf = (mf_integ - mf_integ[0]) / (mf_integ[-1] - mf_integ[0])    
    #Return the mass corresponding to given probability, interpolated between
    #the masses grid given.
    return np.interp(prob, cdf, masses)

#Approximate acceptance fractions for mc_sample Monte Carlo sampling of IMFs.
#These fractions are experimentally determined and slightly underestimated to 
#reduce the likelihood of extra rounds of sampling in mc_sample.
#More precisely, the fractions are: salp - 0.0462
mc_frac = {'salpeter':0.04, 'larson':0.31}


def mc_sample(n, massfunc='salpeter', **kwargs):
    """
    Sample an IMF using Monte Carlo rejection sampling, with a log-uniform
    proposal distribution. Return n masses.
    Adapted from http://python4mpia.github.io/fitting_data/
    """
    mf = get_massfunc(massfunc)
    logmmin = np.log10(mf.mmin)
    logmmax = np.log10(mf.mmax)
    masses = [] 
    #Begin loop to build up exactly n masses, hopefully running once in most
    #cases.
    while len(masses) < n: 
        #Guess number of trials needed to have n accepted samples.
        n_try =  np.ceil((1.*n - len(masses)) / mc_frac[massfunc])
        # print n, n_try
        #Draw proposals from log-uniform distribution, increases efficiency.
        m = 10. ** ((logmmax - logmmin) * np.random.random(n_try) + logmmin)
        
        #SHOULD I USE THE INTEGRAL FORM OF THE MF HERE? I THINK NOT, BUT
        #http://python4mpia.github.io/fitting_data/ DOES?
        #################################################################
        #   ANSWER: This is the log form (dn/dlogm), which looks the same as
        #   the integral form for a Salpeter IMF. Since our proposals are 
        #   log-uniform, we must use the log form of the PDF to sample the
        #   true underlying PDF. BOOM
        ################################################################
        like = mf.dndlogm(m) 
        maxlike = mf.dndlogm(mostcommonmass[massfunc])

        #Accept m randomly, and give to masses 
        u = maxlike * np.random.random(n_try)
        masses = np.concatenate([masses, m[u < like]])
        # print len(masses)
        if len(masses) > n:
            #Keep first n samples.
            masses = masses[:n]
        elif len(masses) < n: print "Need more samples! Relooping"

    return masses

sample_dict = {'mc':mc_sample, 'inverse':inverse_sample}

def sample_imf(tot, tot_is_number=False, massfunc='salpeter',
        samplefunc='inverse', verbose=False, silent=False,
        mtol=0.5, mcm=0.01, **kwargs):
    """
    Stochastically sample an IMF to a total mass of mtot. 
    
    mtol is the tolerance between mtot and the final mass of the cluster.a
    if user defined massfunc, specify mcm (most common mass). Set to 0.01 so 
    that if not specified, the sample_imf still works, just not optimized.
    """
    
    sample = sample_dict[samplefunc]
    if type(massfunc) is str:
        mcm = mostcommonmass[massfunc]

    if tot_is_number:
        ntot = tot
        masses = sample(ntot, massfunc=massfunc, **kwargs)
        mtot = masses.sum()
        if verbose: print "Sampled %i stars. Total mass is %g solar masses." % (ntot, mtot)
    else:
        #Guess the number of stars needed using the most common mass of the given
        #IMF 
        mtot = tot
        ntot = np.ceil(mtot / mcm)
        masses = sample(ntot, massfunc=massfunc, **kwargs)
        #Sum the sampled masses and check if the sum is within mtol from mtot
        msum = masses.sum()
        if verbose: print "Sampled %i stars. Total mass is %g solar masses." % (ntot, msum)

        if msum > mtot + mtol:
            #Throw away as many samples from the end of masses as necessary to 
            #bring msum under mtot+mtol

            mcum = masses.cumsum()
            over_ind = np.argmax(mcum > mtot)
            masses = masses[:over_ind]
            msum = masses.sum()
            if verbose: print "Selected first %i out of %i samples to get %g total solar masses." % (over_ind, len(mcum), msum)

        else:
            while msum < mtot - mtol:
                #Add on as many samples as needed to reach the requested total
                #mass mtot.
                nnew = np.ceil((mtot - msum) / mcm)
                newmasses = sample(nnew, massfunc=massfunc, **kwargs)
                masses = np.concatenate([masses, newmasses])
                msum = masses.sum()
                if verbose:  print "Sampled %i new stars. Total mass is %g solar masses" % (nnew, msum)

                if msum > mtot + mtol:
                    #Throw away as many samples from the end of masses as necessary to 
                    #bring msum under mtot+mtol

                    mcum = masses.cumsum()
                    over_ind = np.argmax(mcum > mtot)
                    masses = masses[:over_ind]
                    msum = masses.sum()
                    if verbose: print "Selected first %i out of %i samples. Total mass is %g solar masses." % (over_ind, len(mcum), msum)

            if not silent: print "Total mass of %i stars is %g solar masses (%g requested)" % (len(masses), msum, mtot)

    return masses

def getML(m, iso='girardi02', isofolder='/Users/jesse/imf/iso/',
        Mcol=None, Lcol=None, Mlog=False, Llog=True, mag=False, magsun=4.83):
    """
    Interpolate the specified isochrone table, computing the corresponding
    luminosities for the input masses. If a mass if outside the range of the
    isochrone, return L=0 for that mass. Stars below the mass range are not
    covered by the isochrone, while stars above the mass range should be
    stellar remnants (thus L=0) by the age of the isochrone. Return the total
    M/L ratio for the population.

    All parameters past iso need only be changed if the user is specifying
    an isochrone table not included in isofolder, or if user specifies other
    columns in isochrone, i.e. mag=True means that the luminosity column is
    interpreted as an absolute magnitude, with magsun specified for the desired
    photometric band (Default V).

    Included isochrones:
    girardi02 - Used by Hernandez et al (2012).
        Age = 10.5 Gyr
        [Fe/H] = -3
    
    bressan12 - Newest version.
        Age = 10.5 Gyr
        [Fe/H] = -3

        
    """
    nheadlines = {'girardi02':10, 'bressan12':12} #Lines before column names
    if iso in nheadlines:
        path = isofolder + iso + '.dat'
        isotable = np.genfromtxt(path, skip_header=nheadlines[iso], names=True)
        M_iso = isotable['M_ini']
        if mag:
            L_iso = 10. ** ((magsun - isotable['V']) / 2.5)
        else:
            L_iso = 10. ** isotable['logLLo']

    else:
        isotable = np.genfromtxt(iso)
        if Mlog:
            M_iso = 10. ** isotable[:,Mcol]
        else:
            M_iso = isotable[:,Mcol]
        if mag:
            L_iso = 10. ** ((magsun - isotable[:,Lcol]) / 2.5)

        elif Llog:
            L_iso = 10. ** isotable[:,Lcol]

        else:
            L_iso = isotable[:,Lcol]
    
    L = np.interp(m, M_iso, L_iso, left=0., right=0.)
    return np.sum(m)/np.sum(L)


