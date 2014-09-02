import sys
import numpy as np
"""
Functions for generating a synthetic pulsar population. 
Adapted from PsrPopPy (Bates et al. 2014)

Need to find an implementation for pointing geometry. I.E. the pulsars
simulated here are a priori beaming at Earth.
"""
pi = np.pi
d2r = 2*pi / 360.

#Dispersion measures at the location of various UFDs as calculated from
#NE2001 galactic free electron model. This is only the DM w/in the MW along
#each line of sight, does not include DM in UFD or IGM.

UFD_DM = {'SegI':37.29, 'Com':19.73} #in pc/cm^3
UFD_lb = {'SegI':[220.4882, 50.4196], 'Com':[241.8867, 83.6112]}
#Distances to UFDs from Martin et al. 2008
UFD_d = {'SegI':20, 'Com':44} #in kpc
UFD_derr = {'SegI':2, 'Com':4} #+/- 1 sigma 

def lum(n, disttype='lognorm', distpars=[-1.1, 0.9]):
    """
    Assign a luminosity to n pulsars. lum is the pseudoluminosity, 
    L_nu @ 1400 MHZ (by convention), such that S_nu = L_nu/d^2
    """
    if disttype == 'lognorm':
        lum = 10. ** np.random.normal(distpars[0], distpars[1], n)

    return lum

def period(n, disttype='lor12', distpars=[]):
    """
    Assign periods to n pulsars. The included period
    distributions are as follows:

    lor12: Discrete empirical distribution defined by Lorimer et al. 2012, as
    described in PsrPopPy populate.py (Bates et al. 2014), for millisecond
    pulsars.
    """
    if disttype == 'lor12':
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


def angles(p):
    """
    Generate pulse width. Following Bates et al. 2014 (PsrPopPy)
    This ASSUMES that the pulsar beam intersects our line of sight,
    thus |beta| < rho. To check if this is true, compute the beaming
    fraction (fraction of the sphere which is ever intersected by the beam)
    using beam_frac, this will represent the probability that the pulsar is
    beamed towards us.

    The pulse width is the fraction (represented by a fraction of 360 deg) of
    the period that is observable to us as a pulse. e.g., a pulsar whose beam
    is at 90 deg to its rotation axis (alpha = 90) and we are observing
    equator-on (beta = 0) has width equal to twice the opening angle of the
    beam (rho), or the angular diameter of the beam. Also return rho and alpha
    for use in beam_frac


    rho: opening angle of pulsar beam (deg)
    beta: angle between magnetic axis and line of sight (deg)
    alpha: angle between magnetic and rotation axes (deg)
    """
    n = len(p)
    p = np.array(p)
    pcut = 30.0 #in ms
    
    #calculate opening angle of pulsar beam in degrees
    r = np.random.uniform(-0.15, 0.15, n)
    rho = np.zeros_like(p)
    rho[p > pcut] = 5.4 * (p[p > pcut] / 1000.) ** (-1./2)
    rho[p <= pcut] = 31.2 * np.ones_like(p[p <= pcut]) #= 31.2 deg
    rho = 10. ** (np.log10(rho) + r)
    
    #choose a beta and alpha in degrees
    beta = np.random.uniform(-1, 1, n) * rho
    alpha = abs(np.arccos(np.random.uniform(0, 1, n)) / d2r) 

    width = (np.sin(0.5*d2r*rho)**2. - np.sin(0.5*d2r*beta)**2.) / \
    (np.sin(d2r*alpha) * np.sin(d2r*(alpha + beta)))
    
    #For invalid values of width, replace width and rho with 0. to ensure that
    #beam_frac = 0.
    width.put(np.where((width < 0.0) | (width > 1.0)), 0.)
    rho.put(np.where((width < 0.0) | (width > 1.0)), 0.) 
    # rho is set to 0 in
    # PsrPopPy, but I don't think that's right, the opening angle exists even
    # if the pulsar not observable.
    #Yes, that's true, but the SNR formula uses EFFECTIVE width, which will
    #give a pulsar with invalid value of beamwidth a nonzero SNR, unless we 
    #guarantee that beam_frac = 0 by setting rho = 0
    width = np.sqrt(width)
    
    #convert width into degrees 0 -> 360
    width = np.arcsin(width) * 4.0 / d2r
    return {'width':width, 'rho':rho, 'alpha':alpha}

def beam_frac(rho, alpha):
    """
    Calculate the fraction of 4pi steradians covered by the beam.
    """
    thetal = np.maximum(0., alpha - rho)
    thetau = np.minimum(90., alpha + rho)
    return np.cos(thetal * d2r) - np.cos(thetau * d2r)

def spindex(n, disttype='gauss', distpars=[-1.4, 0.96]):
    """
    Generate a power law spectral slope, from a normal distribution 
    
    gauss[mean, sigma]: pick a spectral index from a normal distribution,
    with default mean= -1.4, sigma= 0.96 (from Bates+2014)
    """
    if disttype=='gauss':
        return np.random.normal(distpars[0], distpars[1], n)

def SNR_calc(L, p, width, spindex, d=None, l=None, b=None, DM=None, 
        tobs=None, G=None, beta=None, npol=None,
        tsamp=None, Tsys=None, f=None,
        deltaf=None, bw_chan=None, ref_freq=1400., SNR=None, **kwargs):
    """
    L: pseudoluminosity defined at ref_freq [mJy kpc^2]
  
    width: pulse width [degrees]
    spindex: spectral index, assuming power-law radio spectrum
    d: distance to pulsar [kpc]
    l, b: galactic coordinates of pulsar
    DM: dispersion measure 
    Tsys: system temperature [K]
    f: observing frequency [MHz]
    deltaf: bandwidth [MHz]
    bw_chan: bandwith of single channel [MHz]
    ref_freq: frequency of pseudoluminosity L [MHz]
    SNR: signal-to-noise, if specified return Snu
    """

    Snu = L * ((f / ref_freq) ** spindex) / (d ** 2.)
  
    #Compute effective pulse width; tdiss is dispersive smearing, tscatt is
    #smearing due to free electron scattering. 
    tdiss = 8.3e6 * DM * bw_chan / (f ** 3.) #
    logtscatt = -6.46 + 0.154 * np.log10(DM) + 1.07 * (np.log10(DM))**2.\
            - 3.86 * np.log10(f/1000.)
   
    tscatt = 10. ** (logtscatt + np.random.normal(0., 0.8))

    Weff = np.sqrt((width * p / 360.) ** 2. + tsamp ** 2. + tdiss ** 2. +
            tscatt ** 2.) 
    delta = Weff / p

    Tsky = tskypy(l, b, f)
    Ttot = Tsys + Tsky  

    SNR = Snu * G * np.sqrt(npol * tobs * deltaf) * np.sqrt((1 - delta) / delta) / (beta * Ttot)
    SNR.put(np.where(delta > 1.), -1.) #Pulsar is smeared out completely.
    return SNR

def readtskyfile(path):
    """
    From PsrPopPy - galacticops.py
    Read in tsky.ascii into a list from which temps can be retrieved
    """

    tskylist = []
    with open(path) as f:
        for line in f:
            str_idx = 0
            while str_idx < len(line):
                # each temperature occupies space of 5 chars
                temp_string = line[str_idx:str_idx+5]
                try:
                    tskylist.append(float(temp_string))
                except:
                    pass
                str_idx += 5

    return tskylist

def tskypy(l, b, freq, path='tsky.ascii'):
    """
    From PsrPopPy - survey.py
    Calculate tsky from Haslam table, scale to survey frequency
    """
    # ensure l is in range 0 -> 360
    if l < 0.:
        l += 360

    # convert from l and b to list indices
    j = b + 90.5
    if j > 179:
        j = 179

    nl = l - 0.5
    if l < 0.5:
        nl = 359
    i = float(nl) / 4.
    tskylist = readtskyfile(path)
    tsky_haslam = tskylist[180*int(i) + int(j)]
    # scale temperature before returning
    return tsky_haslam * (freq/408.0)**(-2.6)


def obs_UFD(L, p, width, spindex, tobs=None, det='ALFA', UFD='Com', **kwargs):
    
    # L = lum(n)
    # p = period(n)
    # angles = angles(p)

    pars = detpars[det]
    d = UFD_d[UFD]
    l = UFD_lb[UFD][0]
    b = UFD_lb[UFD][1]
    DM = UFD_DM[UFD]
    print d, DM
    # beta, G, tsamp, Tsys, f, deltaf, bw_chan, npol, fwhm = pars['beta'], pars['G'],
    # pars['tsamp'], pars['Tsys'], pars['f'], pars['deltaf'], pars['bw_chan'], pars['npol'],
    # pars['fwhm']
    print pars
    
    SNR = SNR_calc(L, p, width, spindex, d, l, b, DM, tobs, **pars)
    return SNR


       
    



# def genalpha(p):
#     """
#     Generate the angle between magnetic and rotation axes.
#     """
#     n = len(p)
#     return abs(np.arccos(np.random.uniform(0, 1)) / d2r)a

# def genbeta(p):
detpars = {'ALFA':{'beta':1.1, 'G':8.5, 'tsamp':0.064, 'Tsys':25., 'f':1374.,
'deltaf':300., 'bw_chan':0.3, 'npol':2., 'fwhm':3.6}, 'PUPPI_327':{'beta':1.3,
'G':11., 'tsamp':0.08192, 'Tsys':90, 'f':327., 'deltaf':100, 'bw_chan':0.01,
'npol':2., 'fwhm':14.5}}
#http://www.naic.edu/puppi-observing/
#down to "Typical Setups - Fast4K"




