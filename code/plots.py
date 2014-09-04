#Make various plots investigating the effects of stochasticity 
#sampling the IMF
import numpy as np
import matplotlib.pyplot as plt
import imf
def MtoL(plotfile='MtoL.pdf', niters=2000, iso='girardi02', massfunc='larson',
        samplefunc='mc', total_masses=[5e2, 1e3, 3e3, 1e4, 4e4]):
    """
    Replicate Figure 1 of Hernandez et al (2012), which shows the effect
    of stochastic sampling on the distribution of M/L ratios. The default 
    parameters are those used by Hernandez. 
    """
    bins = np.linspace(0, 15, 50)
    xlabel = r'M/L$_V$'
    ylabel = 'Number'

    f, ax = plt.subplots(1)

    for i in range(len(total_masses)):
        mtot = total_masses[i]
        ML = np.array([])
        for j in range(niters):
            m = imf.sample_imf(mtot, massfunc=massfunc, samplefunc=samplefunc)
            MLnew = imf.getML(m, iso=iso, mag=True)
            ML = np.concatenate([ML, [MLnew]])
            if (j+1) % 10 == 0:
                print "Iteration %i out of %i for %e solar masses." % (j+1,
                niters, total_masses[i])
        ax.hist(ML, bins=bins, histtype='step', label=str(total_masses[i]) + r' M$_\odot$')
           
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title('Distribution of M/L in '+str(niters)+' realizations of '+massfunc.capitalize()+' IMF.')
    plt.savefig(plotfile) 


def minflux_period(tobs=[12.*3600, 24.*3600., 48.*3600.], spindex=-1.6,
        SNRmin=[9.], UFD=['Com', 'SegI'],
        ref_freq=1400., det='PUPPI_327', plotfile='327PUPPI_sensitivity.pdf'):
    """
    Plot the sensitivity of an observing setup in terms of the minimum pulsar
    flux detectable (@ SNR = 9), as a function of the pulsar spin period. 
    Compare to Fig. 3 of Stovall et al. 2014 (GBNCC survey) 
    Add histograms of expected distributions of pulsar Snu and period
    """
    import pulsar
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    pars = pulsar.detpars[det]
    x = np.logspace(-1.5, 2, 1000)
    duty_ref = 0.08
    width_ref = 360. * duty_ref   
    L_ref = 0.6 #Dummy reference luminosity.
    
    #Distribution of p 10,0000 beaming pulsars @ d_ref
    p = pulsar.period(100000.)
    print np.mean(p)
    
    f, axes = plt.subplots(2, figsize=(6,12))
    for iax in range(len(axes)):
        ax = axes[iax]
        DM = pulsar.UFD_DM[UFD[iax]]
        d_ref = pulsar.UFD_d[UFD[iax]]
        l, b = pulsar.UFD_lb[UFD[iax]]
        Snu_ref = L_ref * ((pars['f'] / ref_freq) ** spindex / (d_ref ** 2.)) 
        #Distribution of Snu for 10,000 BEAMING pulsars @ d_ref
        Snu = pulsar.lum(100000.) * (pars['f']/ref_freq) ** spindex / d_ref ** 2.
        for i in tobs:
            for j in SNRmin:
                SNRref = pulsar.SNR_calc(L_ref, x, width_ref, spindex, d=d_ref, l=l, b=b,
                    DM=DM, tobs=i, **pars)
                Snu_min = Snu_ref * j / SNRref 
                
                Snu_min_samp = np.interp(p, x, Snu_min)
                detectfrac = np.size(np.where(Snu > Snu_min_samp)) / 100000.
            
                ax.plot(x, Snu_min, label=r"t$_{obs}$ = %i hours (%.3g%% detected at %i$\sigma$)" %
                        (i/3600, detectfrac*100., j))
                ax.loglog()
        ax.legend(fontsize=8, loc='lower center')
        xlab = 'spin period (ms)'
        ylab = r'Minimum S$_{327 MHz}$ (mJy) $\alpha = -1.6$'
        if iax==1:
            ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(10**-4., 1.)

        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
        axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
        plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels() +
                axHistx.get_yticklabels() + axHisty.get_xticklabels(),
            visible=False)
        
        axHistx.hist(p, bins=np.logspace(np.log10(min(p)), np.log10(max(p))))
        axHisty.hist(Snu, bins=np.logspace(np.log10(min(Snu)), np.log10(max(Snu))),
            orientation='horizontal')
        axHistx.set_title(UFD[iax])
    # ax.set_title(r'12 hour $9\sigma$ sensitivity towards Coma of 327 MHz receiver with PUPPI backend')
    plt.savefig(plotfile)

def pulsarmap(plotfile='knownmsps.pdf', pulsarfile='pulsars.dat'):
    """
    Plot the coordinates of all known pulsars, both radio and gamma ray, with UFDs
    marked.

    psrcat.dat is a table from the ATNF pulsar database, and this code assumes
    this format.
    """
    from astropy.io import ascii
    a = ascii.read('psrcat.dat', format='basic', fill_values=[('','0'), ('*','0')])
    
