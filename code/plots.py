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
            # print MLnew
            ML = np.concatenate([ML, [MLnew]])
        ax.hist(ML, bins=bins, histtype='step', label=str(total_masses[i]) + r' M$_\odot$')
           
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title('Distribution of M/L in '+str(niters)+' realizations of '+massfunc.capitalize()+' IMF.')
    plt.savefig(plotfile) 
