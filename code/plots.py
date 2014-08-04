#Make various plots investigating the effects of stochasticity 
#sampling the IMF
import numpy as np
import matplotib.pyplot as plt
import imf

def MtoL(plotfile='MtoL', niters=2000., iso='girardi02', imf='larson',
        mtot = [5e2, 1e3, 3e3, 1e4, 4e4]):
    """
    Replicate Figure 1 of Hernandez et al (2012), which shows the effect
    of stochastic sampling on the distribution of M/L ratios. Use
    """
    mtot = np.array(mtot)
    mtot = [5e2, 1e3, 3e3, 1e4, 4e4]


    f, ax = plt.subplots(1)

    xlabel = 'M/L'
    ylabel = 'Number'

