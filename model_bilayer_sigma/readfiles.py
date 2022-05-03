import numpy as np
from Re_sigma_xx import calc_optical_conductivity, plot_optical_cond
from bilayer_spectrum import CalcSpectra

kstep0 = 0.001
omegas=np.arange(0.05, 1.5, 0.05)
gamma = 0.04 # in units of t_layers
t_layers = 0.4 # eV hopping between A and B atoms in AB bilayer
CalcSpectra(kstep0)
calc_optical_conductivity(omegas=omegas, gamma=gamma*t_layers,kstep0=kstep0)
filename = 'ab_bilayer_sigma'+str(kstep0)+'_gamma'+str(gamma)+'.txt'
plot_save_to = 'ab_bilayer_sigma'+str(kstep0)+'_gamma'+str(gamma)+'.pdf'

plot_optical_cond(filename, plot_save_to)
