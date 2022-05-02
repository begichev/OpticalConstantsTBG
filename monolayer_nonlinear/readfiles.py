import numpy as np
from Re_sigma_xx import calc_optical_conductivity, occup_0K, pick_up_transition_pairs

calc_optical_conductivity()
"""
kstep = 0.001
N = 100
energies_load_from='monolayer_energies_'+str(N)+'_'+str(kstep0)+'.txt'
wavefunctions_load_from='monolayer_wavefunctions_'+str(N)+'_'+str(kstep0)+'.txt'
sigma_save_to = 'monolayer_sigma'+str(n)+'_'+str(0.001)+'_gamma'+str(gamma)'.txt'

vals = np.loadtxt("monolayer_energies_100_0.001.txt", dtype=float)
"""