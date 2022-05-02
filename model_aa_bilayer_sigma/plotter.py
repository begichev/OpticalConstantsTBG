import matplotlib as plt
import numpy as np
from Re_sigma_xx import plot_optical_cond

kstep0 = 0.001
#N = 50
gamma = 0.02
t_layers = 0.2 
filename = 'aa_bilayer_sigma'+str(kstep0)+'_gamma'+str(gamma)+'interlayerhop'+str(t_layers)+'.txt'
plot_save_to = 'aa_bilayer_sigma'+str(kstep0)+'_gamma'+str(gamma)+'interlayerhop'+str(t_layers)+'.pdf'

plot_optical_cond(filename, plot_save_to)