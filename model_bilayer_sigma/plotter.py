import matplotlib as plt
import numpy as np
from Re_sigma_xx import plot_optical_cond

kstep0 = 0.001
gamma = 0.04
filename = 'ab_bilayer_sigma'+str(kstep0)+'_gamma'+str(gamma)+'.txt'
plot_save_to = 'ab_bilayer_sigma'+str(kstep0)+'_gamma'+str(gamma)+'.pdf'

plot_optical_cond(filename, plot_save_to)