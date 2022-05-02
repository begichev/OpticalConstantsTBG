import matplotlib as plt
import numpy as np
from Re_sigma_xx import plot_optical_cond

kstep0 = 0.001
N = 100
gamma = 0.02
filename = 'monolayer_sigma'+str(N)+'_'+str(kstep0)+'_gamma'+str(gamma)+'.txt'
plot_save_to = 'monolayer_sigma'+str(N)+'_'+str(kstep0)+'_gamma'+str(gamma)+'.pdf'

plot_optical_cond(filename, plot_save_to)