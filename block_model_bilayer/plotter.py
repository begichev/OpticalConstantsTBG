import matplotlib as plt
import numpy as np
from Re_sigma_xx import plot_optical_cond

filename = 'bilayer_sigma_200_0.001_gamma0.002_ewin0.02n41_changev.txt'
plot_save_to = 'bilayer_sigma_200_0.001_gamma0.002_ewin0.02n41_changev_scatter.pdf'

plot_optical_cond(filename, plot_save_to)