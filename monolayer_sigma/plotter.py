import matplotlib as plt
import numpy as np
from Re_sigma_xx import plot_optical_cond

filename = 'monolayer_sigma_100_0.001_gamma0.002_ewin0.05.txt'
plot_save_to = 'monolayer_sigma_100_0.001_gamma0.002_ewin0.05.pdf'

plot_optical_cond(filename, plot_save_to)