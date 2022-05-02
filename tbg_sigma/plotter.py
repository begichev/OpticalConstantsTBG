import matplotlib as plt
import numpy as np
from Re_sigma_xx import plot_optical_cond

levels = 200
angle=3.265
filename = 'tbg'+str(angle)+'levels_'+str(levels)+'_sigma.txt'
plot_save_to = 'tbg'+str(angle)+'levels_'+str(levels)+'_sigma.pdf'

plot_optical_cond(filename, plot_save_to)