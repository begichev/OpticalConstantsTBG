import numpy as np
from Re_sigma import calc_optical_conductivity

eigenenergies = np.loadtxt("monolayer_energies.txt", dtype=float)
eigenfunctions = np.loadtxt("monolayer_wavefunctions.txt", dtype=complex)

#print(np.size(eigenenergies), '\t', np.size(eigenfunctions))

calc_optical_conductivity()

#print(eigenfunctions[0])