import numpy as np
from Re_sigma_xx import calc_optical_conductivity, occup_0K, pick_up_transition_pairs

#eigenenergies = np.loadtxt("monolayer_energies.txt", dtype=float)
#eigenfunctions = np.loadtxt("monolayer_wavefunctions.txt", dtype=complex)

#print(np.size(eigenenergies), '\t', np.size(eigenfunctions))

#calc_optical_conductivity()

#monolayer_occup_save_to='monolayer_occup.txt'
#np.savetxt(monolayer_occup_save_to, occup_0K(eigenenergies))

#transition_pairs_save_to = 'monolayer_transition_pairs0.5eV.txt'
#np.savetxt(transition_pairs_save_to, pick_up_transition_pairs(eigenenergies,0.5, 0.3, occup_0K(eigenenergies)))

#transition_pairs = np.loadtxt('monolayer_transition_pairs0.5eV.txt')
#print(eigenenergies[int(transition_pairs[0][0])], '\n')
#print(eigenenergies[int(transition_pairs[0][1])])

#print(transition_pairs[0])

calc_optical_conductivity()
#vals = np.loadtxt("monolayer_energies_20_0.005_lined.txt",dtype=float)
#occup = occup_0K(vals[0])
#i = 1000
#print(vals[i])
#print(pick_up_transition_pairs(vals[i],0.5,0.05,occup))


#vecs = np.loadtxt("monolayer_wavefunctions_20_0.005_lined.txt",dtype=complex)
#vals = np.loadtxt("monolayer_energies_20_0.005_lined.txt",dtype=float)
#print(vals[len(vals)])

#print(eigenfunctions[0])