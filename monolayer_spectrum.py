import numpy as np

N = 50
array = []
hbarv = 1.5*1.42*2.97 # eV

def Hamiltonian(kx, ky):
	ham = np.zeros((2,2),dtype='complex')
	ham[0][0] = ham[1][1] = 0
	ham[1][0] = -(kx + 1.j*ky)
	ham[0][1] = -(kx - 1.j*ky)
	return ham

a = 2.46 # angstrom
k = 2*np.pi/a # inverse angstrom
kstep = 0.005*k 
eigenenergies = []
eigenfunctions = []

for i in np.arange(-N,N+1): # -N ... 0 ... N
	for j in np.arange(-N,N+1):
		ham_mono = Hamiltonian(i*kstep, j*kstep)
		eigenvalues, eigenvectors = np.linalg.eigh(ham_mono)
		eigenenergies.extend(eigenvalues)
		eigenfunctions.extend(eigenvectors)

eigenenergies = np.array(eigenenergies)
#eigenenergies = np.ndarray.flatten(eigenenergies)
real_eigenenergies = np.real(eigenenergies)
sorted_eigenenergies = np.sort(real_eigenenergies)
eigenfunctions = np.array(eigenfunctions)
inds = real_eigenenergies.argsort()
sorted_eigenfunctions = eigenfunctions[inds]

energies_save_to='monolayer_energies.txt'
wavefunctions_save_to='monolayer_wavefunctions.txt'

np.savetxt(energies_save_to, sorted_eigenenergies)
np.savetxt(wavefunctions_save_to,sorted_eigenfunctions)