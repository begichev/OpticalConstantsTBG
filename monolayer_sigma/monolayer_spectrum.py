import numpy as np

N = 100
array = []
#hbarv = 1.5*1.42*2.97 # eV
t = 2.7
hbarv = 0.5*np.sqrt(3)*2.46*t


def Hamiltonian(kx, ky):
	ham = np.zeros((2,2),dtype='complex')
	ham[0][0] = ham[1][1] = 0
	ham[1][0] = -(kx + 1.j*ky)*hbarv
	ham[0][1] = -(kx - 1.j*ky)*hbarv
	return ham

a = 2.46 # angstrom
k = 2*np.pi/a # inverse angstrom
kstep0 = 0.001
kstep = kstep0*k 
eigenenergies = []
eigenfunctions = []

for i in np.arange(-N,N+1): # -N ... 0 ... N
	for j in np.arange(-N,N+1):
		ham_mono = Hamiltonian(i*kstep, j*kstep)
		eigenvalues, eigenvectors = np.linalg.eigh(ham_mono)
		eigenenergies.append(eigenvalues)
		eigenfunctions.extend(eigenvectors)

real_eigenenergies = np.real(eigenenergies)
energies_save_to='monolayer_energies_'+str(N)+'_'+str(kstep0)+'.txt'
wavefunctions_save_to='monolayer_wavefunctions_'+str(N)+'_'+str(kstep0)+'.txt'

np.savetxt(energies_save_to, real_eigenenergies)
np.savetxt(wavefunctions_save_to,eigenfunctions)