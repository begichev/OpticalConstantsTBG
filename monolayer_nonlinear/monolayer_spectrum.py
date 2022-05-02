import numpy as np

array = []
t = 2.7
a0 = 1.42 # angstrom

def NearNeighbSum(kx,ky):
	return np.exp(-1.j*a0*kx)*(1+2*np.exp(0.5*1.j*3*a0*kx)*np.cos(0.5*np.sqrt(3)*a0*ky))

def Hamiltonian(kx, ky):
	ham = np.zeros((2,2),dtype='complex')
	S = NearNeighbSum(kx,ky)
	ham[0][0] = ham[1][1] = 0
	ham[1][0] = -t*np.conj(S)
	ham[0][1] = -t*S
	return ham

a = 2.46 # angstrom
k = 2*np.pi/a # inverse angstrom
Kx = k/np.sqrt(3)
Ky = k/3 # Dirac point coordinates
kstep0 = 0.001
N = 100
kstep = kstep0*k 
eigenenergies = []
eigenfunctions = []

for i in np.arange(-N,N+1): # -N ... 0 ... N
	for j in np.arange(-N,N+1):
		ham_mono = Hamiltonian(Kx+i*kstep, Ky+j*kstep)
		eigenvalues, eigenvectors = np.linalg.eigh(ham_mono)
		eigenenergies.append(eigenvalues)
		eigenfunctions.extend(eigenvectors)

real_eigenenergies = np.real(eigenenergies)
energies_save_to='monolayer_energies_'+str(N)+'_'+str(kstep0)+'.txt'
wavefunctions_save_to='monolayer_wavefunctions_'+str(N)+'_'+str(kstep0)+'.txt'

np.savetxt(energies_save_to, real_eigenenergies)
np.savetxt(wavefunctions_save_to,eigenfunctions)