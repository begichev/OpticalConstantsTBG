import numpy as np
from scipy.linalg import block_diag

t = 3.16 # eV hopping between monolayer neighbours
t_layers = 0.4 # eV hopping between A and B atoms in AB bilayer
hbarv = 0.5*np.sqrt(3)*2.46*t
a0 = 1.42 # angstrom-1
g1 = 0.381
g3 = 0.38
g4 = 0.14
eB1 = eA2 = 0.022


def NearNeighbSum(kx,ky):
	return 2*np.exp(-0.5*1.j*ky*a0)*np.cos(0.5*kx*a0*np.sqrt(3))+np.exp(1.j*ky*a0)

def Hamiltonian(kx, ky):

	S = NearNeighbSum(kx,ky)
	block1 = np.zeros((2,2), dtype='complex')
	block1[0][0] = block1[1][1] = 0
	block1[1][0] = -t*np.conj(S)
	block1[0][1] = -t*S
	block2 = np.conj(block1)
	ham = block_diag(block1, block2)
	ham[2][0] = g4*np.conj(S)
	ham[3][0] = -g3*S
	ham[1][1] = eB1
	ham[2][1] = g1
	ham[3][1] = g4*np.conj(S)
	ham[0][2] = g4*S
	ham[1][2] = g1
	ham[2][2] = eA2
	ham[0][3] = -g3*np.conj(S)
	ham[1][3] = g4*S
	return ham

a = 2.46 # angstrom
k = 2*np.pi/a # inverse angstrom
Kx = 2/3*k
Ky = 0 # Dirac point coordinates
N = 10
kstep0 = 0.02
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
energies_save_to='real_bilayer_energies_'+str(N)+'_'+str(kstep0)+'.txt'
wavefunctions_save_to='real_bilayer_wavefunctions_'+str(N)+'_'+str(kstep0)+'.txt'

np.savetxt(energies_save_to, real_eigenenergies)
np.savetxt(wavefunctions_save_to,eigenfunctions)
