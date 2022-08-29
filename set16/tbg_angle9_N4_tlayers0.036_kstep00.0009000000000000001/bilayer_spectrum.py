import numpy as np
from scipy.linalg import block_diag

t = 2.7 # eV hopping between monolayer neighbours
t_layers = 0.4 # eV hopping between A and B atoms in AB bilayer
hbarv = 0.5*np.sqrt(3)*2.46*t
a0 = 1.42 # angstrom-1

def NearNeighbSum(kx,ky):
	return 2*np.exp(0.5*1.j*kx*a0)*np.cos(0.5*ky*a0*np.sqrt(3))+np.exp(-1.j*kx*a0)

a = 2.46 # angstrom
k = 2*np.pi/a # inverse angstrom
kstep0 = 0.001
kstep = kstep0*k 
N = 10 

Kx = 0
Ky = -2/3*k # Dirac point coordinates
blocks = []

def Hamiltonian(N, kx, ky):

	S = NearNeighbSum(kx,ky)
	block1 = np.zeros((2,2), dtype='complex')
	block1[0][0] = block1[1][1] = 0
	block1[1][0] = t*np.conj(S)
	block1[0][1] = t*S
	block2 = np.conj(block1)
	ham = block_diag(block1, block2)
	sigma_x = np.matrix([[0,1],[1,0]])
	off_diag = np.matrix([[t_layers,0],[0,0]])
	ham += np.kron(sigma_x, off_diag)
	return ham



"""
eigenenergies = []
eigenfunctions = []
"""

for i in np.arange(-N,N+1): # -N ... 0 ... N
	for j in np.arange(-N,N+1):
		ham_mono = Hamiltonian(N, Kx+i*kstep, Ky+j*kstep)
		blocks.append(ham_mono)
"""
		eigenvalues, eigenvectors = np.linalg.eigh(ham_mono)
		eigenenergies.append(eigenvalues)
		eigenfunctions.extend(eigenvectors)
"""

"""
real_eigenenergies = np.real(eigenenergies)
"""

hamiltonian = block_diag(*blocks)
#eigenvalues, eigenvectors = np.linalg.eigh(ham_mono)
eigenenergies, eigenfunctions = np.linalg.eigh(hamiltonian)
real_eigenenergies = np.real(eigenenergies)

energies_save_to='block_bilayer_energies_zerocoupling'+str(N)+'_'+str(kstep0)+'nobrackets.txt'
wavefunctions_save_to='block_bilayer_wavefunctions_zerocoupling'+str(N)+'_'+str(kstep0)+'nobrackets.txt'

#np.savetxt(f, arr, delimiter=' ', newline='\n', header='', footer='', comments='# ')
np.savetxt(energies_save_to, real_eigenenergies)
np.savetxt(wavefunctions_save_to,eigenfunctions, delimiter=' ', newline='\n', header='', footer='')
