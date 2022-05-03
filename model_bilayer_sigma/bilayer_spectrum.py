import numpy as np
from scipy.linalg import block_diag

t = 2.7 # eV hopping between monolayer neighbours
t_layers = 0.4 # eV hopping between A and B atoms in AB bilayer
a0 = 1.42 # angstrom-1

def NearNeighbSum(kx,ky):
	return np.exp(-1.j*a0*kx)*(1+2*np.exp(0.5*1.j*3*a0*kx)*np.cos(0.5*np.sqrt(3)*a0*ky))

def Hamiltonian(kx, ky):

	S = NearNeighbSum(kx,ky)
	block1 = np.zeros((2,2), dtype='complex')
	block1[0][0] = block1[1][1] = 0
	block1[1][0] = -t*np.conj(S)
	block1[0][1] = -t*S
	block2 = np.conj(block1)
	ham = block_diag(block1, block2)
	"""
	sigma_x = np.matrix([[0,1],[1,0]])
	off_diag = np.matrix([[t_layers,0],[0,0]])
	ham += np.kron(sigma_x, off_diag)
	"""
	ham[2][0]=ham[0][2]=t_layers
	return ham

def CalcSpectra(kstep0=0.001):
	a = 2.46 # angstrom
	k = 2*np.pi/a # inverse angstrom
	Kx = k/np.sqrt(3)
	Ky = k/3 # Dirac point coordinates
#	kstep0 = 0.0005
	steps = int(2/3/kstep0) # number of points in brilloine zone near K. whole BZ is covered with 3*steps
	kstepix = -0.5*np.sqrt(3)*kstep0*k
	kstepiy = 0.5*kstep0*k
	kstepjx = 0
	kstepjy = -kstep0*k
	eigenenergies = []
	eigenfunctions = []
	for i in np.arange(steps): 
		for j in np.arange(steps):
			ham_mono = Hamiltonian(Kx+i*kstepix+j*kstepjx, Ky+i*kstepiy+j*kstepjy)
			eigenvalues, eigenvectors = np.linalg.eigh(ham_mono)
			eigenenergies.append(eigenvalues)
			eigenfunctions.extend(eigenvectors)
	real_eigenenergies = np.real(eigenenergies)
	energies_save_to='ab_bilayer_energies_'+str(kstep0)+'.txt'
	wavefunctions_save_to='ab_bilayer_wavefunctions_'+str(kstep0)+'.txt'

	np.savetxt(energies_save_to, real_eigenenergies)
	np.savetxt(wavefunctions_save_to,eigenfunctions)
