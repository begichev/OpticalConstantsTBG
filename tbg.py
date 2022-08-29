import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import time
import os

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

# output consists of 4 blocks. [0,0] block is for 1st intralayer, [1,1] is for 2d. [1,0] block accounts for scatterings from 1st to 2d layer with delta k = [0, -G1M, G2M].
# [0,1] block is hermitian conjugated [1,0] and has the meaning of scatterings from 2d to 1st layer with delta k = [0, G1M, -G2M]

def TwistHamiltonian(kx,ky,angle,N,t_layers):
	"""
	input: kx,ky from 1st moire brilloine zone
	returns 4*N**2 matrix of twisted bilayer hamiltonian near K point
	"""
	theta = angle*np.pi/180
	k0 = 4/3*np.sin(0.5*theta) # distance between K1 and K2 in MBZ; one side of its hexagon
	G1M = k0*np.sqrt(3)*np.array([-0.5*np.sqrt(3),0.5])
	G2M = k0*np.sqrt(3)*np.array([0.5*np.sqrt(3),0.5])
	K1 = k0*np.array([0.5,0])
	K2 = k0*np.array([-0.5,0])

	#list_mono1 = [[[0,-(i*G1M[1]+j*G2M[1]-K1[1]+ky) + 1.j*(i*G1M[0]+j*G2M[0]-K1[0]+kx)],[-(i*G1M[1]+j*G2M[1]-K1[1]+ky) - 1.j*(i*G1M[0]+j*G2M[0]-K1[0]+kx),0]] for i in np.arange(N) for j in np.arange(N)]
	list_mono1 = [[[0,-(i*G1M[1]+j*G2M[1]-K1[1]+ky) + 1.j*(i*G1M[0]+j*G2M[0]-K1[0]+kx)],[-(i*G1M[1]+j*G2M[1]-K1[1]+ky) - 1.j*(i*G1M[0]+j*G2M[0]-K1[0]+kx),0]] for i in np.arange(-int(0.5*N),int(0.5*N)) for j in np.arange(-int(0.5*N),int(0.5*N))]
	ham_mono1 = block_diag(*list_mono1)
	#list_mono2 = [[[0,-(i*G1M[1]+j*G2M[1]-K2[1]+ky) + 1.j*(i*G1M[0]+j*G2M[0]-K2[0]+kx)],[-(i*G1M[1]+j*G2M[1]-K2[1]+ky) - 1.j*(i*G1M[0]+j*G2M[0]-K2[0]+kx),0]] for i in np.arange(N) for j in np.arange(N)]
	list_mono2 = [[[0,-(i*G1M[1]+j*G2M[1]-K2[1]+ky) + 1.j*(i*G1M[0]+j*G2M[0]-K2[0]+kx)],[-(i*G1M[1]+j*G2M[1]-K2[1]+ky) - 1.j*(i*G1M[0]+j*G2M[0]-K2[0]+kx),0]] for i in np.arange(-int(0.5*N),int(0.5*N)) for j in np.arange(-int(0.5*N),int(0.5*N))]
	ham_mono2 = block_diag(*list_mono2)
	keys0 = np.identity(N**2)
	scat0 = np.kron(keys0, np.array([[1,1],[1,1]]))
	keysG2 = np.eye(N**2, k=1) # 1 step down corresponds to +G2M scattering process
	scatG2 = np.kron(keysG2, np.array([[1,np.exp(-1.j*2/3*np.pi)],[np.exp(1.j*2/3*np.pi),1]]))
	keysG1 = np.eye(N**2, k=-N) # N steps up corresponds to -G1M scattering process
	scatG1 = np.kron(keysG1, np.array([[1,np.exp(1.j*2/3*np.pi)],[np.exp(-1.j*2/3*np.pi),1]]))
	scat = scat0 + scatG1 + scatG2
	ham = np.block([[np.pi*np.sqrt(3)*ham_mono1, t_layers*np.conj(scat)],[t_layers*scat, np.pi*np.sqrt(3)*ham_mono2]])
	return ham 

def WriteEigensystem(kstep0, angle, N, dir_path):
	"""
	writes energy bands:
	conduction: 1,2,3,4
	valence: 1,2,3,4 
	"""
	theta = np.pi*angle/180
	k0 = 4/3*np.sin(0.5*theta)
	isteps = int(0.5*np.sqrt(3)*k0/kstep0)
	jsteps = int(k0/kstep0)
	gridi = np.arange(-isteps,isteps+1,1)
	gridj = np.arange(-jsteps,jsteps+1,1)
	iv,jv = np.meshgrid(gridj,gridi)
	kxv = iv*kstep0
	kyv = jv*kstep0

	energies = []
	vmatrix = []
	levels = 4*N**2
	energieslen = levels+1
	vmatrixlen = int(levels*levels/4)

	for i in np.arange(-isteps,isteps+1,1):
		for j in np.arange(-jsteps,jsteps+1,1):
			kx = kstep0*i 
			ky = kstep0*j 
			if -k0+np.abs(kx)/np.sqrt(3)>ky or k0-np.abs(kx)/np.sqrt(3)<ky:
				energies.append(np.zeros((energieslen)))
				vmatrix.append(np.zeros((vmatrixlen)))
				continue
			ham = TwistHamiltonian(kx,ky,angle,N,t_layers)
			v = TwistVelocity(N)
			eigenvalues, eigenvectors = np.linalg.eigh(ham)
			maskval = eigenvalues<0
			eigenvaluesval = eigenvalues[maskval]
			maskcond = eigenvalues>0 
			eigenvaluescond = eigenvalues[maskcond]
			indsval = np.argsort(np.abs(eigenvaluesval))
			indscond = np.argsort(eigenvaluescond)
			eigenvaluesvalsort = eigenvaluesval[indsval]
			eigenvaluescondsort = eigenvaluescond[indscond]
			lenv = len(eigenvaluesval)
			lenc = len(eigenvaluescond)
			code = lenv*1000+lenc
			energies.append(np.hstack((code,eigenvaluesvalsort,eigenvaluescondsort)))
			vmat = np.zeros((vmatrixlen))
			for indn in np.arange(lenv):
				indval = indsval[indn]
				for indm in np.arange(lenc):
					indcond = indscond[indm]
					wf_val = eigenvectors[:,maskval][:,indval]
					wf_cond = eigenvectors[:,maskcond][:,indcond]
					vmat[lenc*indn+indm] = np.abs(np.dot(np.conj(wf_val),np.matmul(v,wf_cond)))**2
			vmatrix.append(vmat)

	energies_save_to = dir_path+'/energies.txt'
	velocities_save_to = dir_path+'/velocities.txt'
	np.savetxt(energies_save_to,energies)
	np.savetxt(velocities_save_to,vmatrix)

def CleanEigensystem(kstep0, angle, N, dir_path):
	energies_save_to = dir_path+'/energies.txt'
	velocities_save_to = dir_path+'/velocities.txt'
	os.remove(energies_save_to)
	os.remove(velocities_save_to)

def TwistVelocity(N):
	return np.kron(np.identity(2*N**2),np.array([[0,1],[1,0]]))

def DOSPlot(kstep0, angle, N, dir_path):
	"""
	input: kstep0 is used to construct a grid in hexagon, Estep0 defines step with which histogram is calculated
	returns density of states as a function of energy normalized on its maximum value from the list
	"""
	theta = np.pi*angle/180
	k0 = 4/3*np.sin(0.5*theta)
	isteps = int(0.5*np.sqrt(3)*k0/kstep0)
	jsteps = int(k0/kstep0)
	gridi = np.arange(-isteps,isteps+1,1)
	gridj = np.arange(-jsteps,jsteps+1,1)
	energies_save_to = dir_path+'/energies.txt'
	loadedenergies = np.loadtxt(energies_save_to,dtype=float)
	energies = []
	for i in np.arange((2*isteps+1)*(2*jsteps+1)):
		ind_i = int(i//(2*jsteps+1))-isteps
		ind_j = int(i%(2*jsteps+1))-jsteps
		kx = kstep0*ind_i 
		ky = kstep0*ind_j 
		if -k0+np.abs(kx)/np.sqrt(3)>ky or k0-np.abs(kx)/np.sqrt(3)<ky:
			continue		
		code = loadedenergies[i][0]
		lenv = int(code//1000)
		lenc = int(code%1000)
		energies_k = loadedenergies[i][1:]
		energies.append(energies_k)
	flat_energies = [num for elem in energies for num in elem]

	from matplotlib import pyplot as plt
	save_to = dir_path+'/DoS.png'
	fig, ax = plt.subplots(figsize=(9,9))
	Estep0 = 0.01
	bins = int(1/Estep0) 
	plt.hist(flat_energies, bins=bins,density=True)
	ax.set_ylabel('DoS')
	ax.set_xlabel('E in units of t')
	plt.savefig(save_to)

def ReSigmaPlot(kstep0, angle, N, dir_path):
	theta = np.pi*angle/180
	k0 = 4/3*np.sin(0.5*theta)
	gamma = 0.01
	v = TwistVelocity(N)
	from matplotlib import pyplot as plt
	import matplotlib as mpl
	#omegamax = 0.5*k0/0.15 boundary energy before cones intercept
	#omegamax=0.07
	omegamax = 0.6
	omegas = np.arange(0.02,omegamax,0.02)
	sigma = np.zeros((len(omegas)))
	index = 0
	"""
	for omega in omegas:
		scaler = 0.15*angle/2 # coef defining sizes of calculation boxes for resigma
		kbox = scaler*omega
		isteps = int(kbox/kstep0)
		jsteps = int(kbox/kstep0)
		gridi = np.arange(-isteps,isteps+1,1)
		gridj = np.arange(-jsteps,jsteps+1,1)
		iv,jv = np.meshgrid(gridj,gridi)
		kxv = iv*kstep0 
		kyv = jv*kstep0
		ReSigma = np.zeros((2,2*isteps+1,2*jsteps+1))
		energycond = np.zeros((2*isteps+1,2*jsteps+1))
		energyval = np.zeros((2*isteps+1,2*jsteps+1))
		vmatrix = np.zeros((2*isteps+1,2*jsteps+1))
		if 1<angle<5: 
			transitions=6
		else:
			transitions=2
		for i in np.arange(-isteps,isteps+1,1):
			for j in np.arange(-jsteps,jsteps+1,1):
				for ind in np.arange(2):
					kx = kstep0*i + 0.5*(2*ind-1)*k0 # ind 0 is for K2, ind 1 is for K1 
					ky = kstep0*j 
					ham = TwistHamiltonian(kx,ky,angle,N,t_layers)
					eigenvalues, eigenvectors = eigsh(ham,k=levels,which='SM')
					maskval = eigenvalues<0
					eigenvaluesval = eigenvalues[maskval]
					maskcond = eigenvalues>0 
					eigenvaluescond = eigenvalues[maskcond]
					indsval = np.argsort(np.abs(eigenvaluesval))
					indscond = np.argsort(eigenvaluescond)
					for indn in np.arange(int(np.sqrt(transitions))):
						indval = indsval[indn]
						for indm in np.arange(int(np.sqrt(transitions))):
							indcond = indscond[indm]
							wf_val = eigenvectors[:,maskval][:,indval]
							en_val = eigenvaluesval[indval]
							wf_cond = eigenvectors[:,maskcond][:,indcond]
							en_cond = eigenvaluescond[indcond]
							vmat = np.abs(np.dot(np.conj(wf_val),np.matmul(v,wf_cond)))**2
							ReSigma[ind, i+isteps,j+jsteps] += 1/omega*1/((en_cond-en_val-omega)**2+gamma**2)*gamma*vmat

		cmap = 'RdGy'
		norm_sigma = np.max(np.abs(ReSigma))
		norm = mpl.colors.Normalize(vmin=0, vmax=norm_sigma)

		
		save_to = dir_path+'/ReSigmaK1K2_'+str(omega)+'.png'
		fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [8,8, 1]})
		ax[0].contourf(kxv-0.5*k0,kyv,ReSigma[0],cmap='RdGy')
		ax[1].contourf(kxv+0.5*k0,kyv,ReSigma[1],cmap='RdGy')
		cax = ax[2]
		plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,label='ReSigma around K1 and K2')
		plt.savefig(save_to)
		plt.close()

		print("ReSigma around K2 = ", 12*kstep0**2*np.sum(ReSigma[0]))
		print("ReSigma around K1 = ", 12*kstep0**2*np.sum(ReSigma[1]))
		sigma[index] = 12*kstep0**2*(np.sum(ReSigma[0])+np.sum(ReSigma[1]))
		if np.abs(sigma[index]-2)>0.2:
			break
		index+=1
	"""
	#omegamin = omega[index]
	#print("go to high energies")
	isteps = int(0.5*np.sqrt(3)*k0/kstep0)
	jsteps = int(k0/kstep0)
	gridi = np.arange(-isteps,isteps+1,1)
	gridj = np.arange(-jsteps,jsteps+1,1)
	iv,jv = np.meshgrid(gridj,gridi)
	kxv = iv*kstep0 
	kyv = jv*kstep0
	energies_save_to = dir_path+'/energies.txt'
	velocities_save_to = dir_path+'/velocities.txt'
	energies = np.loadtxt(energies_save_to,dtype=float)
	velocities = np.loadtxt(velocities_save_to,dtype=float)
	for omega in omegas[index:]:
		ReSigma = np.zeros((2*isteps+1,2*jsteps+1))
		for i in np.arange((2*isteps+1)*(2*jsteps+1)):
			code = energies[i][0]
			lenv = int(code//1000)
			lenc = int(code%1000)
			energies_val = energies[i][1:lenv+1]
			energies_cond = energies[i][lenv+1:]
			for indn in np.arange(lenv):
				for indm in np.arange(lenc):
					vmat = velocities[i][indn*lenc+indm]
					en_val = energies_val[indn]
					en_cond = energies_cond[indm]
					ind_i = int(i//(2*jsteps+1))
					ind_j = int(i%(2*jsteps+1))
					ReSigma[ind_i,ind_j] += 1/omega*1/((en_cond-en_val-omega)**2+gamma**2)*gamma*vmat

		cmap = 'RdGy'
		norm_sigma = np.max(np.abs(ReSigma))
		norm = mpl.colors.Normalize(vmin=0, vmax=norm_sigma)

		save_to = dir_path+'/ReSigma_'+str(index)+'_'+str(omega)[:5]+'.png'
		fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [8, 1]})
		ax[0].contourf(kxv,kyv,ReSigma,cmap='RdGy')
		cax = ax[1]
		plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,label='ReSigma in MBZ')
		plt.savefig(save_to)
		plt.close()

		print("ReSigma in MBZ = ", 12*kstep0**2*np.sum(ReSigma))
		sigma[index] = 12*kstep0**2*np.sum(ReSigma)
		index+=1

	sigmatofile = np.column_stack([omegas, sigma])
	sigma_save_to = dir_path+'/sigma.txt'
	np.savetxt(sigma_save_to, sigmatofile)

	save_to = dir_path+'/ReSigmaEnergy.png'
	fig, ax = plt.subplots()
	ax.set_xlabel('omega in units of t')
	ax.set_ylabel(' Re $ \sigma_{xx}/\sigma_{mono}$')
	ax.scatter(omegas, sigma, alpha=0.5, marker=r'$\clubsuit$',label='$\sigma_{xx}$')
	plt.legend()
	plt.savefig(save_to)
	
if __name__ == "__main__":
	# config of calculations
	for angle in np.arange(1,3,1):
		N = 6
		t_layers = 0.036
		kstep0=0.001*angle/10
		startTime = time.time()
		dir_path = '../set16/tbg_angle'+str(angle)+'_N'+str(N)+'_tlayers'+str(t_layers)+'_kstep0'+str(kstep0)
		mkdir_p(dir_path)
		WriteEigensystem(kstep0, angle, N, dir_path)
		DOSPlot(kstep0, angle, N, dir_path)
		ReSigmaPlot(kstep0, angle, N, dir_path)
		CleanEigensystem(kstep0, angle, N, dir_path)
		executionTime = (time.time() - startTime)
		print('angle = ', angle, 'Execution time in seconds: ' + str(executionTime))





# plot preliminary: contour plots of 6 lowest energy contourplots
# dispersion curves for two different paths through two dirac cones
# contour matrix element of transitions: 5 contourplots for [..., valence-1] [valence, ...]
# builds plots for valence-1--valence, valence-1--valence+1, valence-2--valence+1, valence-2--valence

# build resigma
# plot 4 contourplots for each omega: with each coloring for each sector and then full hexagon
# adds text to each contourplot: sum of contributions
# colect resigma(omega) to file

