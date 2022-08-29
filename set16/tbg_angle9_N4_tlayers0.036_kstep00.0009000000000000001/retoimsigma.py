import numpy as np 

sigma = np.loadtxt('sigma.txt')
omega0 = 0.02
domega = 0.02
#cutoff = 2 # eV
#cutoff = 0.58 # in t
cutoff = 1.2 # in t 
#steps = 28 # until sigma=2 
steps = int((cutoff-omega0)/domega)
droppedsteps = 3 # number of final resigma freq points which are not used for Imsigma calc

#print(sigma[:,1]) 

def Weight(n,k,omega0):
	omega_k = omega0 + k*domega
	nu_n = omega0 + n*domega
	return 0.5/omega_k*((nu_n+0.5*domega)**2/((k-n-0.5)*(2*omega0+(n+k+0.5)*domega))+(nu_n-0.5*domega)**2/((k-n+0.5)*(2*omega0+(n+k-0.5)*domega)))

def Integral(k,omega0):
	sum = 0
	nmax = steps-droppedsteps
	Resigma = sigma[k,1]
	for i in np.arange(nmax):
		weight = Weight(i,k,omega0)
		sum += weight*Resigma
	return sum

#print(Integral(4,omega0))

def ImSigma(k,omega0,cutoff):
	omega_k = omega0 + k*domega
	integral = Integral(k,omega0)
	return 2/np.pi*(integral+2*cutoff/omega_k+np.log((cutoff-omega_k)/(cutoff+omega_k)))

#print(ImSigma(4,omega0,cutoff))

def FullSigma(omega0,cutoff):
	nmax = steps-droppedsteps
	Imsigma = np.zeros((nmax))
	for k in np.arange(nmax):
		Imsigma[k] = ImSigma(k,omega0,cutoff)
	Resigma = sigma[:nmax,1]
	fullsigma = np.column_stack([Resigma,Imsigma])
	return fullsigma

#print(FullSigma(omega0,cutoff))
#np.savetxt('fullsigma.txt',FullSigma(omega0,cutoff))

def OptConsts(omega0,cutoff):
	nmax = steps-droppedsteps
	fullsigma = FullSigma(omega0,cutoff)
	n = np.zeros((nmax))
	k = np.zeros((nmax))
	for m in np.arange(nmax):
		omega_m = omega0 + m*domega
		alpha_m = 2.2/omega_m
		resigma = fullsigma[m,0]
		imsigma = fullsigma[m,1]
		discr = 1+2*alpha_m*resigma+alpha_m**2*(resigma**2+imsigma**2)
		k[m] = np.sqrt(0.5*(np.sqrt(discr)-(1+alpha_m*resigma)))
		n[m] = 0.5*alpha_m*imsigma/k[m]
	optconsts = np.column_stack([n,k])
	return optconsts

#print(OptConsts(omega0,cutoff))

def plot_nk(omega0,cutoff,save_to='optconsts.pdf'):
	#fullsigma = FullSigma(omega0,cutoff)
	#nsquared = fullsigma[:,0]+1.j*fullsigma[:,1]
	bound = 5
	optconsts = OptConsts(omega0,cutoff)
	nmax = steps-droppedsteps
	omegas = np.array([omega0+k*domega for k in np.arange(bound,nmax,1)])
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [8, 8]})
	#n = np.loadtxt('n.txt')
	#k = np.loadtxt('k.txt')
	ax[0].set_xlabel('Energy in t')
	ax[1].set_xlabel('Energy in t')

	#ax.set_ylabel('n, k')
	#ax.scatter(omegas, optconsts[bound:,0], alpha=0.5, marker=r'$\clubsuit$',label='$\sigma_{xx}$')
	ax[0].scatter(omegas, optconsts[bound:,0], alpha=0.5, marker=r'$\clubsuit$',label='n')
	ax[1].scatter(omegas, optconsts[bound:,1], alpha=0.5, marker=r'$\clubsuit$',label='k')
	ax[0].legend()
	ax[1].legend()
	# 0.48 0.96
	ax[0].axvline(x=0.48,color='green', lw=2, alpha=0.7)
	ax[0].axvline(x=0.96,color='green', lw=2, alpha=0.7)
	ax[1].axvline(x=0.48,color='green', lw=2, alpha=0.7)
	ax[1].axvline(x=0.96,color='green', lw=2, alpha=0.7)

	#ax[0].fill_between(omegas,0,1,color='green', alpha=0.5, transform=ax.get_xaxis_transform())
	plt.savefig(save_to)

plot_nk(omega0,cutoff,save_to='optconsts.pdf')
