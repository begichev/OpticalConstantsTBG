def PVintegral(sigma,omega,cutoff):
	integral = 0
	#steps = int((cutoff-omega0)/domega)
	for k in np.arange(1,steps,1):
		nuk = omega0 + k*domega
		integral += nuk/(omega**2-nuk**2)*sigma[k,1]*domega
	return integral

def ImSigmaOmega(omega, sigma, cutoff):
	imsigma =  2/(np.pi*omega)*PVintegral(sigma,omega,cutoff)+2/(np.pi*omega)*2*cutoff+2/(np.pi)*np.log((cutoff-omega)/(cutoff+omega))
	return imsigma

def ImSigma(sigma, cutoff):
	listsigma = [ImSigmaOmega(omega,sigma,cutoff) for omega in np.arange(omega0,omega0+len(sigma)*domega,domega)]
	print(listsigma)
	np.loadtxt('imsigma.txt', listsigma)
	return np.array(listsigma) 

def Extract_nk(sigma,imsigma):
	"""
	input freq in eV
	"""
	alpha = 1/137
	c_over_d = 30000/0.335*0.0041
	complexsigma = [complex(float(a), float(b)) for a, b in zip(sigma, imsigma)]
	#complex_n = np.sqrt(1+alpha*np.pi*c_over_d*complexsigma/omega)
	steps = int((cutoff-omega0)/domega)
	complex_n = np.array(steps,dtype=complex)
	steps = int((cutoff-omega0)/domega)
	for k in np.arange(steps):
		omega = omega0 + k*domega
		complex_n[k] = np.sqrt(1+alpha*np.pi*c_over_d*complexsigma[k]/omega)
	n = np.column_stack([np.arange(omega0,cutoff,domega), np.real(complex_n)])
	k = np.column_stack([np.arange(omega0,cutoff,domega), np.imag(complex_n)])
	n_save_to = 'n.txt'
	k_save_to = 'k.txt'
	np.savetxt(n_save_to, n)
	np.savetxt(k_save_to, k)
	return [np.real(complex_n), np.imag(complex_n)]