import numpy as np
from scipy.linalg import block_diag

t = 2.7
    
def occup_0K(vals, spin=1):
    """
    for graphene system, one pz orbital contribute one electron
    """
  
    if vals[0]==0:
        result = np.where(vals==0, 1, vals)
        return result
    else:
        result1 = np.where(vals>0, 0, vals)
        result2 = np.where(result1<0, 2 ,result1)
        return result2

def pick_up_transition_pairs(vals, omega, e_win, occup):
    omega_minus = max(omega-e_win, 1.e-6)
    omega_plus = omega + e_win
    inds_ef = np.intersect1d(np.where(occup>0)[0], np.where(occup<2)[0])
    if len(inds_ef):
        ind_vbm = max(inds_ef)
        ind_cbm = min(inds_ef)
    else:
        ind_vbm = max(np.where(occup==2)[0])
        ind_cbm = min(np.where(occup==0)[0])

    vbm = vals[ind_vbm]
    cbm = vals[ind_cbm]
    
    e_bott = cbm - omega - e_win
    e_top = vbm + omega + e_win
    inds_shot = np.intersect1d(np.where(vals>=e_bott), np.where(vals<=e_top))

    if len(inds_shot):
        inds_vb = np.arange(ind_vbm, inds_shot[0]-1, -1)
        inds_cb = np.arange(ind_cbm, inds_shot[-1]+1)
    else:
        inds_vb = []
        inds_cb = []

    def add_pair(ind_vb):
        e0 = vals[ind_vb]
        des = vals - e0
        inds_chosen = np.intersect1d(np.where(des>=omega_minus)[0], np.where(des<=omega_plus)[0])
        inds_chosen = np.intersect1d(inds_chosen, inds_cb)
        pairs_chosen = [[ind_vb, indi] for indi in inds_chosen]
        return pairs_chosen
    pairs = [add_pair(ind_vb) for ind_vb in inds_vb]
#    pairs = [add_pair(ind_vbm)]
    pairs = [i for i in pairs if len(i)]
    if len(pairs):
        return np.concatenate(pairs)
    else:
        return []

def Re_optical_conductivity(vals, vecs, omegas, gamma=0.002*t, e_win=0.05*t,kstep0=0.001, steps=100):
    """
    inputs:
        gamma: the energy width for Lorentzian function, which is used for simulate delta function
        e_win: the energy window to pick up energy level pairs for calculating optical transition, which means that for hbar*omega 
               two energy levels with hbar*omega -e_win <= deltaE <= hbar*omega+e_win are picked up for calculating
               the optical conductivity.
        omega_lim: the frequency range in units of eV (measured as hbar*omega)
    """
    size = 4*steps**2
    v = dimensionless_v(0,0,steps)
    def calc_sigma_mn_pair(indm, indn, omega, occup, v):
        vecm = vecs[indm*size:indm*size+size]
        vecn = vecs[indn*size:indn*size+size]
        vmn = np.dot(vecm.conj(), np.matmul(v, vecn))
        fn = occup[indn]
        fm = occup[indm]
        de = vals[indn] - vals[indm]
        denominator = (omega-de)**2 + gamma**2
        denominator_ = (omega+de)**2 + gamma**2
        return np.linalg.norm(vmn)**2*( (fm-fn)/denominator + (fn-fm)/denominator_)*t**2*gamma/omega

    def calc_sigma_one_point(omega):
        sigmas_mn = 0
        energies = np.array(vals)
        occup = occup_0K(energies)
        pairs = pick_up_transition_pairs(energies, omega, e_win, occup)
        sigmas_mn += np.sum([calc_sigma_mn_pair(int(pair[0]), int(pair[1]), omega, occup, v) for pair in pairs])
        return 6*np.sqrt(3)*kstep0**2/np.pi*np.sum(sigmas_mn)
    
    sigmas = [calc_sigma_one_point(omega) for omega in omegas]
    return np.array(sigmas)


def dimensionless_v(kx, ky, steps):
    """
    the matrix of v. made of dimensionless k
    """
    sigma = np.array([[0,1],[1,0]])
    return np.kron(np.identity(2*steps**2),sigma)

def calc_optical_conductivity(omegas=np.arange(0.1, 4, 0.1), gamma=0.02*t, e_win=0.02*t, save_to='sigma.txt'):
    levels = 200
    angle=3.265
    gamma = 0.002
    theta  = angle/180.0*np.pi 
    kstep0 = 4/np.sqrt(3)*np.sin(0.5*theta)
    steps = int(0.5/(np.sqrt(3)*np.sin(0.5*theta)))
    energies_load_from='tbg'+str(angle)+'levels_'+str(levels)+'_energies.txt'
    wavefunctions_load_from='tbg'+str(angle)+'levels_'+str(levels)+'_wavefunctions.txt'
    sigma_save_to = 'tbg'+str(angle)+'levels_'+str(levels)+'_sigma.txt'
    vals = np.loadtxt(energies_load_from, dtype=float)
    vecs = np.loadtxt(wavefunctions_load_from, dtype=complex)
    sigma_x = Re_optical_conductivity(vals, vecs, omegas, gamma=gamma*t, e_win=e_win, kstep0=kstep0, steps=steps) 
    sigma = np.column_stack([omegas, sigma_x])
    np.savetxt(sigma_save_to, sigma)
    return omegas, sigma_x

def plot_optical_cond(sigma_f='sigma.txt', save_to='optical_cond.pdf'):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    sigma = np.loadtxt(sigma_f)
    ax.set_xlabel('Energy, eVs')
    ax.set_ylabel(' Re $ \sigma_{xx}/\sigma_{mono}$')
    ax.scatter(sigma[:,0], sigma[:,1], alpha=0.5, marker=r'$\clubsuit$',label='$\sigma_{xx}$')
    plt.legend()
    plt.savefig(save_to)


