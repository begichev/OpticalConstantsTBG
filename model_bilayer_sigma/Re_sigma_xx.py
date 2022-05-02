import numpy as np
from scipy.linalg import block_diag

t = 2.7
t_layers = 0.4

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
    pairs = [i for i in pairs if len(i)]
    if len(pairs):
        return np.concatenate(pairs)
    else:
        return []

def Re_optical_conductivity(vals, vecs, omegas, gamma=0.002*t, e_win=0.05*t, kstep0=0.001, steps=100):
    """
    inputs:
        gamma: the energy width for Lorentzian function, which is used for simulate delta function
        e_win: the energy window to pick up energy level pairs for calculating optical transition, which means that for hbar*omega 
               two energy levels with hbar*omega -e_win <= deltaE <= hbar*omega+e_win are picked up for calculating
               the optical conductivity.
        omega_lim: the frequency range in units of eV (measured as hbar*omega)
    """
    def calc_sigma_mn_pair(i, indm, indn, omega, occup, v):
        vecm = vecs[i*4+indm]
        vecn = vecs[i*4+indn]
        vmn = np.dot(vecm.conj(), np.matmul(v, vecn))
        fn = occup[indn]
        fm = occup[indm]
        de = vals[i][indn] - vals[i][indm]
        denominator = (omega-de)**2 + gamma**2
        denominator_ = (omega+de)**2 + gamma**2
        return np.linalg.norm(vmn)**2*( (fm-fn)/denominator + (fn-fm)/denominator_)*t**2*gamma/omega

    def calc_sigma_one_point(omega):
        sigmas_mn = 0
        kstepix = -1*kstep0
        kstepiy = 0
        kstepjx = 0.5*kstep0
        kstepjy = -0.5*np.sqrt(3)*kstep0
        for i in np.arange(steps**2):
                energies = np.array(vals[i])
                occup = np.array([2,2,0,0])
                Kx = 1/np.sqrt(3)*2*np.pi
                Ky = 1/3*2*np.pi # Dirac point coordinates
                kx = 2*np.pi*((i//steps)*kstepix + (i%steps)*kstepjx) + Kx # dimensionless k
                ky = 2*np.pi*((i//steps)*kstepiy + (i%steps)*kstepjy) + Ky
                v = dimensionless_v(kx, ky)
                pairs = pick_up_transition_pairs(energies, omega, e_win, occup)   
                if len(pairs):
                    sigmas_mn += np.sum([calc_sigma_mn_pair(i, int(pair[0]), int(pair[1]), omega, occup, v) for pair in pairs])
        return 6*np.sqrt(3)*kstep0**2/np.pi*np.sum(sigmas_mn)
    
    sigmas = [calc_sigma_one_point(omega) for omega in omegas]
    return np.array(sigmas)

def dimensionless_v(kx, ky):
    """
    the matrix of v. made of dimensionless k
    """
    
    v = 1.j*(np.exp(0.5*1.j*kx/np.sqrt(3))*np.cos(0.5*ky)-np.exp(-1.j*kx/np.sqrt(3)))
    block = np.zeros((2,2), dtype='complex')
    block[0][0] = block[1][1] = 0
    block[0][1] = v
    block[1][0] = np.conj(v)
    block2 = np.conj(block)
    return block_diag(block, block2)

def calc_optical_conductivity(omegas=np.arange(0.05, 1.5, 0.05), gamma=0.002*t, e_win=0.02*t, save_to='ab_sigma.txt'):
    kstep0 = 0.001
    gamma = 0.02
    steps = int(2/3/kstep0)
    energies_load_from='ab_bilayer_energies_'+str(kstep0)+'.txt'
    wavefunctions_load_from='ab_bilayer_wavefunctions_'+str(kstep0)+'.txt'
    sigma_save_to = 'ab_bilayer_sigma'+str(kstep0)+'_gamma'+str(gamma)+'.txt'
    vals = np.loadtxt(energies_load_from, dtype=float)
    vecs = np.loadtxt(wavefunctions_load_from, dtype=complex)
    sigma_x = Re_optical_conductivity(vals, vecs, omegas, gamma=gamma*t_layers, e_win=e_win, kstep0=kstep0, steps=steps) 
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


