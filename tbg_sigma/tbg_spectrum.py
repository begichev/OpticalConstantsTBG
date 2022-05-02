import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy as sps

omega  = 0.110        # eV
d      = 1.420         
t = 2.7 # hopping for monolayer
hv     =  0.5*np.sqrt(3)*2.46*t     #eV*angstrom, Fermi velocity for SLG
#Nm      = 5              #truncate range in moire zone
valley = 1             #+1 for K, -1 for K'
I      = complex(0, 1)
ei120  = np.cos(2*np.pi/3) + valley*I*np.sin(2*np.pi/3)
ei240  = np.cos(2*np.pi/3) - valley*I*np.sin(2*np.pi/3)
Tqb    = omega*np.array([[1,1], [1,1]], dtype=complex)
Tqtr   = omega*np.array([[1, ei120], [ei240, 1]], dtype=complex)
Tqtl   = omega*np.array([[1, ei240], [ei120, 1]], dtype=complex)
TqbD   = np.array(np.matrix(Tqb).H)
TqtrD  = np.array(np.matrix(Tqtr).H)
TqtlD  = np.array(np.matrix(Tqtl).H)



levels = 200
a = 2.46 
k = 2*np.pi/a

def Hamiltonian(kx, ky, theta, scale):
    def Lattice(steps):
        count = 0
        for i in np.arange(steps):
            for j in np.arange(steps):
                L.append([i, j])
                invL[i, j] = count
                count = count + 1
        for i in np.arange(steps):
            for j in np.arange(steps):
                L.append([i, j])
    theta  = theta/180.0*np.pi 
    steps = int(0.5/(np.sqrt(3)*np.sin(0.5*theta)))
    L = []
    invL = np.zeros((scale*steps, scale*steps), int)
    Lattice(scale*steps)
    siteN = (scale*steps)**2
    L = np.array(L)
    b1m    = 4/np.sqrt(3)*np.sin(theta/2)*np.array([0.5, -np.sqrt(3)/2])
    b2m    = 4/np.sqrt(3)*np.sin(theta/2)*np.array([0.5, -np.sqrt(3)/2])
    K1     = 2/3*np.sin(theta/2)*np.array([0,-1])
    K2     = 2/3*np.sin(theta/2)*np.array([0, 1])
    H = np.array(np.zeros((4*siteN, 4*siteN)), dtype=complex)
    for i in np.arange(siteN):
        #diagonal term
        ix = L[i, 0]
        iy = L[i, 1]
        ax = kx - valley*K1[0] + (ix*b1m[0] + iy*b2m[0])/scale
        ay = ky - valley*K1[1] + (ix*b1m[1] + iy*b2m[1])/scale

        qx = np.cos(theta/2) * ax + np.sin(theta/2) * ay
        qy =-np.sin(theta/2) * ax + np.cos(theta/2) * ay
         
        H[2*i, 2*i+1] = hv * (valley*qx - I*qy)
        H[2*i+1, 2*i] = hv * (valley*qx + I*qy)

        #off-diagonal term
        if ix%scale==0 and iy%scale==0:
            j = i + siteN
            H[2*j, 2*i]     = TqbD[0, 0]
            H[2*j, 2*i+1]   = TqbD[0, 1]
            H[2*j+1, 2*i]   = TqbD[1, 0]
            H[2*j+1, 2*i+1] = TqbD[1, 1]
            if (iy != scale*valley*(steps-1) and ix != scale*valley*(steps-1)):
                j = invL[ix+scale*valley*1, iy+scale*valley*1] + siteN
                H[2*j, 2*i]     = TqtrD[0, 0]
                H[2*j, 2*i+1]   = TqtrD[0, 1]
                H[2*j+1, 2*i]   = TqtrD[1, 0]
                H[2*j+1, 2*i+1] = TqtrD[1, 1]
            if (ix != scale*valley*(steps-1)):
                j = invL[ix+scale*valley*1, iy] + siteN
                H[2*j, 2*i]     = TqtlD[0, 0]
                H[2*j, 2*i+1]   = TqtlD[0, 1]
                H[2*j+1, 2*i]   = TqtlD[1, 0]
                H[2*j+1, 2*i+1] = TqtlD[1, 1]

    for i in np.arange(siteN, 2*siteN):
        #diagnoal term
        j = i - siteN
        ix = L[j, 0]
        iy = L[j, 1]
        ax = kx  - valley*K2[0] + (ix*b1m[0] + iy*b2m[0])/scale
        ay = ky  - valley*K2[1] + (ix*b1m[1] + iy*b2m[1])/scale

        qx = np.cos(theta/2) * ax - np.sin(theta/2) * ay
        qy = np.sin(theta/2) * ax + np.cos(theta/2) * ay

        H[2*i, 2*i+1] = hv * (valley*qx - I*qy)
        H[2*i+1, 2*i] = hv * (valley*qx + I*qy)

        #off-diagonal term
        if (iy != scale*valley*(steps-1) and ix != scale*valley*(steps-1)):
            H[2*j, 2*i]     = Tqb[0, 0]
            H[2*j, 2*i+1]   = Tqb[0, 1]
            H[2*j+1, 2*i]   = Tqb[1, 0]
            H[2*j+1, 2*i+1] = Tqb[1, 1]
            if (iy != (-scale*valley*(steps-1)) and ix != (-scale*valley*(steps-1))):
                j = invL[ix-scale*valley*1, iy-scale*valley*1]
                H[2*j, 2*i]     = Tqtr[0, 0]
                H[2*j, 2*i+1]   = Tqtr[0, 1]
                H[2*j+1, 2*i]   = Tqtr[1, 0]
                H[2*j+1, 2*i+1] = Tqtr[1, 1]
            if (ix != -scale*valley*(steps-1)):
                j = invL[ix-scale*valley*1, iy]
                H[2*j, 2*i]     = Tqtl[0, 0]
                H[2*j, 2*i+1]   = Tqtl[0, 1]
                H[2*j+1, 2*i]   = Tqtl[1, 0]
                H[2*j+1, 2*i+1] = Tqtl[1, 1]


    eigensystem = sps.sparse.linalg.eigsh(H, k = levels, which = "SM")
    
    return eigensystem

angle = 3.265
scale = 10
a = 2.46 # angstrom

"""
k = 2*np.pi/a*4/np.sqrt(3)*np.sin(angle*np.pi/180/2) # inverse angstrom
kstep0 = 0.002
kstep = kstep0*k 
eigenenergies = []
eigenfunctions = []
N = 20

for i in np.arange(-N,N+1): # -N ... 0 ... N
    for j in np.arange(-N,N+1):
        ham_theta = Hamiltonian(i*kstep, j*kstep,angle)
        eig_ens = ham_theta[0]
        eigenenergies.append(eig_ens)
        for i in np.arange(levels):
            eigenfunctions.extend(ham_theta[1][:,i])
"""

eigensystem = Hamiltonian(0,0, angle,scale)
eigenenergies = eigensystem[0]
eigenfunctions = []
for i in np.arange(levels):
    eigenfunctions.extend(np.array(eigensystem[1][:, i]))
real_eigenenergies = np.real(eigenenergies)

energies_save_to='tbg'+str(angle)+'scale'+str(scale)+'levels_'+str(levels)+'_energies.txt'
wavefunctions_save_to='tbg'+str(angle)+'scale'+str(scale)+'levels_'+str(levels)+'_wavefunctions.txt'

np.savetxt(energies_save_to, real_eigenenergies)
np.savetxt(wavefunctions_save_to,eigenfunctions)