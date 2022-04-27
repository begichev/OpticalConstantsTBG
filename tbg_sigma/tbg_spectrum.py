import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy as sps
import csv

omega  = 110.7          #mev
d      = 1.420          #angstrom, whatever is ok.
hv     = 1.5*d*2970     #meV*angstrom, Fermi velocity for SLG
N      = 10              #truncate range
valley = 1             #+1 for K, -1 for K'
KDens  = 100            #density of k points, 100 is good.

#tune parameters

I      = complex(0, 1)
ei120  = np.cos(2*np.pi/3) + valley*I*np.sin(2*np.pi/3)
ei240  = np.cos(2*np.pi/3) - valley*I*np.sin(2*np.pi/3)

Tqb    = omega*np.array([[1,1], [1,1]], dtype=complex)
Tqtr   = omega*np.array([[1, ei120], [ei240, 1]], dtype=complex)
Tqtl   = omega*np.array([[1, ei240], [ei120, 1]], dtype=complex)
TqbD   = np.array(np.matrix(Tqb).H)
TqtrD  = np.array(np.matrix(Tqtr).H)
TqtlD  = np.array(np.matrix(Tqtl).H)

#define Lattice
L = []
invL = np.zeros((2*N+1, 2*N+1), int)

def Lattice(n):
    count = 0
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])
            invL[i+n, j+n] = count
            count = count + 1
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])

Lattice(N)
siteN = (2*N+1)*(2*N+1)
L = np.array(L)

levels = 25

def Hamiltonian(kx, ky, theta):
    
    theta  = theta/180.0*np.pi 
    b1m    = 8*np.pi*np.sin(theta/2)/3/d*np.array([0.5, -np.sqrt(3)/2])
    b2m    = 8*np.pi*np.sin(theta/2)/3/d*np.array([0.5, np.sqrt(3)/2])
    qb     = 8*np.pi*np.sin(theta/2)/3/np.sqrt(3)/d*np.array([0, -1])
    K1     = 8*np.pi*np.sin(theta/2)/3/np.sqrt(3)/d*np.array([-np.sqrt(3)/2,-0.5])
    K2     = 8*np.pi*np.sin(theta/2)/3/np.sqrt(3)/d*np.array([-np.sqrt(3)/2,0.5])
    H = np.array(np.zeros((4*siteN, 4*siteN)), dtype=complex)
    
    for i in np.arange(siteN):
        #diagonal term
        ix = L[i, 0]
        iy = L[i, 1]
        ax = kx - valley*K1[0] + ix*b1m[0] + iy*b2m[0]
        ay = ky - valley*K1[1] + ix*b1m[1] + iy*b2m[1]

        qx = np.cos(theta/2) * ax + np.sin(theta/2) * ay
        qy =-np.sin(theta/2) * ax + np.cos(theta/2) * ay
         
        H[2*i, 2*i+1] = hv * (valley*qx - I*qy)
        H[2*i+1, 2*i] = hv * (valley*qx + I*qy)

        #off-diagonal term
        j = i + siteN
        H[2*j, 2*i]     = TqbD[0, 0]
        H[2*j, 2*i+1]   = TqbD[0, 1]
        H[2*j+1, 2*i]   = TqbD[1, 0]
        H[2*j+1, 2*i+1] = TqbD[1, 1]
        if (iy != valley*N and ix != valley*N):
            j = invL[ix+1+N, iy+valley*1+N] + siteN
            H[2*j, 2*i]     = TqtrD[0, 0]
            H[2*j, 2*i+1]   = TqtrD[0, 1]
            H[2*j+1, 2*i]   = TqtrD[1, 0]
            H[2*j+1, 2*i+1] = TqtrD[1, 1]
        if (ix != valley*N):
            j = invL[ix+valley*1+N, iy+N] + siteN
            H[2*j, 2*i]     = TqtlD[0, 0]
            H[2*j, 2*i+1]   = TqtlD[0, 1]
            H[2*j+1, 2*i]   = TqtlD[1, 0]
            H[2*j+1, 2*i+1] = TqtlD[1, 1]
        

    for i in np.arange(siteN, 2*siteN):
        #diagnoal term
        j = i - siteN
        ix = L[j, 0]
        iy = L[j, 1]
        ax = kx  - valley*K2[0] + ix*b1m[0] + iy*b2m[0] 
        ay = ky  - valley*K2[1] + ix*b1m[1] + iy*b2m[1]

        qx = np.cos(theta/2) * ax - np.sin(theta/2) * ay
        qy = np.sin(theta/2) * ax + np.cos(theta/2) * ay

        H[2*i, 2*i+1] = hv * (valley*qx - I*qy)
        H[2*i+1, 2*i] = hv * (valley*qx + I*qy)

        #off-diagonal term
        H[2*j, 2*i]     = Tqb[0, 0]
        H[2*j, 2*i+1]   = Tqb[0, 1]
        H[2*j+1, 2*i]   = Tqb[1, 0]
        H[2*j+1, 2*i+1] = Tqb[1, 1]
        if (iy != (-valley*N) and ix != (-valley*N)):
            j = invL[ix-1+N, iy-valley*1+N]
            H[2*j, 2*i]     = Tqtr[0, 0]
            H[2*j, 2*i+1]   = Tqtr[0, 1]
            H[2*j+1, 2*i]   = Tqtr[1, 0]
            H[2*j+1, 2*i+1] = Tqtr[1, 1]
        if (ix != -valley*N):
            j = invL[ix-valley*1+N, iy+N]
            H[2*j, 2*i]     = Tqtl[0, 0]
            H[2*j, 2*i+1]   = Tqtl[0, 1]
            H[2*j+1, 2*i]   = Tqtl[1, 0]
            H[2*j+1, 2*i+1] = Tqtl[1, 1]


    eigensystem = sps.sparse.linalg.eigsh(H, k = levels, which = "SM")
    
    return eigensystem

angle = 3
theta = 3*np.pi/180


eig_vec = Hamiltonian(0,0,angle)
sorted_eig_ens = np.sort(eig_vec[0])
sorted_eig_vec = []
argsort = eig_vec[0].argsort(axis=None)
for i in np.arange(levels):
    sorted_eig_vec.append(eig_vec[1][:,np.where(argsort==i)])
sorted_eig_vec = np.array(sorted_eig_vec)
sorted_eig_vec = np.squeeze(sorted_eig_vec)

energies_save_to='tbg'+str(angle)+'levels'+str(levels)+'N'+str(N)+'_energies.txt'
wavefunctions_save_to='tbg+str(angle)+'levels'+str(levels)+'N'+str(N)'+'_wavefunctions.txt'
np.savetxt(energies_save_to, sorted_eig_ens)
np.savetxt(wavefunctions_save_to,sorted_eig_vec)