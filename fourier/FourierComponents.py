# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np

a = 2.46  # angstrom
theta = 10.00 * np.pi / 180  # degree to rad

# primitive vectors of SLG by Moon Koshino (2013)
lat_vec1 = a * np.array([1, 0])
lat_vec2 = a * np.array([1 / 2, np.sqrt(3) / 2])


def NeededSites(theta, M):
    # describes what N should be taken to have at least M moire patterns on the grid

    return 0.5 * (np.sqrt(M) / (2 * np.sin(theta / 2)) - 1)


M = 4
N = int(NeededSites(theta, M))  # N lat_vec to each side from the origin
Next = int(1.5*NeededSites(theta, M))
# 1.5 is the correction to avoid atoms without interlayer neighbours in (-N,N) ranges

def Rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


lat_vec_bottom1 = np.matmul(Rotation(-theta / 2), lat_vec1)
lat_vec_bottom2 = np.matmul(Rotation(-theta / 2), lat_vec2)
lat_vec_top1 = np.matmul(Rotation(theta / 2), lat_vec1)
lat_vec_top2 = np.matmul(Rotation(theta / 2), lat_vec2)

d0 = 3.35  # angstrom
a0 = 1.42  # angstrom
tau = a0 * np.array([0, 1])  # shift from A to B


def DistBottomTop(bottom_atom, sublattice_bottom, top_atom, sublattice_top):
    # bottom_atom and top_atom are indices [i,j] and [n,k] of lattice elements
    # sublattice =0 stands for A, =1 stands for B
    # returns distance between given atoms

    vec_bottom_atom = bottom_atom[0] * lat_vec_bottom1 + bottom_atom[1] * lat_vec_bottom2
    vec_top_atom = top_atom[0] * lat_vec_top1 + top_atom[1] * lat_vec_top2
    if sublattice_bottom == sublattice_top:
        return np.sqrt(np.linalg.norm(vec_bottom_atom - vec_top_atom) ** 2 + d0 ** 2)
    elif sublattice_bottom == 0:
        return np.sqrt(np.linalg.norm(vec_bottom_atom - vec_top_atom - tau) ** 2 + d0 ** 2)
    elif sublattice_bottom == 1:
        return np.sqrt(np.linalg.norm(vec_bottom_atom - vec_top_atom + tau) ** 2 + d0 ** 2)


# print(DistBottomTop([0,0],1,[0,5],1))


def ExtractFirsts(lst):
    return [item[0] for item in lst]


def ClosestAtomFromTop(bottom_atom, sublattice_bottom, sublattice_top):
    # finds [n,k] of atom from top layer within given sublattice
    # which is the closest to given [i,j] atom from sublattice_bottom from the bottom
    # returns distance to closest atom, its coordinates and vector in plane which connects bottom atom to top atom

    vec_bottom_atom = bottom_atom[0] * lat_vec_bottom1 + bottom_atom[1] * lat_vec_bottom2 + 0.5 * (
                2 * sublattice_bottom - 1) * tau
    dists = []
    for n in np.arange(-Next, Next + 1):
        for k in np.arange(-Next, Next + 1):
            vec_top_atom = n * lat_vec_top1 + k * lat_vec_top2 + 0.5 * (2 * sublattice_top - 1) * tau
            dists.append([DistBottomTop(vec_bottom_atom, sublattice_bottom, vec_top_atom, sublattice_top), [n, k]])
    distances = ExtractFirsts(dists)
    index = np.argmin(np.array(distances))
    vec_top_atom = dists[index][1][0] * lat_vec_top1 + dists[index][1][1] * lat_vec_top2 + 0.5 * (
                2 * sublattice_top - 1) * tau
    delta = vec_top_atom - vec_bottom_atom
    return [dists[index], delta]


# print(ClosestAtomFromTop([4,2],0,0))

Vppsigma = 0.48  # eV
Vpppi = -2.7  # eV
delta0 = 0.184 * a  # angstrom
h0 = 3.349  # angstrom
rc = 6.14  # angstrom
lc = 0.265  # angstrom


def Hopping(vec, sublattice_bottom, sublattice_top):
    # vec is a vector connecting atoms from bottom and top layer in 2D (top minus bottom)
    # returns value -t(d) following Moon Koshino (2013)

    vec = vec + (sublattice_top - sublattice_bottom) * tau
    full_dist = np.sqrt(np.linalg.norm(vec) ** 2 + d0 ** 2)
    Vpi = Vpppi * np.exp(-(full_dist - a0) / delta0)
    Vsigma = Vppsigma * np.exp(-(full_dist - d0) / delta0)
    return Vpi * (1 - (d0 / full_dist) ** 2) + Vsigma * (d0 / full_dist) ** 2


# Hopping([-0,0],0,0)

G1M = 8 * np.pi / (np.sqrt(3) * a) * np.sin(0.5 * theta) * np.array([1, 0])
G2M = 8 * np.pi / (np.sqrt(3) * a) * np.sin(0.5 * theta) * np.array([-0.5, 0.5 * np.sqrt(3)])

K = 2 * np.pi / a * np.array([-2 / 3, 0])


def HamiltForK(bottom_atom, sublattice_bottom, sublattice_top):
    # bottom atom is an array [i,j]
    # tight-binding model with hoppings to 9 closest neighbours
    # returns value of hamiltonian matrix element between bloch states
    # with given wave number K from Dirac point

    top_atom = ClosestAtomFromTop(bottom_atom, sublattice_bottom, sublattice_top)
    delta = top_atom[1]
    ham = 0
    for i in np.arange(-1, 2):
        for j in np.arange(-1, 2):
            vec = delta + i * lat_vec_bottom1 + j * lat_vec_bottom2
            phase = -np.dot(K, vec)
            ham += Hopping(vec, 0, 0) * np.exp(
                1.j * phase)  # delta knows about sublattices indices hence 0,0 in this formula
    return [ham, top_atom]


# HamiltForK([0,5],0,0)

def FourierHoppingMinus(moire_vec, sublattice_bottom, sublattice_top):
    # moire vec is described by an array [n,k] and equals n*G1M+k*G2M
    # returns fourier amplitude of hopping function

    ampl = 0
    for n in np.arange(-N, N + 1):
        for k in np.arange(-N, N + 1):
            vec_bottom_atom = n * lat_vec_bottom1 + k * lat_vec_bottom2
            hamiltonian = HamiltForK([n, k], sublattice_bottom, sublattice_top)
            hopping = hamiltonian[0]
            top_atom = hamiltonian[1]
            delta = top_atom[1]
            G = moire_vec[0] * G1M + moire_vec[1] * G2M
            phase = -np.dot(G, vec_bottom_atom) - np.dot(K, delta)  # phase before delta may be -
            ampl += hopping * np.exp(1.j * phase)
    return np.abs(ampl)

def FourierHoppingPlus(moire_vec, sublattice_bottom, sublattice_top):
    # moire vec is described by an array [n,k] and equals n*G1M+k*G2M
    # returns fourier amplitude of hopping function

    ampl = 0
    for n in np.arange(-N, N + 1):
        for k in np.arange(-N, N + 1):
            vec_bottom_atom = n * lat_vec_bottom1 + k * lat_vec_bottom2
            hamiltonian = HamiltForK([n, k], sublattice_bottom, sublattice_top)
            hopping = hamiltonian[0]
            top_atom = hamiltonian[1]
            delta = top_atom[1]
            G = moire_vec[0] * G1M + moire_vec[1] * G2M
            phase = -np.dot(G, vec_bottom_atom) + np.dot(K, delta)  # phase before delta may be -
            ampl += hopping * np.exp(1.j * phase)
    return np.abs(ampl)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('for minus')
    print(FourierHoppingPlus([-1, -1], 0, 0))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
