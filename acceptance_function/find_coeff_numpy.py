"""
Find the acceptance function given some data:

REVISED: refactoring of original find_coeff.py script

from find_coeff import get_acceptance
accepatnce = get_acceptance(data)

"""

import numpy as np
import numpy.polynomial.legendre as npl

from functools import reduce



### EVALUATION ### 


def legendre_eval(C, ctl, ctk, phi):
    phi = phi / np.pi # Have to rescale phi to range -1 to 1
    return npl.legval3d(ctl, ctk, phi, C)

# projection functions

def legendre_eval_project_1D(C, x, i):
    slices = [0] * 3
    slices[i] = slice(None)
    if i == 2:
        x = x / np.pi
    return npl.legval(x, C[slices[0],slices[1],slices[2]])


def legendre_eval_project_2D(C, x, y, i, j):
    slices = [0] * 3
    slices[i] = slices[j] = slice(None)
    return npl.legval2d(x, y, C[slices[0],slices[1],slices[2]])

# MODIFY FOR UCERTAINTY
def acc_modify(acceptance_all, acceptance_std, modify):
    if modify == 1:
        for b in range(len(acceptance_all)):
            for i in range(len(acceptance_all[b])):
                for j in range(len(acceptance_all[b][i])):
                    for k in range(len(acceptance_all[b][i][j])):
                        a_bijk = acceptance_all[b][i][j][k]
                        if a_bijk > 0:
                            acceptance_all[b][i][j][k] = a_bijk + acceptance_std[b][i][j][k]
                        else:
                            acceptance_all[b][i][j][k] = a_bijk - acceptance_std[b][i][j][k]
    elif modify == -1:
        for b in range(len(acceptance_all)):
            for i in range(len(acceptance_all[b])):
                for j in range(len(acceptance_all[b][i])):
                    for k in range(len(acceptance_all[b][i][j])):
                        a_bijk = acceptance_all[b][i][j][k]
                        if a_bijk > 0:
                            acceptance_all[b][i][j][k] = a_bijk - acceptance_std[b][i][j][k]
                        else:
                            acceptance_all[b][i][j][k] = a_bijk + acceptance_std[b][i][j][k]

    elif modify:
        for b in range(len(acceptance_all)):
            for i in range(len(acceptance_all[b])):
                for j in range(len(acceptance_all[b][i])):
                    for k in range(len(acceptance_all[b][i][j])):
                        a_bijk = acceptance_all[b][i][j][k]
                        r = (rand.random() - 0.5) * 2
                        acceptance_all[b][i][j][k] = a_bijk + r*acceptance_std[b][i][j][k]
    
    return acceptance_all

def load_acceptance(path = "acceptance/acceptance_legendre_coeffs.npz"):
    """
    Parameters
    ----------
    path : str
        The default is "acceptance/acceptance_legendre_coeffs.npz".

    Returns
    -------
    acceptance : list
        One function per bin, each taking list ctl, ctk, phi [ctl, ctk, phi]
        as argument
    acceptance_l : list
        One function per bin, each taking ctl as argument
    acceptance_k : list
        One function per bin, each taking ctk as argument
    acceptance_p : list
        One function per bin, each taking phi as argument
    acceptance_file : npz file
        The original file, you may want it.
    """
    acceptance_file = np.load(path)
    acceptance = []
    acceptance_l = []
    acceptance_k = []
    acceptance_p = []
    
    def all_acceptance_0 (ctl, ctk, phi):
        coeffs = acceptance_file["0"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_0 (ctl):
        coeffs = acceptance_file["0"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_0 (ctk):
        coeffs = acceptance_file["0"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_0 (phi):
        coeffs = acceptance_file["0"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_0)
    acceptance_l.append(acceptance_l_0)
    acceptance_k.append(acceptance_k_0)
    acceptance_p.append(acceptance_p_0)
    
    def all_acceptance_1 (ctl, ctk, phi):
        coeffs = acceptance_file["1"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_1 (ctl):
        coeffs = acceptance_file["1"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_1 (ctk):
        coeffs = acceptance_file["1"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_1 (phi):
        coeffs = acceptance_file["1"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_1)
    acceptance_l.append(acceptance_l_1)
    acceptance_k.append(acceptance_k_1)
    acceptance_p.append(acceptance_p_1)
    
    def all_acceptance_2 (ctl, ctk, phi):
        coeffs = acceptance_file["2"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_2 (ctl):
        coeffs = acceptance_file["2"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_2 (ctk):
        coeffs = acceptance_file["2"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_2 (phi):
        coeffs = acceptance_file["2"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_2)
    acceptance_l.append(acceptance_l_2)
    acceptance_k.append(acceptance_k_2)
    acceptance_p.append(acceptance_p_2)
    
    def all_acceptance_3 (ctl, ctk, phi):
        coeffs = acceptance_file["3"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_3 (ctl):
        coeffs = acceptance_file["3"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_3 (ctk):
        coeffs = acceptance_file["3"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_3 (phi):
        coeffs = acceptance_file["3"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_3)
    acceptance_l.append(acceptance_l_3)
    acceptance_k.append(acceptance_k_3)
    acceptance_p.append(acceptance_p_3)
    
    def all_acceptance_4 (ctl, ctk, phi):
        coeffs = acceptance_file["4"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_4 (ctl):
        coeffs = acceptance_file["4"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_4 (ctk):
        coeffs = acceptance_file["4"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_4 (phi):
        coeffs = acceptance_file["4"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_4)
    acceptance_l.append(acceptance_l_4)
    acceptance_k.append(acceptance_k_4)
    acceptance_p.append(acceptance_p_4)
    
    def all_acceptance_5 (ctl, ctk, phi):
        coeffs = acceptance_file["5"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_5 (ctl):
        coeffs = acceptance_file["5"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_5 (ctk):
        coeffs = acceptance_file["5"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_5 (phi):
        coeffs = acceptance_file["5"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_5)
    acceptance_l.append(acceptance_l_5)
    acceptance_k.append(acceptance_k_5)
    acceptance_p.append(acceptance_p_5)
    
    def all_acceptance_6 (ctl, ctk, phi):
        coeffs = acceptance_file["6"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_6 (ctl):
        coeffs = acceptance_file["6"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_6 (ctk):
        coeffs = acceptance_file["6"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_6 (phi):
        coeffs = acceptance_file["6"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_6)
    acceptance_l.append(acceptance_l_6)
    acceptance_k.append(acceptance_k_6)
    acceptance_p.append(acceptance_p_6)
    
    def all_acceptance_7 (ctl, ctk, phi):
        coeffs = acceptance_file["7"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_7 (ctl):
        coeffs = acceptance_file["7"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_7 (ctk):
        coeffs = acceptance_file["7"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_7 (phi):
        coeffs = acceptance_file["7"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_7)
    acceptance_l.append(acceptance_l_7)
    acceptance_k.append(acceptance_k_7)
    acceptance_p.append(acceptance_p_7)
    
    def all_acceptance_8 (ctl, ctk, phi):
        coeffs = acceptance_file["8"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_8 (ctl):
        coeffs = acceptance_file["8"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_8 (ctk):
        coeffs = acceptance_file["8"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_8 (phi):
        coeffs = acceptance_file["8"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_8)
    acceptance_l.append(acceptance_l_8)
    acceptance_k.append(acceptance_k_8)
    acceptance_p.append(acceptance_p_8)
    
    def all_acceptance_9 (ctl, ctk, phi): 
        coeffs = acceptance_file["9"]
        return legendre_eval(coeffs, ctl, ctk, phi)
    def acceptance_l_9 (ctl):
        coeffs = acceptance_file["9"]
        return legendre_eval_project_1D(coeffs, ctl, 0)
    def acceptance_k_9 (ctk):
        coeffs = acceptance_file["9"]
        return legendre_eval_project_1D(coeffs, ctk, 1)
    def acceptance_p_9 (phi):
        coeffs = acceptance_file["9"]
        return legendre_eval_project_1D(coeffs, phi, 2)
    acceptance.append(all_acceptance_9)
    acceptance_l.append(acceptance_l_9)
    acceptance_k.append(acceptance_k_9)
    acceptance_p.append(acceptance_p_9)
    
    return acceptance, acceptance_l, acceptance_k, acceptance_p, acceptance_file


### CALCULATE COEFFS ###


def find_coeff(data, order=[4]*3):
    if isinstance(order, int):
        order = [order]*3

    order_p1 = [x+1 for x in order]

    i_s = np.arange(order_p1[0], dtype=np.float64)
    j_s = np.arange(order_p1[1], dtype=np.float64)
    k_s = np.arange(order_p1[2], dtype=np.float64)

    ijk_grid = np.meshgrid(i_s, j_s, k_s)
    c_ijk = reduce(np.multiply, (2 * ijk_grid[ith] + 1 for ith in range(3)))
    c_ijk /= float(8 * data.shape[0])

    x = data[:,0]
    y = data[:,1]
    z = data[:,2] / np.pi # rescale phi values 

    V = npl.legvander3d(x, y, z, deg=order)
    # print('V, c', V.shape, c_ijk.shape)
    C = np.sum(V * c_ijk.flat, axis=0).reshape(order_p1)
    
    return C


def get_acceptance(data, order = 4):
    C = find_coeff(data, order)
    return lambda ctl, ctk, phi: legendre_eval(C, ctl, ctk, phi, order)



def indx_between(arr, above, below):
    return np.logical_and(arr > above, arr < below)



if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    order = 4
    n_u = 5000
    n_g = 500
    n = n_u + n_g

    def gen_rand_samples(mean, std, n_g, n_u):
        g = np.random.normal(mean, std, n_g)
        u = np.random.uniform(-1, 1, n_u)
        return np.concatenate((g, u))

    data = np.zeros((n, 3))
    data[:,0] = gen_rand_samples(0.3, 0.5, n_g, n_u)
    data[:,1] = gen_rand_samples(0, 0.3, n_g, n_u)
    data[:,2] = gen_rand_samples(-0.2, 0.3, n_g, n_u)

    # roughly limit data 
    data[data < -1] = 0
    data[data > 1] = 0

    data[:, 2] = np.pi * (data[:, 2] + 1)

    C = find_coeff(data)
    print(C.shape)
    print(C[0])

    ctl_fit = np.linspace(-1, 1, 100)
    ctk_fit = np.linspace(-1, 1, 100)
    phi_fit = np.linspace(-1, 1, 100)
    phi_fit = np.pi * (phi_fit + 1)

    ctl_u, ctk_u, phi_u = np.meshgrid(ctl_fit, ctk_fit, phi_fit)
    z_fit = legendre_eval(C, ctl_u, ctk_u, phi_u)

    z_fit_i = np.sum(z_fit, axis=(1, 2))
    z_fit_j = np.sum(z_fit, axis=(0, 2))
    z_fit_k = np.sum(z_fit, axis=(0, 1))
    # print(z_fit)

    # print(legendre_eval(C, 0, 0, 0))

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.title('ctl')
    plt.hist(data[:,0], density=True)
    plt.plot(ctl_fit, z_fit_i / np.max(z_fit_i), label='i')
    
    plt.subplot(132)
    plt.title('ctk')
    plt.hist(data[:,1], density=True)
    plt.plot(ctk_fit, z_fit_j / np.max(z_fit_j), label='j')

    plt.subplot(133)
    plt.title('phi')
    plt.hist(data[:,2], density=True)
    plt.plot(phi_fit, z_fit_k / np.max(z_fit_k), label='k')

    plt.show()

    # data = np.random.uniform(-1, 1, (n, 3))
    # data[:, 2] = np.pi * (data[:, 2] + 1)

    # t_start = time.time()
    # acc = get_acceptance(data)
    # t_end = time.time()
    # print(f'Took {t_end - t_start:.3f} seconds')

    # a = data[:, 0]
    # b = data[:, 1]
    # c = data[:, 2]

    # indx1 = indx_between(a, 0.5, 0.6)
    # indx2 = indx_between(b, 0.5, 0.6)
    # indx3 = indx_between(c, 0.5, 0.5)
    # indx = np.logical_and(np.logical_and(indx1, indx2), indx3)
    # p1 = data[indx, :]
    # p1 = len(p1)

    # indx1 = indx_between(a, -0.3, -0.2)
    # indx2 = indx_between(b, -0.3, -0.2)
    # indx3 = indx_between(c, -0.3, -0.2)
    # indx = np.logical_and(np.logical_and(indx1, indx2), indx3)
    # p2 = data[indx, :]
    # p2 = len(p2)

    # print(p1, p2)
    # # print(p1 / p2)
    # print(acc(0.55, 0.55, 0.55), acc(-0.25, -0.25, -0.25))
    # print(acc(0.55, 0.55, 0.55) / acc(-0.25, -0.25, -0.25))

