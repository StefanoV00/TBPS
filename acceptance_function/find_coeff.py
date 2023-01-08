"""
Find the acceptance function given some data:

from find_coeff import get_acceptance
accepatnce = get_acceptance(data)

Warning, it is quite slow as there are a lot of coefficients to calculate
IMPORTANT: rescale phi to range -1 to 1
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

L0 = lambda x: 1
L1 = lambda x: x
L2 = lambda x: 1.5 * x**2 - 0.5
L3 = lambda x: 2.5 * x**3 - 1.5 * x
L4 = lambda x: 35 * x**4 / 8 - 30 * x**2 / 8 + 3 / 8
L = np.array([L0, L1, L2, L3, L4])

order = 4
C = np.random.random((order + 1, order + 1, order + 1))

def f_1d(C, var1):
    tot = 0
    for i in range(order + 1):
        tot += C[i] * L[i](var1)
    return tot

def f_2d(C, var1, var2):
    tot = 0
    for i in range(order + 1):
        for j in range(order + 1):
            tot += C[i, j] * L[i](var1) * L[j](var2)
    return tot

def f_3d(C, ctl, ctk, phi):
    tot = 0
    for i in range(order + 1):
        for j in range(order + 1):
            for k in range(order + 1):
                tot += C[i, j, k] * L[i](ctk) * L[j](ctl) * L[k](phi)
    return tot

# Important - rescale phi to between -1 and 1
def find_coeff_1d(data, order = 4):
    C = np.zeros((order + 1, ))
    for i in range(order + 1):
        for x in range(data.shape[0]):
            C[i] += L[i](data[x])
        C[i] *= (2 * i + 1)
    C = C / data.shape[0]
    C = C / 2
    return C

# Important - rescale phi to between -1 and 1
def find_coeff_2d(data, order = 4):
    C = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(order + 1):
            for x in range(data.shape[0]):
                C[i, j] += L[i](data[x, 0])*L[j](data[x, 1])
            C[i, j] *= (2 * i + 1) * (2 * j + 1)
    C = C / data.shape[0]
    C = C / 4
    return C

# Important - rescale phi to between -1 and 1
def find_coeff_3d(data, order = 4):
    C = np.zeros((order + 1, order + 1, order + 1))
    for i in range(order + 1):
        for j in range(order + 1):
            for k in range(order + 1):
                for x in range(data.shape[0]):
                    C[i, j, k] += L[i](data[x, 0])*L[j](data[x, 1])*L[k](data[x, 2])
                C[i, j, k] *= (2 * i + 1) * (2 * j + 1) * (2 * k + 1)
    C = C / data.shape[0]
    C = C / 8
    return C

def get_acceptance_1d(data, order = 4):
    C = find_coeff_1d(data, order)
    return lambda var1: f_1d(C, var1)

def get_acceptance_2d(data, order = 4):
    C = find_coeff_2d(data, order)
    return lambda var1, var2: f_2d(C, var1, var2)

def get_acceptance_3d(data, order = 4):
    C = find_coeff_3d(data, order)
    return lambda var1, var2, var3: f_3d(C, var1, var2, var3)

if __name__ == "__main__":
    n = 5000

    data = np.random.uniform(-1, 1, (n, 3))
    acc = get_acceptance_3d(data)

    x = np.linspace(-1, 1, 30)
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = acc(x[i], 0, 0)

    plt.plot(x, y)
    plt.show()


