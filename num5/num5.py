import math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=18)

size = 100

def jacobi(b, diag_middle, diag_up1, diag_up2, diag_low1, diag_low2, x, number_of_iteration, epsilon):
    iteration = 0

    while (iteration != number_of_iteration):
        x_new = np.zeros(size)
        x_new[0] = (b[0] - (diag_up1[0] * x[1]) - (diag_up2[0] * x[2])) / diag_middle[0]
        x_new[1] = (b[1] - (diag_low1[0] * x[0]) - (diag_up1[1] * x[2]) - (diag_up2[1] * x[3])) / diag_middle[1]

        for i in range(2, size - 2):
            x_new[i] = (b[i] - (diag_low2[i - 2] * x[i - 2]) - (diag_low1[i - 1] * x[i - 1]) - (diag_up1[i] * x[i + 1]) - (diag_up2[i] * x[i + 2])) / diag_middle[i]

        x_new[size - 2] = (b[size - 2] - (diag_low2[size - 4] * x[size - 4]) - (diag_low1[size - 3] * x[size - 3]) - (diag_up1[size - 2] * x[size - 1])) / diag_middle[size - 2]
        x_new[size - 1] = (b[size - 1] - (diag_low2[size - 3] * x[size - 3]) - (diag_low1[size - 2] * x[size - 2])) / diag_middle[size - 1]

        norma2 = 0

        for k in range(0, size):
            add = pow(x_new[k] - x[k], 2)
            norma2 += add

        norma2 = math.sqrt(norma2)
        if (norma2 < epsilon):
            break

        norm2.append(norma2)
        x = x_new
        iteration += 1

    return x, norm2

def gauss_seidel(b, diag_middle, diag_up1, diag_up2, diag_low1, diag_low2, x, number_of_iteration, epsilon):
    iteration = 0

    while (iteration != number_of_iteration):
        x_new = np.zeros(size)
        x_new[0] = (b[0] - (diag_up1[0] * x[1]) - (diag_up2[0] * x[2])) / diag_middle[0]
        x_new[1] = (b[1] - (diag_low1[0] * x_new[0]) - (diag_up1[1] * x[2]) - (diag_up2[1] * x[3])) / diag_middle[1]

        for i in range(2, size - 2):
            x_new[i] = (b[i] - (diag_low2[i - 2] * x_new[i - 2]) - (diag_low1[i - 1] * x_new[i - 1]) - (diag_up1[i] * x[i + 1]) - (diag_up2[i] * x[i + 2])) / diag_middle[i]

        x_new[size - 2] = (b[size - 2] - (diag_low2[size - 4] * x_new[size - 4]) - (diag_low1[size - 3] * x_new[size - 3]) - (diag_up1[size - 2] * x[size - 1])) / diag_middle[size - 2]
        x_new[size - 1] = (b[size - 1] - (diag_low2[size - 3] * x_new[size - 3]) - (diag_low1[size - 2] * x_new[size - 2])) / diag_middle[size - 1]

        norma = 0

        for k in range(0, size):
            add = pow(x_new[k] - x[k], 2)
            norma += add

        norma = math.sqrt(norma)
        if(norma < epsilon):
            break
        norm.append(norma)
        x = x_new
        iteration += 1

    return x, norm

diag_middle = np.empty(size, dtype=np.double)
diag_up1 = np.empty(size, dtype=np.double)
diag_up2 = np.empty(size, dtype=np.double)
diag_low1 = np.empty(size, dtype=np.double)
diag_low2 = np.empty(size, dtype=np.double)
b = np.zeros(size)
x = np.zeros(size)
x_innewartosci = np.zeros(size)

for i in range(0, size):
    diag_middle[i] = 3.0
    b[i] = i + 1
    x_innewartosci[i] = 10 + i * 10

for i in range(0, size - 1):
    diag_up1[i] = 1.0
    diag_low1[i] = 1.0

for i in range(0, size - 2):
    diag_up2[i] = 0.2
    diag_low2[i] = 0.2

norm = []
norm2 = []

A = np.zeros((size, size), dtype=np.double)
A += np.diag(np.full(size - 2, 0.2), 2)
A += np.diag(np.full(size - 1, 1.0), 1)
A += np.diag(np.full(size, 3.0))
A += np.diag(np.full(size - 1, 1.0), -1)
A += np.diag(np.full(size - 2, 0.2), -2)

x_dokladnywynik = np.linalg.solve(A, b)
x_zera = np.zeros(size)

x_jacobi, norm_jacobi = jacobi(b, diag_middle, diag_up1, diag_up2, diag_low1, diag_low2, x_zera, 100, 10e-10)
x_gauss_seidel, norm_gauss_seidel = gauss_seidel(b, diag_middle, diag_up1, diag_up2, diag_low1, diag_low2, x_zera, 100, 10e-10)
print(x_jacobi)
print(x_gauss_seidel)
plt.plot(norm_gauss_seidel, label='Gauss-Seidel')
plt.plot(norm_jacobi, label='Jacobi')
plt.xlabel('Liczba iteracji')
plt.ylabel('||x-x$^*$||$_2$')
plt.grid(True)
plt.yscale('log')
plt.legend()
plt.show()