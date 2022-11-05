import numpy as np
np.set_printoptions(precision=18)

size = 100

############### uzupełnianie macierzy dla bibliotek numerycznych ##############

array1 = np.arange(1, size)
array2 = np.arange(1, size - 1)
gorna_diag1 = 0.1 / array1
gorna_diag2 = 0.4 / np.power(array2, 2)

A = np.zeros((size, size), dtype=np.double)
A += np.diag(np.full(size - 1, 0.2), -1)
A += np.diag(np.full(size, 1.2))
A += np.diag(gorna_diag1, 1)
A += np.diag(gorna_diag2, 2)

#print(A)

############ uzupełnianie macierzy dla naszego algorytmu ###################

x = np.empty(size, dtype=np.double)
diag_middle = np.empty(size, dtype=np.double)
U1 = diag_middle
diag_below1 = np.empty((size - 1), dtype=np.double)
L1 = diag_below1
diag_up1 = np.empty((size - 1), dtype=np.double)
U2 = diag_up1
diag_up2 = np.empty((size - 2), dtype=np.double)
U3 = diag_up2

for i in range(0, size):
    diag_middle[i] = 1.2
    x[i] = i + 1

for i in range(0, size - 1):
    diag_below1[i] = 0.2
    diag_up1[i] = 0.1 / (i + 1)

for i in range(0, size - 2):
    diag_up2[i] = 0.4 / pow(i + 1, 2)

################### Faktoryzacja LU #####################################

for i in range(1, size - 2):
    L1[i - 1] = diag_below1[i - 1] / U1[i - 1]
    U1[i] = diag_middle[i] - (L1[i - 1] * U2[i - 1])
    U2[i] = diag_up1[i] - (L1[i - 1] * U3[i - 1])
    U3[i] = diag_up2[i]

L1[size - 3] = diag_below1[size - 3] / U1[size - 3]
U1[size - 2] = diag_middle[size - 2] - (L1[size - 3] * U2[size - 3])
U2[size - 2] = diag_up1[size - 2] - (L1[size - 3] * U3[size - 3])

L1[size - 2] = diag_below1[size - 2] / U1[size - 2]
U1[size - 1] = diag_middle[size - 1] - (L1[size - 2] * U2[size - 2])

############## forward substitution #################################

z = np.empty(size)
z[0] = x[0]

for i in range(1, size):
    z[i] = x[i] - (z[i - 1] * L1[i - 1])

############### back substitution ############################

y = np.empty(size)
y[size - 1] = z[size - 1] / U1[size - 1]
y[size - 2] = (z[size - 2] - (y[size - 1] * U2[size - 2])) / U1[size - 2]

for i in range(size - 3, 0 - 1, -1):
    y[i] = (z[i] - (y[i + 1] * U2[i]) - (y[i + 2] * U3[i])) / U1[i]

print(y)

############## Wyznacznik macierzy A #####################

detA = 1
for i in range(0, size):
    detA *= diag_middle[i]

print('\n', "Wyznacznik macierzy A: ", detA)

############################ Sprawdzenie wyniku za pomocą bibliotek numerycznych ###################################

y_num = np.linalg.solve(A, x)
print('\n')
print("Wynik równania używając biblioteki numerycznej")
print(np.transpose(y_num))

detA_num = np.linalg.det(A)
print('\n')
print("Wyznacznik macierzy A używając biblioteki numerycznej")
print(detA_num)