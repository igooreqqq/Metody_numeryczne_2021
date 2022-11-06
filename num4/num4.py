import numpy as np
np.set_printoptions(precision=18)

size = 50

############### uzupełnianie macierzy dla bibliotek numerycznych ##############
A = np.zeros((size, size), dtype=np.double)
A += np.diag(np.full(size, 9.0))
A += np.diag(np.full(size - 1, 7.0), 1)
b_num = np.full((50, 1), 5.0)
u_wektor_num = np.full((50, 1), 1.0)
v_trans_num = np.full((1, 50), 1.0)

A_1 = A + u_wektor_num @ v_trans_num  #macierz z zadania

############ uzupełnianie macierzy dla naszego algorytmu ###################

diag_middle = np.empty((size), dtype=np.double)
diag_up_of_middlediag1 = np.empty((size - 1), dtype=np.double)

for i in range(0, size):
    diag_middle[i] = 9.0

for i in range(0, size - 1):
    diag_up_of_middlediag1[i] = 7.0

b = np.full((50, 1), 5.0)
u_wektor = np.full((50, 1), 1.0)
v_trans = np.full((1, 50), 1.0)

######################## Backward substitution dla z #################################

z = b
zprim = u_wektor

x = size
z[x - 1] = z[x - 1] / diag_middle[x - 1]
z[x - 2] = (z[x - 2] - (z[x - 1] * diag_up_of_middlediag1[x - 2])) / diag_middle[x - 2]

for row in range(x - 3, 0 - 1, -1):
    z[row] = (z[row] - (z[row + 1] * diag_up_of_middlediag1[row])) / diag_middle[row]

######################## Backward substitution dla zprim #################################

x = size
zprim[x - 1] = zprim[x - 1] / diag_middle[x - 1]
zprim[x - 2] = (zprim[x - 2] - (zprim[x - 1] * diag_up_of_middlediag1[x - 2])) / diag_middle[x - 2]

for row in range(x - 3, 0 - 1, -1):
    zprim[row] = (zprim[row] - (zprim[row + 1] * diag_up_of_middlediag1[row])) / diag_middle[row]

########################### Liczenie wzorów ##################################

################## dla licznik2 -> skalar
Clicznik2 = np.zeros((1, 1), dtype=np.double)

for i in range(len(v_trans)):
    for j in range(len(z[0])):
         for k in range(len(z)):
            Clicznik2[i][j] += v_trans[i][k] * z[k][j]

################## dla licznik1 -> wektor
Clicznik1 = np.zeros((50, 1), dtype=np.double)

for i in range(len(zprim)):
    for j in range(len(Clicznik2[0])):
         for k in range(len(Clicznik2)):
            Clicznik1[i][j] += zprim[i][k] * Clicznik2[k][j]

################### mianownik -> skalar
Cmianownik = np.zeros((1, 1), dtype=np.double)

for i in range(len(v_trans)):
    for j in range(len(zprim[0])):
         for k in range(len(zprim)):
            Cmianownik[i][j] += v_trans[i][k] * zprim[k][j]

y = z - (Clicznik1 / (1 + Cmianownik)) # rozwiązanie równania z zadania

print("Wynik równania używając naszego algorytmu")
print(np.transpose(y)) #wynik transponowany żeby print zajmował mniej miejsca

############################ Sprawdzenie wyniku za pomocą bibliotek numerycznych ###################################

y_num = np.linalg.solve(A_1, b_num)
print('\n')
print("Wynik równania używając biblioteki numerycznej")
print(np.transpose(y_num))