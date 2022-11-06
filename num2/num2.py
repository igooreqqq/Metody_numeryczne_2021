import numpy as np
import math

################ uzupełnianie macierzy z zadania #############################
A_1 = np.array([[2.40827208, -0.36066254, 0.80575445, 0.46309511, 1.20708553],
             [-0.36066254, 1.14839502, 0.02576113, 0.02672584, -1.03949556],
             [0.80575445, 0.02576113, 2.45964907, 0.13824088, 0.0472749],
             [0.46309511, 0.02672584, 0.13824088, 2.05614464, -0.9434493],
             [1.20708553, -1.03949556, 0.0472749, -0.9434493, 1.92753926]])

A_2 = np.array([[2.61370745, -0.6334453, 0.76061329, 0.24938964, 0.82783473],
               [-0.6334453, 1.51060349, 0.08570081, 0.31048984, -0.53591589],
               [0.76061329, 0.08570081, 2.46956812, 0.18519926, 0.13060923],
               [0.24938964, 0.31048984, 0.18519926, 2.27845311, -0.54893124],
               [0.82783473, -0.53591589, 0.13060923, -0.54893124, 2.6276678]])


b = np.array([[5.40780228], [3.67008677], [3.12306266], [-1.11187948], [0.54437218]])

b_prim = b + np.array([[pow(10, -5)], [0], [0], [0], [0]])

################ rozwiązywanie równań ###############
y_1 = np.linalg.solve(A_1, b)

y_2 = np.linalg.solve(A_2, b)

y_1prim = np.linalg.solve(A_1, b_prim)

y_2prim = np.linalg.solve(A_2, b_prim)

print("\nRozwiązaniem równania A_1 y_1 = b jest wektor: ")
print(y_1)

print("\nRozwiązaniem równania A_2 y_2 = b jest wektor: ")
print(y_2)

print("\nRozwiązaniem równania A_1 y_1prim = b_prim jest wektor: ")
print(y_1prim)

print("\nRozwiązaniem równania A_2 y_2prim = b_prim jest wektor: ")
print(y_2prim)

################ liczenie delty_1 oraz delty_2 ###############
delta_1 = np.linalg.norm(y_1 - y_1prim)
delta_2 = np.linalg.norm(y_2 - y_2prim)

print("\nDelta_1 wynosi: ")
print(delta_1)

print("\nDelta_2 wynosi: ")
print(delta_2)

################ liczenie współczynnika kappa ###############
wartosci_wlasne_A1 = np.linalg.eigvals(A_1)

min = math.fabs(np.min(wartosci_wlasne_A1))
max = math.fabs(np.max(wartosci_wlasne_A1))

kappa_A1 = max * (1 / min)
print("\nWspółczynnik uwarunkowania macierzy A_1 wynosi: ")
print(kappa_A1)

wartosci_wlasne_A2 = np.linalg.eigvals(A_2)

min_A2 = math.fabs(np.min(wartosci_wlasne_A2))
max_A2 = math.fabs(np.max(wartosci_wlasne_A2))

kappa_A2 = max_A2 * (1 / min_A2)
print("\nWspółczynnik uwarunkowania macierzy A_2 wynosi: ")
print(kappa_A2)