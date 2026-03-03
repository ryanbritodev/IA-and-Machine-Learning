import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

def derivadas():
  a = np.sum((X_r - np.mean(X_r)) * (Y_r - np.mean(Y_r)))/np.sum((X_r - np.mean(X_r))**2)
  b = np.mean(Y_r) - a * np.mean(X_r)
  return a,b

def calcula_erro(a, b):
  erro = 0
  for x_r, y_r in zip(X_r, Y_r):
    y_p = a * x_r + b
    erro += (y_r - y_p) ** 2
  return erro

data = pd.read_excel('data.xlsx')
X_r = data['x']
Y_r = data['y']

A = np.linspace(-10, 10, 100)
B = np.linspace(-10, 10, 100)

menor_erro = calcula_erro(A[0], B[0])
melhor_a = A[0]
melhor_b = B[0]

for i, a in enumerate(A):
  for j, b in enumerate(B):
    erro = calcula_erro(a,b)
    if erro < menor_erro:
      menor_erro = erro
      melhor_a = a
      melhor_b = b

a,b = derivadas()

melhor_reta = melhor_a * X_r + melhor_b

plt.plot(X_r, Y_r, 'bo', label = 'Dados')
plt.plot(X_r, melhor_reta, 'r', label = f'Melhor Reta: a = {melhor_a:.2f}\nb = {melhor_b:.2f}')
plt.plot(X_r, a*X_r + b, 'g', label = f"Via derivadas: a = {a:.2f}\n")
plt.legend()
plt.show()
