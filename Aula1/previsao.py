import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

def calcula_erro(a, b):
  erro = 0
  for x_r, y_r in zip(X_r, Y_r):
    y_p = a * x_r + b
    erro += (y_r - y_p) ** 2
  return erro

data = pd.read_excel('/content/data.xlsx')
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


melhor_reta = melhor_a * X_r + melhor_b

plt.plot(X_r, Y_r, 'bo', label = 'Dados')
plt.plot(X_r, melhor_reta, 'r', label = f'Melhor Reta: a = {melhor_a}\nb = {melhor_b}')
plt.legend()
plt.show()
