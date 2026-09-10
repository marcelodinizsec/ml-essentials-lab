import numpy as np
import matplotlib.pyplot as plt

A = np.array([1, 2])
B = np.array([4, 6])

plt.scatter(A[0], A[1], color='blue', label='A')
plt.scatter(B[0], B[1], color='red', label='B')

# Linha entre os pontos
plt.plot([A[0], B[0]], [A[1], B[1]], 'k--')

plt.text(A[0], A[1], ' A')
plt.text(B[0], B[1], ' B')

plt.title("Euclidean Distance between A and B")
plt.legend()
plt.grid(True)
plt.show()
