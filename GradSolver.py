import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

def SolveEq(A, b, max_iter, tol, is_grad_boost):
  iter = 0
  
  x_k = np.zeros(len(b))
  r_k = np.dot(A, x_k) - b
  
  R = []
  R.append(LA.norm(r_k))
  
  while (iter < max_iter and LA.norm(r_k) > tol):
    iter += 1
    
    if (is_grad_boost):
      t_k = np.dot(r_k, r_k) / np.dot(np.dot(A, r_k), r_k)
    else:
      t_k = np.dot(r_k, np.dor(A, r_k)) / (LA.norm(np.dot(A, r_k)) ** 2)
    
    x_k = x_k - t_k * r_k
    r_k = np.dot(A, x_k) - b
    
    R.append(LA.norm(r_k))
  
  return x_k, R, iter

n = 5

A = np.random.rand(n, n)
A = np.dot(A, A.T) + np.eye(n)

b = np.random.rand(n)

x, R, iter = SolveEq(A, b, 1000, 1e-6, True)

print(iter, "iterations were counted")
print("Mistake is ", LA.norm(x - LA.solve(A, b)))

fig, ax = plt.subplots(figsize = (10, 5))
ax.semilogy(R)
ax.grid(True)

plt.show()