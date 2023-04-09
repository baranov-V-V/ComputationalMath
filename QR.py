import numpy as np
np.warnings.filterwarnings("ignore")

def MakeQR()

def qr_solve(A, b):
    # QR-decomposition
    P = np.zeros(A.shape)
    Q = np.zeros(A.shape)
    R = np.zeros((A.shape[1], A.shape[1]))
    for j in range(A.shape[1]):
        P[:,j] = A[:,j]
        for i in range(j):
            R[i,j] = np.dot(P[:,j],Q[:,i])
            P[:,j] = P[:,j] - Q[:,i]*R[i,j]
        R[j,j] = np.linalg.norm(P[:,j])
        Q[:,j] = P[:,j] / R[j,j]
        
    
    # Finding y = Q*b for Rx = Q*b  
    y = Q.T @ b
    
    # Finding solution x for Rx = y
    x = np.zeros(A.shape[1])
    for i in range(len(x) - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, len(x)):
            x[i] = x[i] - R[i,j]*x[j]
        x[i] = x[i] / R[i,i]
    return Q, R, x
    
# Testing
n = 4
A = np.random.rand(n,n)
b = np.random.rand(n)
print(A)
A = A + np.eye(n)
Q, R, x = qr_solve(A, b)
print("My QR-decomposition method: x = ", x)
x_1 = np.linalg.solve(A, b)
print("Built-in function: x = ", x_1)

m, n = 7, 4
A = np.random.rand(m,n)
for i in range(n):
    x = np.zeros(m)
    x[i] = 1
    A[:,i] = A[:,i] + x
b = np.random.rand(m)
Q, R, x = qr_solve(A, b)
print("My QR-decomposition method: x = ", x)
x_1 = np.linalg.lstsq(A, b)
print("Built-in function: x = ", x_1[0])