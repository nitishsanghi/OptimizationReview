import math
import numpy as np
#from numpy import linalg

# measurement model y = Ax; y is 2x1; x is 3x1, A is 2x3

y = np.array([1, 1, 1, 1])
A = np.array([[1, 2, 0],[2, 1, 3],[1, 3, 1], [2, 1, 2]])

# solution using inversion
B = np.transpose(A)
C = np.matmul(B,A);
D = np.linalg.inv(C)
x = np.matmul(D, np.matmul(B,y))
print("Solution using inversion")
print("x solution : " + str(x))
print("abs Error in y from x solution: " + str(np.abs(np.matmul(A,x) - y)))

# Solution using cholesky
L = np.linalg.cholesky(C)
b_bar = np.matmul(B,y)
xy = np.matmul(np.linalg.inv(L),b_bar)
x = np.matmul(np.linalg.inv(np.transpose(L)), xy)
print("Solution using cholesky")
print("x solution : " + str(x))
print("abs Error in y from x solution: " + str(np.abs(np.matmul(A,x) - y)))


