import numpy as np
import ctypes as ct

def cg(A, b, x_init):
    x = x_init
    r0 = b - np.dot(A,x)
    p = r0
    k = 0
    for i in range(10):
        a = float( np.dot(r0.T,r0) / np.dot(np.dot(p.T, A),p) )
        x = x + p*a
        r1 = r0 - np.dot(A*a, p)
        print np.linalg.norm(r1)
        if np.linalg.norm(r1) < 1.0e-10:
            return x
        b = float( np.dot(r1.T, r1) / np.dot(r0.T, r0) )
        p = r1 + b * p
        r0 = r1
    return x

n = 10
max_it = 10
tol = 1e-6
A = np.random.rand(n,n)
A = A + A.transpose() + n * np.identity(n)
b = np.random.rand(n)
x = np.zeros(n)
wrapper = ct.CDLL('cg.so')
wrapper.solve(A.ctypes.data_as(ct.c_void_p),
              b.ctypes.data_as(ct.c_void_p),
              x.ctypes.data_as(ct.c_void_p),
              n, max_it, ct.c_double(tol))
x2 = np.linalg.solve(A, b)
x3 = np.zeros(n)
x3 = cg(A, b, x3)
print x
print x2
print x3
