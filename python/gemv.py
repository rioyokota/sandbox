import numpy as np
import ctypes as ct
a = np.array(range(10),dtype=float)
for i in range(10):
    a[i] = i
wrapper = ct.CDLL('gemv.so')
wrapper.gemv(a.ctypes.data_as(ct.c_void_p),10)
