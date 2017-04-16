import ctypes, numpy
lib = ctypes.CDLL('./libstep05.so')
lib.matrix.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.uintp, ndim=1, flags='C'), ctypes.c_int, ctypes.c_int]
u = numpy.zeros((2,3))
upp = (u.__array_interface__['data'][0] + numpy.arange(u.shape[0])*u.strides[0]).astype(numpy.uintp)
lib.matrix(upp, 2, 3)
print u
