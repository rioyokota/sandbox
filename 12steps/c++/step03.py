import ctypes, numpy
lib = ctypes.CDLL('./libstep03.so')
a = numpy.linspace(0, 9, 10)
lib.minus.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.minus(a.ctypes.data, 10)
print a
