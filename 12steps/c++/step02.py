import ctypes
lib = ctypes.CDLL('./libstep02.so')
print lib.add1(5)
