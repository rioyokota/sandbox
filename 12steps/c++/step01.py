import ctypes
lib = ctypes.CDLL('./libstep01.so')
lib.hello()
