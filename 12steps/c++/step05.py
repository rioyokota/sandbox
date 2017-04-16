from ctypes import *
import numpy
lib = CDLL('./libstep05.so')
u = numpy.ones((2,3))
lib.matrix(u.ctypes.data_as(numpy.ctypeslib.ndpointer(dtype=numpy.float64,),int(2),int(3))
print u
