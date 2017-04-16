import ctypes, numpy
from matplotlib import pyplot
lib = ctypes.CDLL('./libstep04.so')
lib.convection.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]

nx = 41
dx = 2./(nx-1)
nt = 50
dt = .01
c = 1
x = numpy.linspace(0, 2, nx)
u = numpy.ones(nx)
u[10:20] = 2

for n in range(nt):
    lib.convection(u.ctypes.data, nx, dx, dt, c)
    pyplot.plot(x, u)
    pyplot.axis([0, 2, .5, 2.5])
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
