import math, numpy
from matplotlib import pyplot

nx = 16
pi = math.pi
dx = 2*pi/(nx-1)
x = numpy.linspace(0,2*pi-dx,nx)
k = numpy.linspace(0,nx-1,nx)
f = numpy.ones(nx)
f[8:] = -1

fk = numpy.fft.fft(f,nx)

pyplot.subplot(121)
pyplot.plot(x, f, 'o-')
pyplot.axis([0, 2*pi, -1.5, 1.5])
pyplot.subplot(122)
pyplot.plot(k, fk.real, 'o-', label='real')
pyplot.plot(k, fk.imag, 'o-', label='imag')
pyplot.axis([0, nx, -nx, nx])
pyplot.legend(loc='upper right')
pyplot.show()
