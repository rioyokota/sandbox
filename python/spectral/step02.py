import math, numpy
from matplotlib import pyplot

nx = 8
pi = math.pi
dx = 2*pi/nx
x = numpy.linspace(0,2*pi-dx,nx)
k = numpy.linspace(0,nx-1,nx)
f = numpy.sin(2*x)
g = numpy.sin(3*x)
#h = f * g

fk = numpy.fft.fft(f,nx)/nx
gk = numpy.fft.fft(g,nx)/nx
hk = numpy.zeros(nx, dtype=complex)

for i in range(0,nx):
    for j in range(0,nx):
        hk[i] += fk[j] * gk[(i-j) % nx]

h = numpy.fft.ifft(hk,nx) * nx
pyplot.subplot(121)
pyplot.plot(x, f, 'o-', label='f(x)')
pyplot.plot(x, g, 'o-', label='g(x)')
pyplot.plot(x, h, 'o-', label='h(x)')
pyplot.axis([0, 2*pi, -2, 2])
pyplot.legend(loc='upper right')
pyplot.subplot(122)
pyplot.plot(k, fk.imag, 'o-', label='imag(f_k)')
pyplot.plot(k, gk.imag, 'o-', label='imag(g_k)')
pyplot.plot(k, hk.real, 'o-', label='real(h_k)')
pyplot.axis([0, nx, -1, 1])
pyplot.legend(loc='upper right')
pyplot.show()
