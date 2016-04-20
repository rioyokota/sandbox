import math, numpy
from matplotlib import pyplot

nx = 8
pi = math.pi
dx = 2*pi/nx
x = numpy.linspace(0,2*pi-dx,nx)
X = numpy.linspace(0,2*pi,1000)
k = numpy.arange(nx)
k[nx/2:] = numpy.arange(-nx/2,0)
f = numpy.cos(3*x)
exact = -3*numpy.sin(3*X)
fdm = numpy.gradient(f,dx)

fk = numpy.fft.fft(f,nx)/nx
dfk = 1j * k * fk;
spectral = numpy.fft.ifft(dfk,nx)*nx

pyplot.subplot(121)
pyplot.plot(X, exact, '-', label='Exact')
pyplot.plot(x, fdm, 'o-', label='FDM')
pyplot.plot(x, spectral.real, 'o-', label='Spectral')
pyplot.axis([0, 2*pi, -6, 6])
pyplot.legend(loc='upper right')
pyplot.subplot(122)
pyplot.plot(k, fk.real, 'o-', label='real(f_k)')
pyplot.plot(k, dfk.imag, 'o-', label='imag(df_k)')
pyplot.axis([-nx/2, nx/2, -3, 3])
pyplot.legend(loc='upper right')
pyplot.show()
