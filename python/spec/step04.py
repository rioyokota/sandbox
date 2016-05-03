import math, numpy
from matplotlib import pyplot

nx = 32
pi = math.pi
dx = 2*pi/(nx-1)
x = numpy.linspace(0,2*pi,nx)
X = numpy.linspace(0,2*pi,1000)
k = numpy.arange(nx)
k[nx/2:] = numpy.arange(-nx/2,0)
f = 2*pi*x-x*x
exact = 2*pi-2*X
fdm = numpy.gradient(f,dx)

fk = numpy.fft.fft(f,nx)/nx
dfk = 1j * k * fk;
spectral = numpy.fft.ifft(dfk,nx)*nx

pyplot.subplot(121)
pyplot.plot(X, exact, '-', label='Exact')
pyplot.plot(x, fdm, 'o-', label='FDM')
pyplot.plot(x, spectral, 'o-', label='Spectral')
pyplot.axis([0, 2*pi, -10, 10])
pyplot.legend(loc='upper right')
pyplot.subplot(122)
pyplot.plot(k, abs(fk), 'o-', label='|f_k|')
pyplot.plot(k, abs(dfk), 'o-', label='|df_k|')
pyplot.axis([-nx/2, nx/2, -10, 10])
pyplot.legend(loc='upper right')
pyplot.show()
