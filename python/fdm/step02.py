import numpy
from matplotlib import pyplot

nx = 41
dx = 2./(nx-1)
nt = 50
dt = .01
x = numpy.linspace(0,2,nx)
u = numpy.ones(nx)
u[10:20] = 2

for n in range(nt):
    un = u.copy()
    for i in range(1,nx-1):
        u[i] = un[i]-un[i]*dt/dx*(un[i]-un[i-1])
    pyplot.plot(x, u)
    pyplot.axis([0, 2, .5, 2.5])
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
