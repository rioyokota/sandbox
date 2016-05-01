import numpy
from matplotlib import pyplot
from scipy import interpolate

nx = 41
dx = 2./(nx-1)
nt = 50
dt = .01
x = numpy.linspace(0,2,nx)
xp = x.copy()
u = numpy.ones(nx)
u[10:20] = 2
up = u.copy()

for n in range(nt):
    un = u.copy()
    for i in range(1,nx-1):
        u[i] = un[i]-un[i]*dt/dx*(un[i]-un[i-1])
        xp[i] += up[i]*dt
        if(xp[i] > 2): xp[i] -= 2
    ui = interpolate.interp1d(xp,up)
    pyplot.plot(x, u, 'o-', label='FDM')
    pyplot.plot(xp, up, 'o-', label='PM')
    pyplot.axis([0, 2, .5, 2.5])
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
