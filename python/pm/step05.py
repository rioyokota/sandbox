import numpy
from matplotlib import pyplot
from scipy import interpolate

nx = 41
dx = 2./(nx-1)
nt = 50
dt = .01
c = 1
x = numpy.linspace(0,2,nx)
xp = x.copy()
u = numpy.ones(nx)
u[10:20] = 2
ui = u.copy()

for n in range(nt):
    fp = interpolate.interp1d(x,ui,kind='nearest')
    up = fp(xp)
    un = u.copy()
    for i in range(1,nx-1):
        u[i] = un[i]-un[i]*dt/dx*(un[i]-un[i-1])
        xp[i] += up[i]*dt
        if(xp[i] > 2): xp[i] -= 2
    fi = interpolate.interp1d(xp,up,kind='nearest')
    ui = fi(x)
    pyplot.plot(x, u, 'o-', label='FDM')
    pyplot.plot(x, ui, 'o-', label='PM')
    pyplot.axis([0, 2, .5, 2.5])
    pyplot.legend(loc='upper right')
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
