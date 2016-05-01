import numpy, copy
from matplotlib import pyplot
from scipy import interpolate

nx = 41
dx = 2./(nx-1)
nt = 50
dt = .01
c = 1
x = numpy.linspace(0,2,nx)
xp = copy.deepcopy(x)
u = numpy.ones(nx)
u[10:20] = 2
up = copy.deepcopy(u)

for n in range(nt):
    un = u.copy()
    for i in range(1,nx-1):
        u[i] = un[i]-c*dt/dx*(un[i]-un[i-1])
        xp[i] += c*dt
        if(xp[i] > 2): xp[i] -= 2
    ui = interpolate.interp1d(xp,up)
    pyplot.plot(x, u, x, ui(x))
    pyplot.axis([0, 2, .5, 2.5])
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
