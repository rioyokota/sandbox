import numpy, random
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

n = 10
nt = 50
dt = 1e-2
x = numpy.zeros(n)
y = numpy.zeros(n)
z = numpy.zeros(n)
xo = numpy.zeros(n)
yo = numpy.zeros(n)
zo = numpy.zeros(n)
ax = numpy.zeros(n)
ay = numpy.zeros(n)
az = numpy.zeros(n)

for i in range(n):
    x[i] = random.random()-.5
    y[i] = random.random()-.5
    z[i] = random.random()-.5
    xo[i] = x[i]
    yo[i] = y[i]
    zo[i] = z[i]

fig = pyplot.figure(figsize=(11,7), dpi=100)

for it in range(nt):
    for i in range(n):
        ax[i] = 0
        ay[i] = 0
        az[i] = 0
        for j in range(n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dz = z[i]-z[j]
            if i==j:
                invR2 = 0
            else:
                invR2 = 1/(dx*dx+dy*dy+dz*dz)
            ax[i] -= dx*invR2
            ay[i] -= dy*invR2
            az[i] -= dz*invR2
    for i in range(n):
        tmp = x[i]
        x[i] = 2*x[i]-xo[i]+ax[i]*dt*dt
        xo[i] = tmp
        tmp = y[i]
        y[i] = 2*y[i]-yo[i]+ay[i]*dt*dt
        yo[i] = tmp
        tmp = z[i]
        z[i] = 2*z[i]-zo[i]+az[i]*dt*dt
        zo[i] = tmp
    axes = fig.gca(projection='3d')
    axes.scatter(x, y, z, 'o')
    axes.axis([-1,1,-1,1])
    axes.set_zlim3d(-1,1)
    pyplot.pause(0.05)
    pyplot.clf()
pyplot.show()
