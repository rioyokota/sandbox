import numpy, math
from matplotlib import pyplot

n = 2
nt = 40
dt = 0.2
x = numpy.zeros(n)
y = numpy.zeros(n)
xo = numpy.zeros(n)
yo = numpy.zeros(n)
ax = numpy.zeros(n)
ay = numpy.zeros(n)

for i in range(n):
    x[i] = i-.5
    y[i] = 0
for i in range(n):
    ax[i] = 0
    ay[i] = 0
    for j in range(n):
        dx = x[i]-x[j]
        dy = y[i]-y[j]
        if i==j:
            invR2 = 0
        else:
            invR2 = 1/(dx*dx+dy*dy)
            ax[i] -= dx*invR2
            ay[i] -= dy*invR2
for i in range(n):
    xo[i] = x[i]
    yo[i] = y[i]
    x[i] = xo[i]+.5*ax[i]*dt*dt
    y[i] = yo[i]+(1-2*i)/math.sqrt(2)*dt+.5*ay[i]*dt*dt

for it in range(nt):
    for i in range(n):
        ax[i] = 0
        ay[i] = 0
        for j in range(n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            if i==j:
                invR2 = 0
            else:
                invR2 = 1/(dx*dx+dy*dy)
                ax[i] -= dx*invR2
                ay[i] -= dy*invR2
    for i in range(n):
        tmp = x[i]
        x[i] = 2*x[i]-xo[i]+ax[i]*dt*dt
        xo[i] = tmp
        tmp = y[i]
        y[i] = 2*y[i]-yo[i]+ay[i]*dt*dt
        yo[i] = tmp
    pyplot.plot(x, y, 'o')
    pyplot.axis([-1,1,-1,1])
    pyplot.gca().add_artist(pyplot.Circle((0,0),.5,color='b',fill=False))
    pyplot.gca().set_aspect('equal', adjustable='box')
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
