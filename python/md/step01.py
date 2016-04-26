import numpy, math
from matplotlib import pyplot

n = 2
nt = 40
dt = 0.2
x = numpy.zeros(n)
y = numpy.zeros(n)
vx = numpy.zeros(n)
vy = numpy.zeros(n)
ax = numpy.zeros(n)
ay = numpy.zeros(n)

for i in range(n):
    x[i] = i-.5
    y[i] = 0
    vx[i] = 0
    vy[i] = (1-2*i)/math.sqrt(2)

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
        vx[i] += ax[i]*dt;
        vy[i] += ay[i]*dt;
        x[i] += vx[i]*dt
        y[i] += vy[i]*dt
    if it == nt/2: dt = -dt
    pyplot.plot(x, y, 'o')
    pyplot.axis([-1,1,-1,1])
    pyplot.gca().add_artist(pyplot.Circle((0,0),.5,color='b',fill=False))
    pyplot.gca().set_aspect('equal', adjustable='box')
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
