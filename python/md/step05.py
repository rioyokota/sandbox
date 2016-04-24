import numpy, random
from matplotlib import pyplot

n = 10
nt = 50
dt = 2e-3
sigma = 0.4
epsilon = 1e-4
x = numpy.zeros(n)
y = numpy.zeros(n)
xo = numpy.zeros(n)
yo = numpy.zeros(n)
ax = numpy.zeros(n)
ay = numpy.zeros(n)

for i in range(n):
    x[i] = random.random()-.5
    y[i] = random.random()-.5
    xo[i] = x[i]
    yo[i] = y[i]

for it in range(nt):
    for i in range(n):
        ax[i] = 0
        ay[i] = 0
        for j in range(n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            R2 = dx*dx+dy*dy
            if R2 < 0.01:
                invR2 = 0
            else:
                invR2 = 1/R2
            invR2s = sigma*sigma*invR2
            invR6 = invR2s*invR2s*invR2s
            tmp = epsilon*invR2s*invR6*(2*invR6-1)            
            ax[i] -= dx*(tmp+invR2)
            ay[i] -= dy*(tmp+invR2)
    for i in range(n):
        tmp = x[i]
        x[i] = 2*x[i]-xo[i]+ax[i]*dt*dt
        xo[i] = tmp
        tmp = y[i]
        y[i] = 2*y[i]-yo[i]+ay[i]*dt*dt
        yo[i] = tmp
    pyplot.plot(x, y, 'o')
    pyplot.axis([-1,1,-1,1])
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
