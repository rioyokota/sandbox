import math, numpy, random
from matplotlib import pyplot

ni = 10
nw = 10
nj = ni+nw
nt = 50
pi = math.pi
dt = 5e-2
h = 0.2
k = 1
x = numpy.zeros(nj)
y = numpy.zeros(nj)
xo = numpy.zeros(nj)
yo = numpy.zeros(nj)
for i in range(nj):
    if i < ni:
        x[i] = random.random()
        y[i] = random.random()
    else:
        x[i] = (i-ni+0.0) / nw
        y[i] = 0
    xo[i] = x[i]
    yo[i] = y[i]
m = numpy.ones(nj)
rho = numpy.zeros(nj)
p = numpy.zeros(nj)
ax = numpy.zeros(ni)
ay = numpy.zeros(ni)

for it in range(nt):
    for i in range(nj):
        rhoi = 0
        for j in range(nj):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r = math.sqrt(dx*dx+dy*dy)
            if r < h:
                W = 315/64/pi/h**9*(h**2-r**2)**3
                rhoi += m[j]*W
        rho[i] = rhoi
        p[i] = k*rho[i]
    for i in range(ni):
        axi = 0
        ayi = 0
        for j in range(nj):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r = math.sqrt(dx*dx+dy*dy)
            if 0 < r and r < h:
                dW = 45/pi/h**6/r*(h-r)**2
                c = (p[i]+p[j])/2*m[j]/rho[i]/rho[j]
                axi += dx*c*dW
                ayi += dy*c*dW
        ax[i] = axi
        ay[i] = ayi-2
    for i in range(ni):
        tmp = x[i]
        x[i] = 2*x[i]-xo[i]+ax[i]*dt*dt
        xo[i] = tmp
        tmp = y[i]
        y[i] = 2*y[i]-yo[i]+ay[i]*dt*dt
        yo[i] = tmp
    pyplot.plot(x[:ni], y[:ni], 'bo', x[ni:nj], y[ni:nj], 'go')
    pyplot.axis([-1, 2, -1, 2])
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
