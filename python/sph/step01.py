import math, numpy, random
from matplotlib import pyplot

n = 10
nt = 50
pi = math.pi
dt = 5e-2
h = 0.2
k = 1
x = numpy.zeros(n)
y = numpy.zeros(n)
xo = numpy.zeros(n)
yo = numpy.zeros(n)
for i in range(n):
    x[i] = random.random()
    y[i] = random.random()
    xo[i] = x[i]
    yo[i] = y[i]
m = numpy.ones(n)
rho = numpy.zeros(n)
p = numpy.zeros(n)
ax = numpy.zeros(n)
ay = numpy.zeros(n)

for it in range(nt):
    for i in range(n):
        rhoi = 0
        for j in range(n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r = math.sqrt(dx*dx+dy*dy)
            if r < h:
                W = 315/64/pi/h**9*(h**2-r**2)**3
                rhoi += m[j]*W
        rho[i] = rhoi
        p[i] = k*rho[i]
    for i in range(n):
        axi = 0
        ayi = 0
        for j in range(n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r = math.sqrt(dx*dx+dy*dy)
            if 0 < r and r < h:
                dW = 45/pi/h**6/r*(h-r)**2
                c = (p[i]+p[j])/2*m[j]/rho[i]/rho[j]
                axi += dx*c*dW
                ayi += dy*c*dW
        ax[i] = axi
        ay[i] = ayi
    for i in range(n):
        tmp = x[i]
        x[i] = 2*x[i]-xo[i]+ax[i]*dt*dt
        xo[i] = tmp
        tmp = y[i]
        y[i] = 2*y[i]-yo[i]+ay[i]*dt*dt
        yo[i] = tmp
    pyplot.plot(x, y, 'o')
    pyplot.axis([-1, 2, -1, 2])
    pyplot.pause(.05)
    pyplot.cla()
pyplot.show()
