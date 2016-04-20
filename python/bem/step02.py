import numpy, math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

n = 64
pi = math.pi
x = numpy.zeros(n)
y = numpy.zeros(n)
u = numpy.zeros(n)
bc = numpy.zeros(n)
for i in range(0,n):
    if i < (n/4):
        x[i] = i*8./n
        y[i] = 0
        u[i] = 0 
        bc[i] = 1
    elif i < n/2+1:
        x[i] = 2
        y[i] = (i-n/4)*4./n
        u[i] = (i-n/4)*4./n
        bc[i] = 0
    elif i < 3*n/4:
        x[i] = (3*n/4-i)*8./n
        y[i] = 1
        u[i] = 0
        bc[i] = 1
    else:
        x[i] = 0
        y[i] = (n-i)*4./n
        u[i] = 0
        bc[i] = 0
ip1 = numpy.arange(n)
ip1 += 1
ip1[n-1] = 0

xm = 0.5*(x+x[ip1])
ym = 0.5*(y+y[ip1])
dx = x[ip1]-x
dy = y[ip1]-y
d = numpy.zeros(n)
for i in range(0,n):
    d[i] = math.sqrt(dx[i]*dx[i]+dy[i]*dy[i])

G = numpy.zeros((n,n))
H = numpy.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if i !=j:
            rx = xm[i]-xm[j]
            ry = ym[i]-ym[j]
            r = math.sqrt(rx*rx+ry*ry)
            G[i,j] = -math.log(r)*d[j]/2/pi
            H[i,j] = (rx*dy[j]-ry*dx[j])/r/r/2/pi
    G[i,i] = d[i]*(1-math.log(d[i]/2))/2/pi
    H[i,i] = 0.5

for i in range(0,n):
    if bc[i] == 1:
        for j in range(0,n):
            tmp = G[j,i]
            G[j,i] = -H[j,i]
            H[j,i] = -tmp

b = numpy.zeros(n)
for i in range(0,n):
    bb = 0
    for j in range(0,n):
        bb += H[i,j]*u[j]
    b[i] = bb

un = numpy.linalg.solve(G,b)

for i in range(0,n):
    if bc[i] == 1:
        tmp = u[i]
        u[i] = un[i]
        un[i] = tmp

ux = numpy.zeros(n)
for i in range(0,n):
    uxi = 0
    for j in range(1,101,2):
        uxi += 1/(j*pi)**2/math.sinh(2*j*pi)*math.sinh(j*pi*x[i])*math.cos(j*pi*y[i])
    ux[i] = x[i]/4-4*uxi
    
fig = pyplot.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
ax.scatter(x, y, u, c='b')
ax.scatter(x, y, ux, c='r')
ax.set_zlim3d(0, 1)
ax.view_init(elev=40., azim=-130.)
pyplot.show()
