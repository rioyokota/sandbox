import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nx = 11
ny = 11
dx = 2./(nx-1)
dy = 1./(ny-1)
nit = 500
x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,1,ny)
X, Y = numpy.meshgrid(x,y)
p = numpy.zeros((ny,nx))
fig = pyplot.figure(figsize=(11,7), dpi=100)

for it in range(nit):
    for i in range(0,nx):
        p[0,i] = p[1,i];
        p[ny-1,i] = p[ny-2,i];
    for j in range(0,ny):
        p[j,0] = 0
        p[j,nx-1] = y[j]
    pn = p.copy()
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            p[j,i] = (dy**2*(pn[j,i+1]+pn[j,i-1])+dx**2*(pn[j+1,i]+pn[j-1,i]))/(2*(dx**2+dy**2))
    diff = numpy.linalg.norm(p-pn) / numpy.linalg.norm(p)
    if it % 10 == 0:
        print it, diff
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, p, rstride=1, cstride=1, cmap=cm.coolwarm)
        ax.set_zlim3d(0, 1)
        ax.view_init(elev=50., azim=-130.)
        pyplot.pause(0.05)
        pyplot.clf()
pyplot.show()
