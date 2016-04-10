import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nx = 11
ny = 11
dx = 2./(nx-1)
dy = 1./(ny-1)
nit = 50
x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,1,ny)
X, Y = numpy.meshgrid(x,y) 
p = numpy.zeros((ny,nx))
b  = numpy.zeros((ny,nx))
b[5,5]  = 100
fig = pyplot.figure(figsize=(11,7), dpi=100)

for it in range(nit):
    p[0,:] = p[1,:];
    p[-1,:] = p[-2,:];
    p[:,0] = 0
    p[:,-1] = y[:]
    pn = p.copy()
    p[1:-1,1:-1] = (dy**2*(pn[1:-1,2:]+pn[1:-1,:-2])+dx**2*(pn[2:,1:-1]+pn[:-2,1:-1])-b[1:-1,1:-1]*dx**2*dy**2)/(2*(dx**2+dy**2))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, p, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim3d(0, 1)
    ax.view_init(elev=50., azim=-130.)
    pyplot.pause(0.001)
    pyplot.clf()
pyplot.show()
