import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nx = 41
ny = 41
dx = 2./(nx-1)
dy = 2./(ny-1)
nt = 50
nit = 50
rho = 1
nu = .01
dPdx = 1
dt = .01
x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,2,ny)
X, Y = numpy.meshgrid(x,y) 
u = numpy.zeros((ny,nx))
v = numpy.zeros((ny,nx))
p = numpy.zeros((ny,nx))
b = numpy.zeros((ny,nx))

fig = pyplot.figure(figsize=(11,7), dpi=100)

for n in range(nt):
    un = u.copy()
    vn = v.copy()
    b[1:-1,1:-1]=rho*(1/dt*((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx)+(v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))-((u[1:-1,2:]-u[1:-1,0:-2])/(2*dx))**2-\
                      2*((u[2:,1:-1]-u[0:-2,1:-1])/(2*dy)*(v[1:-1,2:]-v[1:-1,0:-2])/(2*dx))-((v[2:,1:-1]-v[0:-2,1:-1])/(2*dy))**2)
    for it in range(nit):
        p[:,-1] = p[:,-2]
        p[0,:] = p[1,:]
        p[:,0] = p[:,1]
        p[-1,:] = p[-2,:]
        pn = p.copy()
        p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,0:-2])*dy**2+(pn[2:,1:-1]+pn[0:-2,1:-1])*dx**2)/(2*(dx**2+dy**2))-dx**2*dy**2/(2*(dx**2+dy**2))*b[1:-1,1:-1]
    u[1:-1,1:-1] = un[1:-1,1:-1]-un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])-vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1])-dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,0:-2])+\
                   nu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])+dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]))+dPdx*dt
    v[1:-1,1:-1] = vn[1:-1,1:-1]-un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1])-dt/(2*rho*dy)*(p[2:,1:-1]-p[0:-2,1:-1])+\
                   nu*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])+(dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])))
    u[:,0] = u[:,-1]
    u[0,:] = 0
    u[-1,:] = 0
    v[:,0] = v[:,-1]
    v[0,:] = 0
    v[-1,:] = 0
    pyplot.contourf(X, Y, p, alpha=0.5)
    pyplot.colorbar()
    pyplot.quiver(X[::2,::2], Y[::2,::2], u[::2,::2], v[::2,::2])
    pyplot.xlabel('X')
    pyplot.ylabel('Y');
    pyplot.pause(0.05)
    pyplot.clf()
pyplot.show()
