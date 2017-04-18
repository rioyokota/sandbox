import ctypes, numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
lib = ctypes.CDLL('./libstep06.so')
lib.convection.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.uintp, ndim=1, flags='C'), ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

nx = 321
ny = 321
dx = 2./(nx-1)
dy = 2./(ny-1)
nt = 10
dt = .0025
c = 1
x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,2,ny)
X, Y = numpy.meshgrid(x,y)
u = numpy.ones((ny,nx))
u[80:160, 80:160] = 2
upp = (u.__array_interface__['data'][0] + numpy.arange(u.shape[0])*u.strides[0]).astype(numpy.uintp)

fig = pyplot.figure(figsize=(11,7), dpi=100)

for n in range(nt):
    lib.convection(upp, nx, ny, dx, dy, dt, c)
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, u, rstride=8, cstride=8, cmap=cm.coolwarm)
    #ax.set_zlim3d(1, 2)
    #pyplot.pause(0.05)
    #pyplot.clf()
#pyplot.show()
