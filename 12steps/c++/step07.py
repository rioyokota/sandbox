import ctypes, numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
lib = ctypes.CDLL('./libstep07.so')
lib.navierstokes.argtypes = [numpy.ctypeslib.ndpointer(dtype=numpy.uintp, ndim=1, flags='C'),
                             numpy.ctypeslib.ndpointer(dtype=numpy.uintp, ndim=1, flags='C'),
                             numpy.ctypeslib.ndpointer(dtype=numpy.uintp, ndim=1, flags='C'),
                             ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                             ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]

nx = 41
ny = 41
dx = 2./(nx-1)
dy = 2./(ny-1)
nt = 50
nit = 50
rho = 1
nu = .01
dt = .01
x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,2,ny)
X, Y = numpy.meshgrid(x,y)
u = numpy.zeros((ny,nx))
v = numpy.zeros((ny,nx))
p = numpy.zeros((ny,nx))
upp = (u.__array_interface__['data'][0] + numpy.arange(u.shape[0])*u.strides[0]).astype(numpy.uintp)
vpp = (v.__array_interface__['data'][0] + numpy.arange(v.shape[0])*v.strides[0]).astype(numpy.uintp)
ppp = (p.__array_interface__['data'][0] + numpy.arange(p.shape[0])*p.strides[0]).astype(numpy.uintp)

fig = pyplot.figure(figsize=(11,7), dpi=100)

for n in range(nt):
    lib.navierstokes(upp, vpp, ppp, nx, ny, dx, dy, nit, rho, nu, dt)
    pyplot.contourf(X, Y, p, alpha=0.5)
    pyplot.colorbar()
    pyplot.quiver(X[::2,::2], Y[::2,::2], u[::2,::2], v[::2,::2])
    pyplot.xlabel('X')
    pyplot.ylabel('Y');
    pyplot.pause(0.05)
    pyplot.clf()
pyplot.show()
