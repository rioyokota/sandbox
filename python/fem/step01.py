import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nt = 50
k12 = 10
k23 = 2
K = np.matrix(((k12,-k12), (-k12,k12+k23)))

fig = pyplot.figure(figsize=(11,7), dpi=100)

for n in range(nt):
    f = [np.sin(n*0.1*np.pi),0]
    u = np.linalg.solve(K, f)
    pyplot.plot([2+u[0],1+u[1],0],[0,0,0],'o-',linewidth=5,markersize=20)
    pyplot.axis([-.5,2.5,-1,1])
    pyplot.pause(0.05)
    pyplot.clf()
pyplot.show()
