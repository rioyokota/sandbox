#include "types.h"
#include "vtk.h"

int main() {
  const int numBodies = 10000;
  real_t R0 = 0.5;
  vec3 X0 = 0.5;
  Bodies bodies(numBodies);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    for (int d=0; d<3; d++) B->X[d] = drand48();
    B->IBODY = (B-bodies.begin()) / (numBodies / 4);
  }
  vtk3DPlot vtk;
  vtk.setBounds(R0,X0);
  vtk.setGroupOfPoints(bodies);
  vtk.plot();
}
