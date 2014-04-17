#include "vtk.h"

int main() {
  const int N = 10000;
  real_t R0 = 0.5;
  vec3 X0 = 0.5;
  vec3 X;
  vtkPlot vtk;
  vtk.setDomain(R0,X0);
  vtk.setGroup(0,N);
  for (int i=0; i<N; i++) {
    for (int d=0; d<3; d++) X[d] = drand48();
    vtk.setPoints(0,X);
  }
  vtk.setGroup(1,N);
  for (int i=0; i<N; i++) {
    for (int d=0; d<3; d++) X[d] = drand48();
    vtk.setPoints(1,X);
  }
  vtk.plot(2);
}
