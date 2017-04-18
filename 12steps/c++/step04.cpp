extern "C" void convection(double * u, int nx, double dx, double dt, double c) {
  double un[nx];
  for (int i=0; i<nx; i++)
    un[i] = u[i];
  for (int i=1; i<nx; i++)
    u[i] = un[i] - c * dt / dx * (un[i] - un[i-1]);
}
