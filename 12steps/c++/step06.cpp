extern "C" void convection(double ** u, int nx, int ny, double dx, double dy, double dt, double c) {
  double un[ny][nx];
  for (int i=0; i<nx; i++)
    for (int j=0; j<ny; j++)
      un[j][i] = u[j][i];
  for (int i=1; i<nx; i++)
    for (int j=1; j<ny; j++)
      u[j][i] = un[j][i] -c * dt / dx * (un[j][i] - un[j][i-1]) - c * dt / dy * (un[j][i] - un[j-1][i]);
}
