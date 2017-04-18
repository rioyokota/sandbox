#include <cstdio>
extern "C" void navierstokes(double ** u, double ** v, double ** p, int nx, int ny, double dx, double dy, int nit, double rho, double nu, double dt) {
  double un[ny][nx];
  double vn[ny][nx];
  double pn[ny][nx];
  double b[ny][nx];
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      un[j][i] = u[j][i];
      vn[j][i] = v[j][i];
    }
  }
  for (int i=1; i<nx-1; i++) {
    for (int j=1; j<ny-1; j++) {
      b[j][i] = rho * (1. / dt * ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) - ((u[j][i+1] - u[j][i-1]) / (2 * dx)) * ((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx)) - ((v[j+1][i] - v[j-1][i]) / (2 * dy)) * ((v[j+1][i] - v[j-1][i]) / (2 * dy)));
    }
  }
  for (int it=0; it<nit; it++) {
    for (int j=0; j<ny; j++)
      p[j][nx-1] = p[j][nx-2];
    for (int i=0; i<nx; i++)
      p[0][i] = p[1][i];
    for (int j=0; j<ny; j++)
      p[j][0] = p[j][1];
    for (int i=0; i<nx; i++)
      p[ny-1][i] = 0;
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
        pn[j][i] = p[j][i];
    for (int i=1; i<nx-1; i++) {
      for (int j=1; j<ny-1; j++) {
        p[j][i] = ((pn[j][i+1] + pn[j][i-1]) * dy * dy + (pn[j+1][i] + pn[j-1][i]) * dx * dx) / (2 * (dx * dx + dy * dy)) - dx * dx * dy * dy / (2 * (dx * dx + dy * dy)) * b[j][i];
      }
    }
  }
  for (int i=1; i<nx-1; i++) {
    for (int j=1; j<ny-1; j++) {
      u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i-1]) - vn[j][i] * dt / dy * (un[j][i] - un[j-1][i]) - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]) + nu * (dt / dx / dx * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]) + dt / dy / dy * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]));
      v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i-1]) - vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i]) - dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]) + nu * (dt / dx / dx * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]) + dt / dy / dy * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]));
    }
  }
  for (int i=0; i<nx; i++)
    u[0][i] = 0;
  for (int j=0; j<ny; j++)
    u[j][0] = 0;
  for (int j=0; j<ny; j++)
    u[j][nx-1] = 0;
  for (int i=0; i<nx; i++)
    u[ny-1][i] = 1;
  for (int i=0; i<nx; i++)
    v[0][i] = 0;
  for (int i=0; i<nx; i++)
    v[ny-1][i] = 0;
  for (int j=0; j<ny; j++)
    v[j][0] = 0;
  for (int j=0; j<ny; j++)
    v[j][nx-1] = 0;
}
