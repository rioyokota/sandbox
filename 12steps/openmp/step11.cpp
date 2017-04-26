#include <cmath>
#include <cstdio>
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}

extern "C" void navierstokes(double ** u, double ** v, double ** p, int nx, int ny, double dx, double dy, int nit, double rho, double nu, double dt) {
  double tic = get_time();
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
  double toc = get_time();
  printf("%-20s : %lf s\n","Setup", toc-tic);
  tic = get_time();
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
    double diff = 0, norm = 0;
    for (int i=1; i<nx-1; i++) {
      for (int j=1; j<ny-1; j++) {
        p[j][i] = ((pn[j][i+1] + pn[j][i-1]) * dy * dy + (pn[j+1][i] + pn[j-1][i]) * dx * dx) / (2 * (dx * dx + dy * dy)) - dx * dx * dy * dy / (2 * (dx * dx + dy * dy)) * b[j][i];
        diff += (p[j][i] - pn[j][i]) * (p[j][i] - pn[j][i]);
        norm += p[j][i] * p[j][i];
      }
    }
    printf("%d %lf\n",it, sqrtf(diff / norm));
    //if (sqrtf(diff / norm) < 1e-5) break;
  }
  toc = get_time();
  printf("%-20s : %lf s\n","Poisson", toc-tic);
  tic = get_time();
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
  toc = get_time();
  printf("%-20s : %lf s\n","Navier-Stokes", toc-tic);
}

int main(int argc, char ** argv) {
  int nx = 41;
  int ny = 41;
  int nt = 10;
  int nit = 500;
  double dx = 2./(nx-1);
  double dy = 2./(ny-1);
  double rho = 1;
  double nu = .01;
  double dt = .01;
  double ** u = new double * [ny];
  double ** v = new double * [ny];
  double ** p = new double * [ny];
  for (int j=0; j<ny; j++) {
    u[j] = new double [nx];
    v[j] = new double [nx];
    p[j] = new double [nx];
    for (int i=0; i<nx; i++)
      u[j][i] = v[j][i] = p[j][i] = 0;
  }
  for (int it=0; it<nt; it++) {
    printf("%d\n",it);
    navierstokes(u, v, p, nx, ny, dx, dy, nit, rho, nu, dt);
  }
  for (int j=0; j<ny; j++) {
    delete[] u[j];
    delete[] v[j];
    delete[] p[j];
  }
  delete[] u;
  delete[] v;
  delete[] p;
}
