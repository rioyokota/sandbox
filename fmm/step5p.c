// Step 5. Multi-level

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int getIndex(int iX[2], int level){
  int d, l, index = 0;
  int jX[2] = {iX[0], iX[1]};
  for (l=0; l<level; l++) {
    index += (jX[1] & 1) << (2*l);
    jX[1] >>= 1;
    index += (jX[0] & 1) << (2*l+1);
    jX[0] >>= 1;
  }
  return index;
}

void getIX(int iX[2], int index) {
  int l = 0;
  iX[0] = iX[1] = 0;
  while (index > 0) {
    iX[1] += (index & 1) << l;
    index >>= 1;
    iX[0] += (index & 1) << l;
    index >>= 1;
    l++;
  }
}

int main() {
  int i, j, N = 100;
  int l, level = 3;
  double x[N], y[N], u[N], q[N];
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    u[i] = 0;
    q[i] = 1;
  }
  int nx = 1 << level;
  int Ncell = nx * nx;
  double M[level+1][Ncell], L[level+1][Ncell];
  for (l=0; l<=level; l++) {
    for (i=0; i<Ncell; i++) {
      M[l][i] = L[l][i] = 0;
    }
  }
  double tic = get_time();
  // P2M
  int iX[2];
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    j = getIndex(iX, level);
    M[level][j] += q[i];
  }
  double toc = get_time();
  printf("%f\n",toc-tic);
  // M2M
  for (l=level; l>2; l--) {
    nx = 1 << l;
    Ncell = nx * nx;
    for (i=0; i<Ncell; i++) {
      M[l-1][i/4] += M[l][i];
    }
  }
  tic = get_time();
  printf("%f\n",tic-toc);
  // M2L
  int jX[2];
  for (l=2; l<=level; l++) {
    nx = 1 << l;
    Ncell = nx * nx;
    for (i=0; i<Ncell; i++) {
      getIX(iX, i);
      for (j=0; j<Ncell; j++) {
	getIX(jX, j);
	if (abs(iX[0]/2-jX[0]/2) <= 1 && abs(iX[1]/2-jX[1]/2) <= 1) {
	  if (abs(iX[0]-jX[0]) > 1 || abs(iX[1]-jX[1]) > 1) {
	    double dx = (double)(iX[0] - jX[0]) / nx;
	    double dy = (double)(iX[1] - jX[1]) / nx;
	    double r = sqrt(dx * dx + dy * dy);
	    i = getIndex(iX, level);
	    j = getIndex(jX, level);
	    L[l][i] += M[l][j] / r;
	  }
	}
      }
    }
  }
  toc = get_time();
  printf("%f\n",toc-tic);
  // L2L
  for (l=3; l<=level; l++) {
    nx = 1 << l;
    Ncell = nx * nx;
    for (i=0; i<Ncell; i++) {
      L[l][i] += L[l-1][i/4];
    }
  }
  tic = get_time();
  printf("%f\n",tic-toc);
  // L2P
  nx = 1 << level;
  Ncell = nx * nx;
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    j = getIndex(iX, level);
    u[i] += L[level][j];
  }
  toc = get_time();
  printf("%f\n",toc-tic);
  // P2P
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    for (j=0; j<N; j++) {
      jX[0] = x[j] * nx;
      jX[1] = y[j] * nx;
      if (abs(iX[0]-jX[0]) <= 1 && abs(iX[1]-jX[1]) <= 1) {
	double dx = x[i] - x[j];
	double dy = y[i] - y[j];
	double r = sqrt(dx * dx + dy * dy);
	if (r!=0) u[i] += q[j] / r;
      }
    }
  }
  tic = get_time();
  printf("%f\n",tic-toc);
  // Check answer
  for (i=0; i<N; i++) {
    double ui = 0;
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      double r = sqrt(dx * dx + dy * dy);
      if (r != 0) ui += q[j] / r;
    }
    //printf("%d %lf %lf\n", i, u[i], ui);
  }
  toc = get_time();
  printf("%f\n",toc-tic);
}
