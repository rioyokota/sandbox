// Step 3. Morton index

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
  int i, j, N = 10;
  double x[N], y[N], u[N], q[N];
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    u[i] = 0;
    q[i] = 1;
  }
  double M[16], L[16];
  for (i=0; i<16; i++) {
    M[i] = L[i] = 0;
  }
  double tic = get_time();
  // P2M
  int iX[2];
  for (i=0; i<N; i++) {
    iX[0] = x[i] * 4;
    iX[1] = y[i] * 4;
    j = getIndex(iX, 2);
    M[j] += q[i];
  }
  double toc = get_time();
  printf("%f\n",toc-tic);
  // M2L
  int jX[2];
  for (i=0; i<16; i++) {
    getIX(iX, i);
    for (j=0; j<16; j++) {
      getIX(jX, j);
      if (abs(iX[0]-jX[0]) > 1 || abs(iX[1]-jX[1]) > 1) {
	double dx = (iX[0] - jX[0]) / 4.;
	double dy = (iX[1] - jX[1]) / 4.;
	double r = sqrt(dx * dx + dy * dy);
	i = getIndex(iX, 2);
	j = getIndex(jX, 2);
	L[i] += M[j] / r;
      }
    }
  }
  tic = get_time();
  printf("%f\n",tic-toc);
  // L2P
  for (i=0; i<N; i++) {
    iX[0] = x[i] * 4;
    iX[1] = y[i] * 4;
    j = getIndex(iX, 2);
    u[i] += L[j];
  }
  toc = get_time();
  printf("%f\n",toc-tic);
  // P2P
  for (i=0; i<N; i++) {
    iX[0] = x[i] * 4;
    iX[1] = y[i] * 4;
    for (j=0; j<N; j++) {
      jX[0] = x[j] * 4;
      jX[1] = y[j] * 4;
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
