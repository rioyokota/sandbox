// Step 7. Sort bodies

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void bucketSort(int N, int * index, double * x, double * y) {
  int i, iMax = index[0];
  for (i=1; i<N; i++) {
    if (iMax < index[i]) iMax = index[i];
  }
  iMax++;
  int * bucket = malloc(iMax*sizeof(int));
  double x2[N], y2[N];
  for (i=0; i<iMax; i++) bucket[i] = 0;
  for (i=0; i<N; i++) bucket[index[i]]++;
  for (i=1; i<iMax; i++) bucket[i] += bucket[i-1];
  for (i=N-1; i>=0; i--) {
    bucket[index[i]]--;
    int inew = bucket[index[i]];
    x2[inew] = x[i];
    y2[inew] = y[i];
  }
  for (i=0; i<N; i++) {
    x[i] = x2[i];
    y[i] = y2[i];
  }
  free(bucket);
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
  int iX[2], icell[N];
  int nx = 1 << level;
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    icell[i] = getIndex(iX, level);
    printf("%d %d %lf %lf\n",i,icell[i],x[i],y[i]);
  }
  bucketSort(N, icell, x, y);
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    icell[i] = getIndex(iX, level);
    printf("%d %d %lf %lf\n",i,icell[i],x[i],y[i]);
  }
  int levelOffset[level+1];
  levelOffset[0] = 0;
  for (l=0; l<level; l++) {
    levelOffset[l+1] = levelOffset[l] + (1 << (2*l));
  }
  int Ncell = nx * nx;
  double M[Ncell+levelOffset[level]], L[Ncell+levelOffset[level]];
  for (i=0; i<Ncell+levelOffset[level]; i++) {
    M[i] = L[i] = 0;
  }
  // P2M
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    j = getIndex(iX, level);
    M[j+levelOffset[level]] += q[i];
  }
  // M2M
  for (l=level; l>2; l--) {
    nx = 1 << l;
    Ncell = nx * nx;
    for (i=0; i<Ncell; i++) {
      M[i/4+levelOffset[l-1]] += M[i+levelOffset[l]];
    }
  }
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
	    L[i+levelOffset[l]] += M[j+levelOffset[l]] / r;
	  }
	}
      }
    }
  }
  // L2L
  for (l=3; l<=level; l++) {
    nx = 1 << l;
    Ncell = nx * nx;
    for (i=0; i<Ncell; i++) {
      L[i+levelOffset[l]] += L[i/4+levelOffset[l-1]];
    }
  }
  // L2P
  nx = 1 << level;
  Ncell = nx * nx;
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    j = getIndex(iX, level);
    u[i] += L[j+levelOffset[level]];
  }
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
  // Check answer
  for (i=0; i<N; i++) {
    double ui = 0;
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      double r = sqrt(dx * dx + dy * dy);
      if (r != 0) ui += q[j] / r;
    }
    printf("%d %lf %lf\n", i, u[i], ui);
  }
}
