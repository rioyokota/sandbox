// Step 12. High order kernels

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <tgmath.h>

int P = 6;

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

void P2M(double complex *M, double q, double dx, double dy) {
  double complex Z = dx + I * dy;
  double complex powZ = 1;
  int n;  
  M[0] += q;
  for (n=1; n<P; n++) {
    powZ *= Z / n;
    M[n] += powZ * q;
  }
}

void M2M(double complex *Mi, double complex *Mj, double dx, double dy) {
  double complex Z = dx + I * dy;
  int k, n;
  for (k=0; k<P; k++) {
    double complex powZ = 1;
    Mi[k] += Mj[k];
    for (n=1; n<=k; n++) {
      powZ *= Z / n;
      Mi[k] += Mj[k-n] * powZ;
    }
  }
}

void M2L(double complex *L, double complex *M, double dx, double dy) {
  double complex Z = dx + I * dy;
  double complex powZn = 1;
  double complex powZnk = 1;
  double complex invZ = powZn / Z;
  L[0] += -M[0] * log(Z);
  if (P>1) L[0] += M[1] * invZ;
  powZn = invZ;
  int k, n;
  for (k=2; k<P; k++) {
    powZn *= (k-1) * invZ;
    L[0] += M[k] * powZn;
  }
  if (P>1) L[1] += -M[0] * invZ;
  powZn = invZ;
  for (k=1; k<P; k++) {
    powZn *= k * invZ;
    L[1] += -M[k] * powZn;
  }
  double Cnk = -1;
  for (n=2; n<P; n++) {
    Cnk *= -1;
    powZnk *= invZ;
    powZn = Cnk * powZnk;
    for (k=0; k<P; k++) {
      powZn *= (n+k-1) * invZ;
      L[n] += M[k] * powZn;
    }
    powZnk *= n-1;
  }
}

void L2L(double complex *Li, double complex *Lj, double dx, double dy) {
  double complex Z = dx + I * dy;
  int k, n;
  for (n=0; n<P; n++) {
    double complex powZ = 1;
    Li[n] += Lj[n];
    for (k=1; k<P-n; k++) {
      powZ *= Z / k;
      Li[n] += Lj[n+k] * powZ;
    }
  }
}

void L2P(double *u, double complex *L, double dx, double dy) {
  double complex Z = dx + I * dy;
  double complex powZ = 1;
  *u += creal(L[0]);
  int n;
  for (n=1; n<P; n++) {
    powZ *= Z / n;
    *u += creal(L[n] * powZ);
  }
}

void P2P(double *u, double q, double dx, double dy) {
  double r = sqrt(dx * dx + dy * dy);
  if (r!=0) *u -= q * log(r);
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
  }
  bucketSort(N, icell, x, y);
  int Ncell[level+1];
  Ncell[level] = 0;
  int ic = -1;
  int cells[level+1][1 << (2*level)];
  int cells2[level+1][1 << (2*level)];
  int offset[1 << (2*level)];
  for (l=0; l<=level; l++) {
    for (i=0; i<(1 << (2*level)); i++) {
      cells2[l][i] = -1;
    }
  }
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    icell[i] = getIndex(iX, level);
    if (ic != icell[i]) {
      ic = icell[i];
      cells[level][Ncell[level]] = ic;
      cells2[level][ic] = Ncell[level];
      offset[Ncell[level]] = i;
      Ncell[level]++;
    }
  }
  for (l=level; l>0; l--) {
    ic = -1;
    Ncell[l-1] = 0;
    for (i=0; i<Ncell[l]; i++) {
      if (ic != cells[l][i]/4) {
	ic = cells[l][i]/4;
	cells[l-1][Ncell[l-1]] = ic;
	cells2[l-1][ic] = Ncell[l-1];
	Ncell[l-1]++;
      }
    }
  }
  offset[Ncell[level]] = N;
  int levelOffset[level+1];
  levelOffset[0] = 0;
  for (l=0; l<level; l++) {
    levelOffset[l+1] = levelOffset[l] + Ncell[l];
  }
  double complex M[Ncell[level]+levelOffset[level]][P], L[Ncell[level]+levelOffset[level]][P];
  for (i=0; i<Ncell[level]+levelOffset[level]; i++) {
    for (j=0; j<P; j++) {
      M[i][j] = L[i][j] = 0;
    }
  }
  // P2M
  for (i=0; i<Ncell[level]; i++) {
    nx = 1 << level;
    getIX(iX, cells[level][i]);
    for (j=offset[i]; j<offset[i+1]; j++) {
      double dx = x[j] - (iX[0] + 0.5) / nx;
      double dy = y[j] - (iX[1] + 0.5) / nx;
      P2M(M[i+levelOffset[level]],q[j],dx,dy);
    }
  }
  // M2M
  for (l=level; l>2; l--) {
    nx = 1 << l;
    for (i=0; i<Ncell[level]; i++) {
      j = cells[l][i];
      getIX(iX, j);
      if (cells2[l][j]>=0) {
	double dx = (iX[0] + 0.5) / nx - (iX[0]/2 + 0.5) / (nx / 2);
	double dy = (iX[1] + 0.5) / nx - (iX[1]/2 + 0.5) / (nx / 2);
	M2M(M[cells2[l-1][j/4]+levelOffset[l-1]],M[i+levelOffset[l]],-dx,-dy);
      }
    }
  }
  // M2L
  int jc, jX[2];
  for (l=2; l<=level; l++) {
    nx = 1 << l;
    for (ic=0; ic<Ncell[l]; ic++) {
      getIX(iX, cells[l][ic]);
      for (jc=0; jc<Ncell[l]; jc++) {
	getIX(jX, cells[l][jc]);
	if (abs(iX[0]/2-jX[0]/2) <= 1 && abs(iX[1]/2-jX[1]/2) <= 1) {
	  if (abs(iX[0]-jX[0]) > 1 || abs(iX[1]-jX[1]) > 1) {
	    double dx = (double)(iX[0] - jX[0]) / nx;
	    double dy = (double)(iX[1] - jX[1]) / nx;
	    i = getIndex(iX, level);
	    j = getIndex(jX, level);
	    if (cells2[l][i]>=0&&cells2[l][j]>=0)
	      M2L(L[cells2[l][i]+levelOffset[l]],M[cells2[l][j]+levelOffset[l]],dx,dy);
	  }
	}
      }
    }
  }
  // L2L
  for (l=3; l<=level; l++) {
    for (i=0; i<Ncell[level]; i++) {
      j = cells[l][i];
      getIX(iX, j);
      if (cells2[l][j]>=0) {
	double dx = (iX[0] + 0.5) / nx - (iX[0]/2 + 0.5) / (nx / 2);
	double dy = (iX[1] + 0.5) / nx - (iX[1]/2 + 0.5) / (nx / 2);
	L2L(L[i+levelOffset[l]],L[cells2[l-1][j/4]+levelOffset[l-1]],dx,dy);
      }
    }
  }
  // L2P
  for (i=0; i<Ncell[level]; i++) {
    getIX(iX, cells[level][i]);
    for (j=offset[i]; j<offset[i+1]; j++) {
      double dx = x[j] - (iX[0] + 0.5) / nx;
      double dy = y[j] - (iX[1] + 0.5) / nx;
      L2P(&u[j],L[i+levelOffset[level]],dx,dy);
    }
  }
  // P2P
  for (ic=0; ic<Ncell[level]; ic++) {
    getIX(iX, cells[level][ic]);
    for (jc=0; jc<Ncell[level]; jc++) {
      getIX(jX, cells[level][jc]);
      if (abs(iX[0]-jX[0]) <= 1 && abs(iX[1]-jX[1]) <= 1) {
	for (i=offset[ic]; i<offset[ic+1]; i++) {
	  for (j=offset[jc]; j<offset[jc+1]; j++) {
	    double dx = x[i] - x[j];
	    double dy = y[i] - y[j];
	    P2P(&u[i],q[j],dx,dy);
	  }
	}
      }
    }
  }
  // Check answer
  for (i=0; i<N; i++) {
    double ui = 0;
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      P2P(&ui,q[j],dx,dy);
    }
    printf("%d %lf %lf\n", i, u[i], ui);
  }
}
