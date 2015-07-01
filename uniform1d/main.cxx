#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>

const int PP = 7;
#define for_2 for (int d=0; d<2; d++)
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  const int numBodies = 10000;
  const int numTargets = 10000;
  const int ncrit = 10;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2) : 0;
  const int numCells = ((1 << (maxLevel + 1)) - 1);
  const int numLeafs = 1 << maxLevel;
  const int numNeighbors = 1;
  const float cycle = 2 * M_PI;

  printf("--- FMM Profiling ----------------\n");
  double tic = get_time();
  float (*Ibodies)[2] = new float [numBodies][2]();
  float (*Ibodies2)[2] = new float [numBodies][2]();
  float (*Jbodies)[2] = new float [numBodies][2]();
  float (*Multipole)[PP] = new float [numCells][PP]();
  float (*Local)[PP] = new float [numCells][PP]();
  int (*Leafs)[2] = new int [numLeafs][2]();
  for (int i=0; i<numCells; i++) {
    for (int n=0; n<PP; n++) {
      Multipole[i][n] = 0;
      Local[i][n] = 0;
    }
  }
  for (int i=0; i<numLeafs; i++) {
    Leafs[i][0] = Leafs[i][1] = 0;
  }
  double toc = get_time();
  printf("Allocate             : %lf s\n",toc-tic);

  float R0 = cycle * .5;
  float X0 = R0;
  srand48(0);
  float average = 0;
  for (int i=0; i<numBodies; i++) {
    Jbodies[i][0] = 2 * R0 * drand48();
    Jbodies[i][1] = (drand48() - .5) / numBodies;
    average += Jbodies[i][1];
  }
  average /= numBodies;
  for (int i=0; i<numBodies; i++) {
    Jbodies[i][1] -= average;
  }
  tic = get_time();
  printf("Init bodies          : %lf s\n",tic-toc);

  int *key = new int [numBodies];
  float diameter = 2 * R0 / (1 << maxLevel);
  for (int i=0; i<numBodies; i++) {
    key[i] = int((Jbodies[i][0] + R0 - X0) / diameter);
  }
  int Imax = key[0];
  int Imin = key[0];
  for( int i=0; i<numBodies; i++ ) {
    Imax = MAX(Imax,key[i]);
    Imin = MIN(Imin,key[i]);
  }
  int numBucket = Imax - Imin + 1;
  int *bucket = new int [numBucket];
  for (int i=0; i<numBucket; i++) bucket[i] = 0;
  for (int i=0; i<numBodies; i++) bucket[key[i]-Imin]++;
  for (int i=1; i<numBucket; i++) bucket[i] += bucket[i-1];
  for (int i=numBodies-1; i>=0; i--) {
    bucket[key[i]-Imin]--;
    int inew = bucket[key[i]-Imin];
    for_2 Ibodies[inew][d] = Jbodies[i][d];
  }
  for (int i=0; i<numBodies; i++) {
    for_2 Jbodies[i][d] = Ibodies[i][d];
    for_2 Ibodies[i][d] = 0;
  }
  delete[] bucket;
  delete[] key;
  toc = get_time();
  printf("Sort bodies          : %lf s\n",toc-tic);

  diameter = 2 * R0 / (1 << maxLevel);
  int ileaf = int((Jbodies[0][0] + R0 - X0) / diameter);
  Leafs[ileaf][0] = 0;
  for (int i=0; i<numBodies; i++) {
    int inew = int((Jbodies[i][0] + R0 - X0) / diameter);
    if (ileaf != inew) {
      Leafs[ileaf][1] = Leafs[inew][0] = i;
      ileaf = inew;
    }
  }
  Leafs[ileaf][1] = numBodies;
  tic = get_time();
  printf("Fill leafs           : %lf s\n",tic-toc);

  int levelOffset = ((1 << maxLevel) - 1);
  float R = R0 / (1 << maxLevel);
#pragma omp parallel for
  for (int i=0; i<numLeafs; i++) {
    float center = X0 - R0 + (2 * i + 1) * R;
    for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
      float dx = center - Jbodies[j][0];
      float M[PP];
      M[0] = Jbodies[j][1];
      for (int n=1; n<PP; n++) {
	M[n] = M[n-1] * dx / n;
      }
      for (int n=0; n<PP; n++) {
	Multipole[i+levelOffset][n] += M[n];
      }
    }
  }
  toc = get_time();
  printf("P2M                  : %lf s\n",toc-tic);

  for (int lev=maxLevel; lev>0; lev--) {
    int childOffset = ((1 << lev) - 1);
    int parentOffset = ((1 << (lev - 1)) - 1);
    float radius = R0 / (1 << lev);
#pragma omp parallel for
    for (int i=0; i<(1 << lev); i++) {
      int c = i + childOffset;
      int p = (i >> 1) + parentOffset;
      float dx = (1 - (i & 1) * 2) * radius;
      float C[PP];
      C[0] = 1;
      for (int n=1; n<PP; n++) {
	C[n] = C[n-1] * dx / n;
      }
      for (int n=0; n<PP; n++) {
	for (int k=0; k<=n; k++) {
	  Multipole[p][n] += C[n-k] * Multipole[c][k];
	}
      }
    }
  }
  tic = get_time();
  printf("M2M                  : %lf s\n",tic-toc);

  for (int lev=1; lev<=maxLevel; lev++) {
    levelOffset = ((1 << lev) - 1);
    int nunit = 1 << lev;
    diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
    for (int i=0; i<(1 << lev); i++) {
      float L[PP];
      for (int n=0; n<PP; n++) {
	L[n] = 0;
      }
      int jmin =  MAX(0, (i >> 1) - numNeighbors) << 1;
      int jmax = (MIN((nunit >> 1) - 1, (i >> 1) + numNeighbors) << 1) + 1;
      for (int j=jmin; j<=jmax; j++) {
	if(j < i-numNeighbors || i+numNeighbors < j) {
	  float dx = (i - j) * diameter;
	  float invR2 = 1. / (dx * dx);
	  float invR  = sqrt(invR2);
	  float C[PP];
	  C[0] = invR;
	  C[1] = -dx * C[0] * invR2;
	  for (int n=2; n<PP; n++) {
	    C[n] = ((1 - 2 * n) * dx * C[n-1] + (1 - n) * C[n-2]) / n * invR2;
	  }
	  float fact = 1;
	  for (int n=1; n<PP; n++) {
	    fact *= n;
	    C[n] *= fact;
	  }
	  for (int k=0; k<PP; k++) {
	    L[0] += Multipole[j+levelOffset][k] * C[k];
	  }
	  for (int n=1; n<PP; n++) {
	    for (int k=0; k<PP-n; k++) {
	      L[n] += Multipole[j+levelOffset][k] * C[n+k];
	    }
	  }
	}
      }
      for (int n=0; n<PP; n++) {
	Local[i+levelOffset][n] += L[n];
      }
    }
  }
  toc = get_time();
  printf("M2L                  : %lf s\n",toc-tic);

  for (int lev=1; lev<=maxLevel; lev++) {
    int childOffset = ((1 << lev) - 1);
    int parentOffset = ((1 << (lev - 1)) - 1);
    float radius = R0 / (1 << lev);
#pragma omp parallel for
    for (int i=0; i<(1 << lev); i++) {
      int c = i + childOffset;
      int p = (i >> 1) + parentOffset;
      float dx = ((i & 1) * 2 - 1) * radius;
      float C[PP];
      C[0] = 1;
      for (int n=1; n<PP; n++) {
	C[n] = C[n-1] * dx / n;
      }
      for (int n=0; n<PP; n++) {
	for (int k=n; k<PP; k++) {
	  Local[c][n] += C[k-n] * Local[p][k];
	}
      }
    }
  }
  tic = get_time();
  printf("L2L                  : %lf s\n",tic-toc);

  levelOffset = ((1 << maxLevel) - 1);
  R = R0 / (1 << maxLevel);
  #pragma omp parallel for
  for (int i=0; i<numLeafs; i++) {
    float center = X0 - R0 + (2 * i + 1) * R;
    float L[PP];
    for (int n=0; n<PP; n++) {
      L[n] = Local[i+levelOffset][n];
    }
    for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
      float dx = Jbodies[j][0] - center;
      float C[PP];
      C[0] = 1;
      for (int n=1; n<PP; n++) {
	C[n] = C[n-1] * dx / n;
      }
      for (int l=0; l<PP; l++) Ibodies[j][0] += C[l] * L[l];
      for (int l=0; l<PP-1; l++) Ibodies[j][1] += C[l] * L[l+1];
    }
  }
  toc = get_time();
  printf("L2P                  : %lf s\n",toc-tic);

  int nunit = 1 << maxLevel;
#pragma omp parallel for
  for (int i=0; i<numLeafs; i++) {
    int jmin = MAX(0, i - numNeighbors);
    int jmax = MIN(nunit - 1, i + numNeighbors);
    for (int j=jmin; j<=jmax; j++) {
      for (int ii=Leafs[i][0]; ii<Leafs[i][1]; ii++) {
	float Po = 0, Fx = 0;
	for (int jj=Leafs[j][0]; jj<Leafs[j][1]; jj++) {
	  float dx = Jbodies[ii][0] - Jbodies[jj][0];
	  float R2 = dx * dx;
	  float invR2 = R2 == 0 ? 0 : 1.0 / R2;
	  float invR = Jbodies[jj][1] * sqrt(invR2);
	  float invR3 = invR2 * invR;
	  Po += invR;
	  Fx += dx * invR3;
	}
	Ibodies[ii][0] += Po;
	Ibodies[ii][1] -= Fx;
      }
    }
  }
  tic = get_time();
  printf("P2P                  : %lf s\n",tic-toc);

  float Ibodies3[2], Jbodies2[2];
  for (int i=0; i<numTargets; i++) {
    for_2 Ibodies3[d] = 0;
    for_2 Jbodies2[d] = Jbodies[i][d];
    for (int j=0; j<numBodies; j++) {
      float dx = Jbodies2[0] - Jbodies[j][0];
      float R2 = dx * dx;
      float invR2 = R2 == 0 ? 0 : 1.0 / R2;
      float invR = Jbodies[j][1] * sqrtf(invR2);
      dx *= invR2 * invR;
      Ibodies3[0] += invR;
      Ibodies3[1] -= dx;
    }
    for_2 Ibodies2[i][d] = Ibodies3[d];
  }
  double potDif = 0, potNrm = 0, accDif = 0, accNrm = 0;
  double potSum = 0, potSum2 = 0;
  for (int i=0; i<numTargets; i++) {
    potSum += Ibodies[i][0] * Jbodies[i][1];
    potSum2 += Ibodies2[i][0] * Jbodies[i][1];
  }
  potDif = (potSum - potSum2) * (potSum - potSum2);
  potNrm = potSum2 * potSum2;
  for (int i=0; i<numTargets; i++) {
    accDif += (Ibodies[i][1] - Ibodies2[i][1]) * (Ibodies[i][1] - Ibodies2[i][1]);
    accNrm += (Ibodies2[i][1] * Ibodies2[i][1]);
  }
  toc = get_time();
  printf("Verify               : %lf s\n",toc-tic);

  delete[] Ibodies;
  delete[] Ibodies2;
  delete[] Jbodies;
  delete[] Multipole;
  delete[] Local;
  delete[] Leafs;
  tic = get_time();
  printf("Deallocate           : %lf s\n",tic-toc);

  printf("--- FMM vs. direct ---------------\n");
  printf("Rel. L2 Error (pot)  : %g\n",std::sqrt(potDif/potNrm));
  printf("Rel. L2 Error (acc)  : %g\n",std::sqrt(accDif/accNrm));
}
