#include <cmath>
#include "types.h"
#include "logger.h"

int main() {
  const int numBodies = 10000;
  const int numTargets = 10000;
  const int ncrit = 10;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2) : 0;
  const int numCells = ((1 << (maxLevel + 1)) - 1);
  const int numLeafs = 1 << maxLevel;
  const int numNeighbors = 1;
  const float cycle = 2 * M_PI;

  logger::verbose = true;
  logger::printTitle("FMM Profiling");

  logger::startTimer("Allocate");
  real (*Ibodies)[2] = new real [numBodies][2]();
  real (*Ibodies2)[2] = new real [numBodies][2]();
  real (*Jbodies)[2] = new real [numBodies][2]();
  real (*Multipole)[PP] = new real [numCells][PP]();
  real (*Local)[PP] = new real [numCells][PP]();
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
  logger::stopTimer("Allocate");

  logger::startTimer("Init bodies");
  float R0 = cycle * .5;
  float X0 = R0;
  srand48(0);
  real average = 0;
  for (int i=0; i<numBodies; i++) {
    Jbodies[i][0] = 2 * R0 * drand48();
    Jbodies[i][1] = (drand48() - .5) / numBodies;
    average += Jbodies[i][1];
  }
  average /= numBodies;
  for (int i=0; i<numBodies; i++) {
    Jbodies[i][1] -= average;
  }
  logger::stopTimer("Init bodies");
  
  logger::startTimer("Sort bodies");
  int *key = new int [numBodies];
  real diameter = 2 * R0 / (1 << maxLevel);
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
  logger::stopTimer("Sort bodies");

  logger::startTimer("Fill leafs");
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
  logger::stopTimer("Fill leafs");
  
  logger::startTimer("P2M");
  int levelOffset = ((1 << maxLevel) - 1);
  real R = R0 / (1 << maxLevel);
#pragma omp parallel for
  for (int i=0; i<numLeafs; i++) {
    real center = X0 - R0 + (2 * i + 1) * R;
    for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
      real dx = center - Jbodies[j][0];
      real M[PP];
      M[0] = Jbodies[j][1];
      for (int n=1; n<PP; n++) {
	M[n] = M[n-1] * dx / n;
      }
      for (int n=0; n<PP; n++) {
	Multipole[i+levelOffset][n] += M[n];
      }
    }
  }
  logger::stopTimer("P2M");

  logger::startTimer("M2M");
  for (int lev=maxLevel; lev>0; lev--) {
    int childOffset = ((1 << lev) - 1);
    int parentOffset = ((1 << (lev - 1)) - 1);
    real radius = R0 / (1 << lev);
#pragma omp parallel for
    for (int i=0; i<(1 << lev); i++) {
      int c = i + childOffset;
      int p = (i >> 1) + parentOffset;
      real dx = (1 - (i & 1) * 2) * radius;
      real C[PP];
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
  logger::stopTimer("M2M");

  logger::startTimer("M2L");
  for (int lev=1; lev<=maxLevel; lev++) {
    levelOffset = ((1 << lev) - 1);
    int nunit = 1 << lev;
    diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
    for (int i=0; i<(1 << lev); i++) {
      real L[PP];
      for (int n=0; n<PP; n++) {
	L[n] = 0;
      }
      int jmin =  MAX(0, (i >> 1) - numNeighbors) << 1;
      int jmax = (MIN((nunit >> 1) - 1, (i >> 1) + numNeighbors) << 1) + 1;
      for (int j=jmin; j<=jmax; j++) {
	if(j < i-numNeighbors || i+numNeighbors < j) {
	  real dx = (i - j) * diameter;
	  real invR2 = 1. / (dx * dx);
	  real invR  = sqrt(invR2);
	  real C[PP];
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
  logger::stopTimer("M2L");

  logger::startTimer("L2L");
  for (int lev=1; lev<=maxLevel; lev++) {
    int childOffset = ((1 << lev) - 1);
    int parentOffset = ((1 << (lev - 1)) - 1);
    real radius = R0 / (1 << lev);
#pragma omp parallel for
    for (int i=0; i<(1 << lev); i++) {
      int c = i + childOffset;
      int p = (i >> 1) + parentOffset;
      real dx = ((i & 1) * 2 - 1) * radius;
      real C[PP];
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
  logger::stopTimer("L2L");

  logger::startTimer("L2P");
  levelOffset = ((1 << maxLevel) - 1);
  R = R0 / (1 << maxLevel);
  #pragma omp parallel for
  for (int i=0; i<numLeafs; i++) {
    real center = X0 - R0 + (2 * i + 1) * R;
    real L[PP];
    for (int n=0; n<PP; n++) {
      L[n] = Local[i+levelOffset][n];
    }
    for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
      real dx = Jbodies[j][0] - center;
      real C[PP];
      C[0] = 1;
      for (int n=1; n<PP; n++) {
	C[n] = C[n-1] * dx / n;
      }
      for (int l=0; l<PP; l++) Ibodies[j][0] += C[l] * L[l];
      for (int l=0; l<PP-1; l++) Ibodies[j][1] += C[l] * L[l+1];
    }
  }
  logger::stopTimer("L2P");

  logger::startTimer("P2P");
  int nunit = 1 << maxLevel;
#pragma omp parallel for
  for (int i=0; i<numLeafs; i++) {
    int jmin = MAX(0, i - numNeighbors);
    int jmax = MIN(nunit - 1, i + numNeighbors);
    for (int j=jmin; j<=jmax; j++) {
      for (int ii=Leafs[i][0]; ii<Leafs[i][1]; ii++) {
	real Po = 0, Fx = 0;
	for (int jj=Leafs[j][0]; jj<Leafs[j][1]; jj++) {
	  real dx = Jbodies[ii][0] - Jbodies[jj][0];
	  real R2 = dx * dx;
	  real invR2 = R2 == 0 ? 0 : 1.0 / R2;
	  real invR = Jbodies[jj][1] * sqrt(invR2);
	  real invR3 = invR2 * invR;
	  Po += invR;
	  Fx += dx * invR3;
	}
	Ibodies[ii][0] += Po;
	Ibodies[ii][1] -= Fx;
      }
    }
  }
  logger::stopTimer("P2P");

  logger::startTimer("Verify");
  real Ibodies3[2], Jbodies2[2];
  for (int i=0; i<numTargets; i++) {
    for_2 Ibodies3[d] = 0;
    for_2 Jbodies2[d] = Jbodies[i][d];
    for (int j=0; j<numBodies; j++) {
      real dx = Jbodies2[0] - Jbodies[j][0];
      real R2 = dx * dx;
      real invR2 = R2 == 0 ? 0 : 1.0 / R2;
      real invR = Jbodies[j][1] * sqrtf(invR2);
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
  logger::stopTimer("Verify");

  logger::startTimer("Deallocate");
  delete[] Ibodies;
  delete[] Ibodies2;
  delete[] Jbodies;
  delete[] Multipole;
  delete[] Local;
  delete[] Leafs;
  logger::stopTimer("Deallocate");

  logger::printTitle("FMM vs. direct");
  logger::printError("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
  logger::printError("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
}
