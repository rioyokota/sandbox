#include <cassert>
#include <cstdlib>
#include <cstdio>
#include "kernels.h"

#define for_2 for (int d=0; d<2; d++)

class Fmm : public Kernel {
private:
  void sort(real (*bodies)[2], real (*bodies2)[2], int *key) const {
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
      for_2 bodies2[inew][d] = bodies[i][d];
    }
    delete[] bucket;
  }

public:
  void allocate(int NumBodies, int MaxLevel, int NumNeighbors) {
    numBodies = NumBodies;
    maxLevel = MaxLevel;
    numNeighbors = NumNeighbors;
    numCells = ((1 << (maxLevel + 1)) - 1);
    numLeafs = 1 << maxLevel;
    Ibodies = new real [numBodies][2]();
    Ibodies2 = new real [numBodies][2]();
    Jbodies = new real [numBodies][2]();
    Multipole = new real [numCells][PP]();
    Local = new real [numCells][PP]();
    Leafs = new int [numLeafs][2]();
    for (int i=0; i<numCells; i++) {
      for (int n=0; n<PP; n++) {
	Multipole[i][n] = 0;
	Local[i][n] = 0;
      }
    }
    for (int i=0; i<numLeafs; i++) {
      Leafs[i][0] = Leafs[i][1] = 0;
    }
  }

  void deallocate() {
    delete[] Ibodies;
    delete[] Ibodies2;
    delete[] Jbodies;
    delete[] Multipole;
    delete[] Local;
    delete[] Leafs;
  }

  void initBodies(real cycle) {
    R0 = cycle * .5;
    X0 = R0;
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
  }

  void sortBodies() const {
    int *key = new int [numBodies];
    real diameter = 2 * R0 / (1 << maxLevel);
    for (int i=0; i<numBodies; i++) {
      key[i] = int((Jbodies[i][0] + R0 - X0) / diameter);
    }
    sort(Jbodies,Ibodies,key);
    for (int i=0; i<numBodies; i++) {
      for_2 Jbodies[i][d] = Ibodies[i][d];
      for_2 Ibodies[i][d] = 0;
    }
    delete[] key;
  }

  void fillLeafs() const {
    real diameter = 2 * R0 / (1 << maxLevel);
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
  }

  void direct(int numTargets) {
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
  }

  void verify(int numTargets, real & potDif, real & potNrm, real & accDif, real & accNrm) {
    real potSum = 0, potSum2 = 0;
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
  }
};
