#include <cassert>
#include <cstdlib>
#include <cstdio>
#include "kernels.h"

class Fmm : public Kernel {
private:
  void sort(real (*bodies)[4], real (*bodies2)[4], int *key) const {
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
      for_4 bodies2[inew][d] = bodies[i][d];
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
    Ibodies = new real [numBodies][4]();
    Ibodies2 = new real [numBodies][4]();
    Jbodies = new real [numBodies][4]();
    Multipole = new real [numCells][MTERM]();
    Local = new real [numCells][LTERM]();
    Leafs = new int [numLeafs][2]();
    for (int i=0; i<numCells; i++) {
      for_m Multipole[i][m] = 0;
      for_l Local[i][l] = 0;
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
    int ix[3] = {0, 0, 0};
    R0 = cycle * .5;
    for_1 X0[d] = R0;
    srand48(0);
    real average = 0;
    for (int i=0; i<numBodies; i++) {
      Jbodies[i][0] = 2 * R0 * (drand48() + ix[0]);
      Jbodies[i][1] = 0;
      Jbodies[i][2] = 0;
      Jbodies[i][3] = (drand48() - .5) / numBodies;
      average += Jbodies[i][3];
    }
    average /= numBodies;
    for (int i=0; i<numBodies; i++) {
      Jbodies[i][3] -= average;
    }
  }

  void sortBodies() const {
    int *key = new int [numBodies];
    real diameter = 2 * R0 / (1 << maxLevel);
    int ix[3] = {0, 0, 0};
    for (int i=0; i<numBodies; i++) {
      for_1 ix[d] = int((Jbodies[i][d] + R0 - X0[d]) / diameter);
      key[i] = getKey(ix,maxLevel,false);
    }
    sort(Jbodies,Ibodies,key);
    for (int i=0; i<numBodies; i++) {
      for_4 Jbodies[i][d] = Ibodies[i][d];
      for_4 Ibodies[i][d] = 0;
    }
    delete[] key;
  }

  void fillLeafs() const {
    real diameter = 2 * R0 / (1 << maxLevel);
    int ix[3] = {0, 0, 0};
    for_1 ix[d] = int((Jbodies[0][d] + R0 - X0[d]) / diameter);
    int ileaf = getKey(ix,maxLevel,false);
    Leafs[ileaf][0] = 0;
    for (int i=0; i<numBodies; i++) {
      for_1 ix[d] = int((Jbodies[i][d] + R0 - X0[d]) / diameter);
      int inew = getKey(ix,maxLevel,false);
      if (ileaf != inew) {
        Leafs[ileaf][1] = Leafs[inew][0] = i;
        ileaf = inew;
      }
    }
    Leafs[ileaf][1] = numBodies;
  }

  void direct() {
    real Ibodies3[4], Jbodies2[4], dX[3];
    for (int i=0; i<100; i++) {
      for_4 Ibodies3[d] = 0;
      for_4 Jbodies2[d] = Jbodies[i][d];
      for (int j=0; j<numBodies; j++) {
	for_1 dX[d] = Jbodies2[d] - Jbodies[j][d];
	real R2 = dX[0] * dX[0];
	real invR2 = R2 == 0 ? 0 : 1.0 / R2;
	real invR = Jbodies[j][3] * sqrtf(invR2);
	for_1 dX[d] *= invR2 * invR;
	Ibodies3[0] += invR;
	Ibodies3[1] -= dX[0];
      }
      for_2 Ibodies2[i][d] = Ibodies3[d];
    }
  }

  void verify(int numTargets, real & potDif, real & potNrm, real & accDif, real & accNrm) {
    real potSum = 0, potSum2 = 0;
    for (int i=0; i<numTargets; i++) {
      potSum += Ibodies[i][0] * Jbodies[i][3];
      potSum2 += Ibodies2[i][0] * Jbodies[i][3];
    }
    potDif = (potSum - potSum2) * (potSum - potSum2);
    potNrm = potSum2 * potSum2;
    for (int i=0; i<numTargets; i++) {
      accDif += (Ibodies[i][1] - Ibodies2[i][1]) * (Ibodies[i][1] - Ibodies2[i][1]);
      accNrm += (Ibodies2[i][1] * Ibodies2[i][1]);
    }
  }
};
