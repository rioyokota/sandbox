#include <cmath>
#include "types.h"
#include "core.h"

#define for_1 for (int d=0; d<1; d++)
#define for_2 for (int d=0; d<2; d++)

class Kernel {
public:
  int maxLevel;
  int numBodies;
  int numCells;
  int numLeafs;
  int numNeighbors;

  real X0[3];
  real R0;
  real (*Ibodies)[4];
  real (*Ibodies2)[4];
  real (*Jbodies)[4];
  real (*Multipole)[MTERM];
  real (*Local)[LTERM];
  int (*Leafs)[2];

private:
  void P2PSum(int ibegin, int iend, int jbegin, int jend) const {
    for (int i=ibegin; i<iend; i++) {
      real Po = 0, Fx = 0;
      for (int j=jbegin; j<jend; j++) {
	real dist[3];
	for_1 dist[d] = Jbodies[i][d] - Jbodies[j][d];
	real R2 = dist[0] * dist[0];
	real invR2 = R2 == 0 ? 0 : 1.0 / R2;
	real invR = Jbodies[j][3] * sqrt(invR2);
	real invR3 = invR2 * invR;
	Po += invR;
	Fx += dist[0] * invR3;
      }
      Ibodies[i][0] += Po;
      Ibodies[i][1] -= Fx;
    }
  }
  
public:
  void P2P() const {
    int nunit = 1 << maxLevel;
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      int jmin = MAX(0, i - numNeighbors);
      int jmax = MIN(nunit - 1, i + numNeighbors);
      for (int j=jmin; j<=jmax; j++) {
	P2PSum(Leafs[i][0],Leafs[i][1],Leafs[j][0],Leafs[j][1]);
      }
    }
  }

  void P2M() const {
    int levelOffset = ((1 << maxLevel) - 1);
    real R = R0 / (1 << maxLevel);
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      real center = X0[0] - R0 + (2 * i + 1) * R;
      for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
        real dist[3] = {0,0,0};
        for_1 dist[d] = center - Jbodies[j][d];
        real M[MTERM] = {0};
        M[0] = Jbodies[j][3];
        powerM(M,dist);
        for_m Multipole[i+levelOffset][m] += M[m];
      }
    }
  }

  void M2M() const {
    for (int lev=maxLevel; lev>0; lev--) {
      int childOffset = ((1 << lev) - 1);
      int parentOffset = ((1 << (lev - 1)) - 1);
      real radius = R0 / (1 << lev);
#pragma omp parallel for schedule(static, 4)
      for (int i=0; i<(1 << lev); i++) {
        int c = i + childOffset;
        int p = (i >> 1) + parentOffset;
        real dist[3] = {0,0,0};
        dist[0] = (1 - (i & 1) * 2) * radius;
        real C[MTERM] = {0};
        C[0] = 1;
        powerM(C,dist);
        for_m Multipole[p][m] += C[m] * Multipole[c][0];
        M2MSum(Multipole[p],C,Multipole[c]);
      }
    }
  }

  void M2L() const {
    for (int lev=1; lev<=maxLevel; lev++) {
      int levelOffset = ((1 << lev) - 1);
      int nunit = 1 << lev;
      real diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
      for (int i=0; i<(1 << lev); i++) {
        real L[LTERM];
        for_l L[l] = 0;
        int jmin =  MAX(0, (i >> 1) - numNeighbors) << 1;
        int jmax = (MIN((nunit >> 1) - 1, (i >> 1) + numNeighbors) << 1) + 1;
	for (int j=jmin; j<=jmax; j++) {
	  if(j < i-numNeighbors || i+numNeighbors < j) {
	    real dist[3] = {0,0,0};
	    dist[0] = (i - j) * diameter;
	    real invR2 = 1. / (dist[0] * dist[0]);
	    real invR  = sqrt(invR2);
	    real C[LTERM];
	    getCoef(C,dist,invR2,invR);
	    M2LSum(L,C,Multipole[j+levelOffset]);
	  }
	}
        for_l Local[i+levelOffset][l] += L[l];
      }
    }
  }

  void L2L() const {
    for (int lev=1; lev<=maxLevel; lev++) {
      int childOffset = ((1 << lev) - 1);
      int parentOffset = ((1 << (lev - 1)) - 1);
      real radius = R0 / (1 << lev);
#pragma omp parallel for
      for (int i=0; i<(1 << lev); i++) {
        int c = i + childOffset;
        int p = (i >> 1) + parentOffset;
        real dist[3] = {0,0,0};
        for_1 dist[d] = ((i & 1) * 2 - 1) * radius;
        real C[LTERM] = {0};
        C[0] = 1;
        powerL(C,dist);
        for_l Local[c][l] += Local[p][l];
        for (int l=1; l<LTERM; l++) Local[c][0] += C[l] * Local[p][l];
        L2LSum(Local[c],C,Local[p]);
      }
    }
  }

  void L2P() const {
    int levelOffset = ((1 << maxLevel) - 1);
    real R = R0 / (1 << maxLevel);
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      real center = X0[0] - R0 + (2 * i + 1) * R;
      real L[LTERM];
      for_l L[l] = Local[i+levelOffset][l];
      for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
        real dist[3] = {0,0,0};
        for_1 dist[d] = Jbodies[j][d] - center;
        real C[LTERM] = {0};
        C[0] = 1;
        powerL(C,dist);
        for_2 Ibodies[j][d] += L[d];
        for (int l=1; l<LTERM; l++) Ibodies[j][0] += C[l] * L[l];
        L2PSum(Ibodies[j],C,L);
      }
    }
  }
};
