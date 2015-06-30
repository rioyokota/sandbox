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
  inline void getIndex(int *ix, int index) const {
    for_3 ix[d] = 0;
    ix[0] = index;
    /*
    int d = 0, level = 0;
    while (index != 0) {
      ix[d] += (index & 1) * (1 << level);
      index >>= 1;
      d = (d+1) % 3;
      if (d == 0) level++;
    }
    */
  }

  void getCenter(real *dist, int index, int level) const {
    real R = R0 / (1 << level);
    int ix[3] = {0, 0, 0};
    getIndex(ix, index);
    for_1 dist[d] = X0[d] - R0 + (2 * ix[d] + 1) * R;
  }

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
  
protected:
  inline int getKey(int *ix, int level, bool levelOffset=true) const {
    int id = 0;
    if (levelOffset) id = ((1 << level) - 1);
    for_1 id += ix[0];
    return id;
  }

public:
  void P2P() const {
    int nunit = 1 << maxLevel;
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      int ix[3] = {0, 0, 0};
      getIndex(ix,i);
      int jmin = MAX(0, ix[0] - numNeighbors);
      int jmax = MIN(nunit - 1, ix[0] + numNeighbors);
      for (int j=jmin; j<=jmax; j++) {
	P2PSum(Leafs[i][0],Leafs[i][1],Leafs[j][0],Leafs[j][1]);
      }
    }
  }

  void P2M() const {
    int levelOffset = ((1 << maxLevel) - 1);
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      real center[3];
      getCenter(center,i,maxLevel);
      for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
        real dist[3] = {0,0,0};
        for_1 dist[d] = center[d] - Jbodies[j][d];
        real M[MTERM];
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
        int ix[3];
        ix[0] = 1 - (i & 1) * 2;
        real dist[3] = {0,0,0};
        for_1 dist[d] = ix[d] * radius;
        real C[LTERM];
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
      int nmin = 0;
      int nmax = (nunit >> 1) - 1;
#pragma omp parallel for
      for (int i=0; i<(1 << lev); i++) {
        real L[LTERM];
        for_l L[l] = 0;
        int ix[3] = {0,0,0};
        getIndex(ix,i);
        int jxmin[3];
        for_1 jxmin[d] =  MAX(nmin, (ix[d] >> 1) - numNeighbors) << 1;
        int jxmax[3];
        for_1 jxmax[d] = (MIN(nmax, (ix[d] >> 1) + numNeighbors) << 1) + 1;
        int jx[3];
	for (jx[0]=jxmin[0]; jx[0]<=jxmax[0]; jx[0]++) {
	  if(jx[0] < ix[0]-numNeighbors || ix[0]+numNeighbors < jx[0]) {
	    int jxp[3];
	    for_1 jxp[d] = (jx[d] + nunit) % nunit;
	    int j = getKey(jxp,lev);
	    real dist[3] = {0,0,0};
	    for_1 dist[d] = (ix[d] - jx[d]) * diameter;
	    real invR2 = 1. / (dist[0] * dist[0]);
	    real invR  = sqrt(invR2);
	    real C[LTERM];
	    getCoef(C,dist,invR2,invR);
	    M2LSum(L,C,Multipole[j]);
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
        int ix[3];
        ix[0] = (i & 1) * 2 - 1;
        real dist[3] = {0,0,0};
        for_1 dist[d] = ix[d] * radius;
        real C[LTERM];
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
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      real center[3];
      getCenter(center,i,maxLevel);
      real L[LTERM];
      for_l L[l] = Local[i+levelOffset][l];
      for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
        real dist[3] = {0,0,0};
        for_1 dist[d] = Jbodies[j][d] - center[d];
        real C[LTERM];
        C[0] = 1;
        powerL(C,dist);
        for_2 Ibodies[j][d] += L[d];
        for (int l=1; l<LTERM; l++) Ibodies[j][0] += C[l] * L[l];
        L2PSum(Ibodies[j],C,L);
      }
    }
  }
};
