#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <omp.h>

typedef double real;
const int PP = 6;
const int MTERM = PP*(PP+1)*(PP+2)/6;
const int LTERM = (PP+1)*(PP+2)*(PP+3)/6;

#include "core.h"

#define for_3d for( int d=0; d<3; d++ )
#define for_4d for( int d=0; d<4; d++ )
#define for_m for( int m=0; m<MTERM; m++ )
#define for_l for( int l=0; l<LTERM; l++ )
#define FMMMAX(a,b) (((a) > (b)) ? (a) : (b))
#define FMMMIN(a,b) (((a) < (b)) ? (a) : (b))

class Kernel {
public:
  int maxLevel;
  int numBodies;
  int numCells;
  int numLeafs;

  real X0[3];
  real R0;
  int *Index;
  int *Index2;
  int *Rank;
  real (*Ibodies)[4];
  real (*Jbodies)[4];
  real (*Multipole)[MTERM];
  real (*Local)[LTERM];
  real (*globMultipole)[MTERM];
  real (*globLocal)[LTERM];
  int (*Leafs)[2];

private:
  inline void getIndex(int *ix, int index) const {
    for_3d ix[d] = 0;
    int d = 0, level = 0;
    while( index != 0 ) {
      ix[d] += (index % 2) * (1 << level);
      index >>= 1;
      d = (d+1) % 3;
      if( d == 0 ) level++;
    }
  }

  void getCenter(real *dist, int index, int level) const {
    real R = R0 / (1 << level);
    int ix[3] = {0, 0, 0};
    getIndex(ix, index);
    for_3d dist[d] = X0[d] - R0 + (2 * ix[d] + 1) * R;
  }

protected:
  inline int getKey(int *ix, int level, bool levelOffset=true) const {
    int id = 0;
    if( levelOffset ) id = ((1 << 3 * level) - 1) / 7;
    for( int lev=0; lev<level; ++lev ) {
      for_3d id += ix[d] % 2 << (3 * lev + d);
      for_3d ix[d] >>= 1;
    }
    return id;
  }

public:
  void P2P(int ibegin, int iend, int jbegin, int jend) const {
    for( int i=ibegin; i<iend; i++ ) {
      real Po = 0, Fx = 0, Fy = 0, Fz = 0;
      for( int j=jbegin; j<jend; j++ ) {
        real dist[3];
        for_3d dist[d] = Jbodies[i][d] - Jbodies[j][d];
        real R2 = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];
        real invR2 = 1.0 / R2;
        if( R2 == 0 ) invR2 = 0;
        real invR = Jbodies[j][3] * sqrt(invR2);
        real invR3 = invR2 * invR;
        Po += invR;
        Fx += dist[0] * invR3;
        Fy += dist[1] * invR3;
        Fz += dist[2] * invR3;
      }
      Ibodies[i][0] += Po;
      Ibodies[i][1] -= Fx;
      Ibodies[i][2] -= Fy;
      Ibodies[i][3] -= Fz;
    }
  }

  void P2P() const {
    int nunit = 1 << maxLevel;
    int nunitGlob[3];
    for_3d nunitGlob[d] = nunit;
    int nxmin[3], nxmax[3];
    for_3d nxmin[d] = 0;
    for_3d nxmax[d] = nunitGlob[d] + nxmin[d] - 1;
#pragma omp parallel for
    for( int i=0; i<numLeafs; i++ ) {
      int ix[3] = {0, 0, 0};
      getIndex(ix,i);
      int jxmin[3];
      for_3d jxmin[d] = FMMMAX(nxmin[d],ix[d] - 1);
      int jxmax[3];
      for_3d jxmax[d] = FMMMIN(nxmax[d],ix[d] + 1) + 1;
      int jx[3];
      for( jx[2]=jxmin[2]; jx[2]<jxmax[2]; jx[2]++ ) {
        for( jx[1]=jxmin[1]; jx[1]<jxmax[1]; jx[1]++ ) {
          for( jx[0]=jxmin[0]; jx[0]<jxmax[0]; jx[0]++ ) {
            int jxp[3];
            for_3d jxp[d] = (jx[d] + nunit) % nunit;
            int j = getKey(jxp,maxLevel,false);
            int rankOffset = 13 * numLeafs;
            j += rankOffset;
            P2P(Leafs[i+rankOffset][0],Leafs[i+rankOffset][1],Leafs[j][0],Leafs[j][1]);
          }
        }
      }
    }
  }

  void P2M() const {
    int rankOffset = 13 * numLeafs;
    int levelOffset = ((1 << 3 * maxLevel) - 1) / 7 + 13 * numCells;
#pragma omp parallel for
    for( int i=0; i<numLeafs; i++ ) {
      real center[3];
      getCenter(center,i,maxLevel);
      for( int j=Leafs[i+rankOffset][0]; j<Leafs[i+rankOffset][1]; j++ ) {
        real dist[3];
        for_3d dist[d] = center[d] - Jbodies[j][d];
        real M[MTERM];
        M[0] = Jbodies[j][3];
        powerM(M,dist);
        for_m Multipole[i+levelOffset][m] += M[m];
      }
    }
  }

  void M2M() const {
    int rankOffset = 13 * numCells;
    for( int lev=maxLevel; lev>0; lev-- ) {
      int childOffset = ((1 << 3 * lev) - 1) / 7 + rankOffset;
      int parentOffset = ((1 << 3 * (lev - 1)) - 1) / 7 + rankOffset;
      real radius = R0 / (1 << lev);
#pragma omp parallel for schedule(static, 8)
      for( int i=0; i<(1 << 3 * lev); i++ ) {
        int c = i + childOffset;
        int p = (i >> 3) + parentOffset;
        int ix[3];
        ix[0] = 1 - (i & 1) * 2;
        ix[1] = 1 - ((i / 2) & 1) * 2;
        ix[2] = 1 - ((i / 4) & 1) * 2;
        real dist[3];
        for_3d dist[d] = ix[d] * radius;
        real M[MTERM];
        real C[LTERM];
        C[0] = 1;
        powerM(C,dist);
        for_m M[m] = Multipole[c][m];
        for_m Multipole[p][m] += C[m] * M[0];
        M2MSum(Multipole[p],C,M);
      }
    }
  }

  void M2L() const {
    for( int lev=1; lev<=maxLevel; lev++ ) {
      int levelOffset = ((1 << 3 * lev) - 1) / 7;
      int nunit = 1 << lev;
      int nunitGlob[3];
      for_3d nunitGlob[d] = nunit;
      int nxmin[3], nxmax[3];
      for_3d nxmin[d] = 0;
      for_3d nxmax[d] = (nunitGlob[d] >> 1) + nxmin[d] - 1;
      real diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
      for( int i=0; i<(1 << 3 * lev); i++ ) {
        real L[LTERM];
        for_l L[l] = 0;
        int ix[3] = {0, 0, 0};
        getIndex(ix,i);
        int jxmin[3];
        for_3d jxmin[d] =  FMMMAX(nxmin[d],(ix[d] >> 1) - 1)      << 1;
        int jxmax[3];
        for_3d jxmax[d] = (FMMMIN(nxmax[d],(ix[d] >> 1) + 1) + 1) << 1;
        int jx[3];
        for( jx[2]=jxmin[2]; jx[2]<jxmax[2]; jx[2]++ ) {
          for( jx[1]=jxmin[1]; jx[1]<jxmax[1]; jx[1]++ ) {
            for( jx[0]=jxmin[0]; jx[0]<jxmax[0]; jx[0]++ ) {
              if(jx[0] < ix[0]-1 || ix[0]+1 < jx[0] ||
                 jx[1] < ix[1]-1 || ix[1]+1 < jx[1] ||
                 jx[2] < ix[2]-1 || ix[2]+1 < jx[2]) {
                int jxp[3];
                for_3d jxp[d] = (jx[d] + nunit) % nunit;
                int j = getKey(jxp,lev);
                for_3d jxp[d] = (jx[d] + nunit) / nunit;
                int rankOffset = 13 * numCells;
                j += rankOffset;
                real M[MTERM];
                for_m M[m] = Multipole[j][m];
                real dist[3];
                for_3d dist[d] = (ix[d] - jx[d]) * diameter;
                real invR2 = 1. / (dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]);
                real invR  = sqrt(invR2);
                real C[LTERM];
                getCoef(C,dist,invR2,invR);
                M2LSum(L,C,M);
              }
            }
          }
        }
        for_l Local[i+levelOffset][l] += L[l];
      }
    }
  }

  void L2L() const {
    for( int lev=1; lev<=maxLevel; lev++ ) {
      int childOffset = ((1 << 3 * lev) - 1) / 7;
      int parentOffset = ((1 << 3 * (lev - 1)) - 1) / 7;
      real radius = R0 / (1 << lev);
#pragma omp parallel for
      for( int i=0; i<(1 << 3 * lev); i++ ) {
        int c = i + childOffset;
        int p = (i >> 3) + parentOffset;
        int ix[3];
        ix[0] = (i & 1) * 2 - 1;
        ix[1] = ((i / 2) & 1) * 2 - 1;
        ix[2] = ((i / 4) & 1) * 2 - 1;
        real dist[3];
        for_3d dist[d] = ix[d] * radius;
        real C[LTERM];
        C[0] = 1;
        powerL(C,dist);
        for_l Local[c][l] += Local[p][l];
        for( int l=1; l<LTERM; l++ ) Local[c][0] += C[l] * Local[p][l];
        L2LSum(Local[c],C,Local[p]);
      }
    }
  }

  void L2P() const {
    int rankOffset = 13 * numLeafs;
    int levelOffset = ((1 << 3 * maxLevel) - 1) / 7;
#pragma omp parallel for
    for( int i=0; i<numLeafs; i++ ) {
      real center[3];
      getCenter(center,i,maxLevel);
      real L[LTERM];
      for_l L[l] = Local[i+levelOffset][l];
      for( int j=Leafs[i+rankOffset][0]; j<Leafs[i+rankOffset][1]; j++ ) {
        real dist[3];
        for_3d dist[d] = Jbodies[j][d] - center[d];
        real C[LTERM];
        C[0] = 1;
        powerL(C,dist);
        for_4d Ibodies[j][d] += L[d];
        for( int l=1; l<LTERM; l++ ) Ibodies[j][0] += C[l] * L[l];
        L2PSum(Ibodies[j],C,L);
      }
    }
  }

};
