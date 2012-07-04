#ifndef kernel_h
#define kernel_h
#include "types.h"

__device__ void P2P(Cell *Ci, Cell *Cj, vec4 *Ibodies, vec4 *Jbodies) {
  for( int bi=Ci->LEAF; bi<Ci->LEAF+Ci->NDLEAF; ++bi ) {
    real P0 = 0;
    vec3 F0 = 0;
    for( int bj=Cj->LEAF; bj<Cj->LEAF+Cj->NDLEAF; ++bj ) {
      vec3 dX;
      for( int d=0; d<3; d++ ) dX[d] = Jbodies[bi][d] - Jbodies[bj][d];
      real R2 = norm(dX) + EPS2;
      real invR2 = 1.0 / R2;
      if( R2 == 0 ) invR2 = 0;
      real invR = Jbodies[bi][3] * Jbodies[bj][3] * std::sqrt(invR2);
      dX *= invR2 * invR;
      P0 += invR;
      F0 += dX;
    }
    Ibodies[bi][0] += P0;
    Ibodies[bi][1] -= F0[0];
    Ibodies[bi][2] -= F0[1];
    Ibodies[bi][3] -= F0[2];
  }
}

__device__ void M2L(Cell *Ci, Cell *Cj, Cell *Cells, vecM *Multipole, vecL *Local) {
  int ci = Ci - Cells;
  int cj = Cj - Cells;
  vec3 dist = Ci->X - Cj->X;
  real invR2 = 1 / norm(dist);
  real invR  = Multipole[ci][0] * Multipole[cj][0] * std::sqrt(invR2);
  invR2 = -invR2;
  real invR3 = invR * invR2;
  real invR5 = 3 * invR3 * invR2;
  real invR7 = 5 * invR5 * invR2;
  vecL C;
  C[0] = invR;
  C[1] = dist[0] * invR3;
  C[2] = dist[1] * invR3;
  C[3] = dist[2] * invR3;
  real t = dist[0] * invR5;
  C[4] = dist[0] * t + invR3;
  C[5] = dist[1] * t;
  C[6] = dist[2] * t;
  t = dist[1] * invR5;
  C[7] = dist[1] * t + invR3;
  C[8] = dist[2] * t;
  C[9] = dist[2] * dist[2] * invR5 + invR3;
  t = dist[0] * dist[0] * invR7;
  C[10] = dist[0] * (t + 3 * invR5);
  C[11] = dist[1] * (t +     invR5);
  C[12] = dist[2] * (t +     invR5);
  t = dist[1] * dist[1] * invR7;
  C[13] = dist[0] * (t +     invR5);
  C[16] = dist[1] * (t + 3 * invR5);
  C[17] = dist[2] * (t +     invR5);
  t = dist[2] * dist[2] * invR7;
  C[15] = dist[0] * (t +     invR5);
  C[18] = dist[1] * (t +     invR5);
  C[19] = dist[2] * (t + 3 * invR5);
  C[14] = dist[0] * dist[1] * dist[2] * invR7;
  vecM M = Multipole[cj];
  vecL L = C;
  L[0] += C[4] *M[1] + C[5] *M[2] + C[6] *M[3] + C[7] *M[4] + C[8] *M[5] + C[9] *M[6];
  L[1] += C[10]*M[1] + C[11]*M[2] + C[12]*M[3] + C[13]*M[4] + C[14]*M[5] + C[15]*M[6];
  L[2] += C[11]*M[1] + C[13]*M[2] + C[14]*M[3] + C[16]*M[4] + C[17]*M[5] + C[18]*M[6];
  L[3] += C[12]*M[1] + C[14]*M[2] + C[15]*M[3] + C[17]*M[4] + C[18]*M[5] + C[19]*M[6];
  Local[ci] += L; 
}

class Kernel {
protected:
  vec3 X0;
  real R0;

  cudaVec<vec4> Ibodies;
  cudaVec<vec4> Jbodies;
  cudaVec<Cell> Cells;
  cudaVec<vecM> Multipole;
  cudaVec<vecL> Local;

public:
  Kernel() : X0(0), R0(0) {}
  ~Kernel() {}

  void DIRECT(Cell *Ci, Cell *Cj) const {
    for( int bi=Ci->LEAF; bi<Ci->LEAF+Ci->NDLEAF; ++bi ) {
      real P0 = 0;
      vec3 F0 = 0;
      for( int bj=Cj->LEAF; bj<Cj->LEAF+Cj->NDLEAF; ++bj ) {
        vec3 dX;
        for( int d=0; d<3; d++ ) dX[d] = Jbodies[bi][d] - Jbodies[bj][d];
        real R2 = norm(dX) + EPS2;
        real invR2 = 1.0 / R2;
        if( R2 == 0 ) invR2 = 0;
        real invR = Jbodies[bi][3] * Jbodies[bj][3] * std::sqrt(invR2);
        dX *= invR2 * invR;
        P0 += invR;
        F0 += dX;
      }
      Ibodies[bi][0] += P0;
      Ibodies[bi][1] -= F0[0];
      Ibodies[bi][2] -= F0[1];
      Ibodies[bi][3] -= F0[2];
    }
  }

  void P2M(Cell *C) const {
    real Rmax = 0;
    int c = C - Cells.host();
    for( int b=C->LEAF; b<C->LEAF+C->NCLEAF; b++ ) {
      vec3 dist;
      for( int d=0; d<3; d++ ) dist[d] = C->X[d] - Jbodies[b][d];
      real R = std::sqrt(norm(dist));
      if( R > Rmax ) Rmax = R;
      real M = Jbodies[b][3];
      real tmp = M * dist[0];
      Multipole[c][0] += M;
      Multipole[c][1] += dist[0] * tmp / 2;
      Multipole[c][2] += dist[1] * tmp;
      Multipole[c][3] += dist[2] * tmp;
      tmp = M * dist[1];
      Multipole[c][4] += dist[1] * tmp / 2;
      Multipole[c][5] += dist[2] * tmp;
      Multipole[c][6] += M * dist[2] * dist[2] / 2;
    }
    C->RCRIT = std::min(C->R,Rmax);
  }

  void M2M(Cell *Ci) const {
    real Rmax = 0;
    int ci = Ci - Cells.host();
    for( int cj=Ci->CHILD; cj<Ci->CHILD+Ci->NCHILD; cj++ ) {
      Cell *Cj = Cells.host() + cj;
      vec3 dist = Ci->X - Cj->X;
      real R = std::sqrt(norm(dist)) + Cj->RCRIT;
      if( R > Rmax ) Rmax = R;
      Multipole[ci] += Multipole[cj];
      real M = Multipole[cj][0];
      real tmp = M * dist[0];
      Multipole[ci][1] += dist[0] * tmp / 2;
      Multipole[ci][2] += dist[1] * tmp;
      Multipole[ci][3] += dist[2] * tmp;
      tmp = M * dist[1];
      Multipole[ci][4] += dist[1] * tmp / 2;
      Multipole[ci][5] += dist[2] * tmp;
      Multipole[ci][6] += M * dist[2] * dist[2] / 2;
    }
    if( Ci->RCRIT == 0 ) Ci->RCRIT = std::min(Ci->R,Rmax);
  }

  void L2L(Cell *Ci) const {
    int ci = Ci - Cells.host();
    int cj = Ci->PARENT;
    Cell *Cj = Cells.host() + cj;
    vec3 dist = Ci->X - Cj->X;
    vecL Li = Local[cj];
    vecL Lj = Li;
    Local[ci] /= Multipole[ci][0];
    vecL C;
    C[0] = Lj[1] *dist[0] + Lj[2] *dist[1] + Lj[3] *dist[2];
    C[1] = Lj[4] *dist[0] + Lj[5] *dist[1] + Lj[6] *dist[2];
    C[2] = Lj[5] *dist[0] + Lj[7] *dist[1] + Lj[8] *dist[2];
    C[3] = Lj[6] *dist[0] + Lj[8] *dist[1] + Lj[9] *dist[2];
    C[4] = Lj[10]*dist[0] + Lj[11]*dist[1] + Lj[12]*dist[2];
    C[5] = Lj[11]*dist[0] + Lj[13]*dist[1] + Lj[14]*dist[2];
    C[6] = Lj[12]*dist[0] + Lj[14]*dist[1] + Lj[15]*dist[2];
    C[7] = Lj[13]*dist[0] + Lj[16]*dist[1] + Lj[17]*dist[2];
    C[8] = Lj[14]*dist[0] + Lj[17]*dist[1] + Lj[18]*dist[2];
    C[9] = Lj[15]*dist[0] + Lj[18]*dist[1] + Lj[19]*dist[2];
    for( int d=0; d<10; d++ ) Li[d] += C[d];
    C[0] = (C[1]*dist[0] + C[2]*dist[1] + C[3]*dist[2]) / 2;
    C[1] = (C[4]*dist[0] + C[5]*dist[1] + C[6]*dist[2]) / 2;
    C[2] = (C[5]*dist[0] + C[7]*dist[1] + C[8]*dist[2]) / 2;
    C[3] = (C[6]*dist[0] + C[8]*dist[1] + C[9]*dist[2]) / 2;
    for( int d=0; d<4; d++ ) Li[d] += C[d];
    Li[0] += (dist[0]*C[1]+dist[1]*C[2]+dist[2]*C[3]) / 3;
    Local[ci] += Li;
  }

  void L2P(Cell *Ci) const {
    for( int b=Ci->LEAF; b<Ci->LEAF+Ci->NCLEAF; b++ ) {
      int ci = Ci - Cells.host();
      vec3 dist;
      for( int d=0; d<3; d++ ) dist[d] = Jbodies[b][d] - Ci->X[d];
      Ibodies[b] /= Jbodies[b][3];
      vecL C, L = Local[ci];
      for( int d=0; d<4; d++ ) Ibodies[b][d] += L[d];
      C[0] = L[1] *dist[0] + L[2] *dist[1] + L[3] *dist[2];
      C[1] = L[4] *dist[0] + L[5] *dist[1] + L[6] *dist[2];
      C[2] = L[5] *dist[0] + L[7] *dist[1] + L[8] *dist[2];
      C[3] = L[6] *dist[0] + L[8] *dist[1] + L[9] *dist[2];
      C[4] = L[10]*dist[0] + L[11]*dist[1] + L[12]*dist[2];
      C[5] = L[11]*dist[0] + L[13]*dist[1] + L[14]*dist[2];
      C[6] = L[12]*dist[0] + L[14]*dist[1] + L[15]*dist[2];
      C[7] = L[13]*dist[0] + L[16]*dist[1] + L[17]*dist[2];
      C[8] = L[14]*dist[0] + L[17]*dist[1] + L[18]*dist[2];
      C[9] = L[15]*dist[0] + L[18]*dist[1] + L[19]*dist[2];
      for( int d=0; d<4; d++ ) Ibodies[b][d] += C[d];
      C[0] = (C[1]*dist[0] + C[2]*dist[1] + C[3]*dist[2]) / 2;
      C[1] = (C[4]*dist[0] + C[5]*dist[1] + C[6]*dist[2]) / 2;
      C[2] = (C[5]*dist[0] + C[7]*dist[1] + C[8]*dist[2]) / 2;
      C[3] = (C[6]*dist[0] + C[8]*dist[1] + C[9]*dist[2]) / 2;
      for( int d=0; d<4; d++ ) Ibodies[b][d] += C[d];
      Ibodies[b][0] += (dist[0]*C[1]+dist[1]*C[2]+dist[2]*C[3]) / 3;
    }
  }
};

#endif
