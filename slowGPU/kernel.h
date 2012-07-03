#ifndef kernel_h
#define kernel_h
#include "types.h"

class Kernel {
protected:
  vec3   X0;
  real   R0;

  Cell *C0;
  vec4 *Ibodies;
  vec4 *Jbodies;
  vecM *Multipole;
  vecL *Local;

private:
  inline void getCoef(real *C, const vec3 &dist, real &invR2, const real &invR) const {
    C[0] = invR;
    invR2 = -invR2;
    real x = dist[0], y = dist[1], z = dist[2];

    real invR3 = invR * invR2;
    C[1] = x * invR3;
    C[2] = y * invR3;
    C[3] = z * invR3;

    real invR5 = 3 * invR3 * invR2;
    real t = x * invR5;
    C[4] = x * t + invR3;
    C[5] = y * t;
    C[6] = z * t;
    t = y * invR5;
    C[7] = y * t + invR3;
    C[8] = z * t;
    C[9] = z * z * invR5 + invR3;

    real invR7 = 5 * invR5 * invR2;
    t = x * x * invR7;
    C[10] = x * (t + 3 * invR5);
    C[11] = y * (t +     invR5);
    C[12] = z * (t +     invR5);
    t = y * y * invR7;
    C[13] = x * (t +     invR5);
    C[16] = y * (t + 3 * invR5);
    C[17] = z * (t +     invR5);
    t = z * z * invR7;
    C[15] = x * (t +     invR5);
    C[18] = y * (t +     invR5);
    C[19] = z * (t + 3 * invR5);
    C[14] = x * y * z * invR7;
  }

  inline void sumM2L(real *L, const real *C, const real *M) const {
    for( int i=0; i<LTERM; ++i ) L[i] += C[i];
    L[0] += C[4] *M[1] + C[5] *M[2] + C[6] *M[3] + C[7] *M[4] + C[8] *M[5] + C[9] *M[6];
    L[1] += C[10]*M[1] + C[11]*M[2] + C[12]*M[3] + C[13]*M[4] + C[14]*M[5] + C[15]*M[6];
    L[2] += C[11]*M[1] + C[13]*M[2] + C[14]*M[3] + C[16]*M[4] + C[17]*M[5] + C[18]*M[6];
    L[3] += C[12]*M[1] + C[14]*M[2] + C[15]*M[3] + C[17]*M[4] + C[18]*M[5] + C[19]*M[6];
  }

public:
  Kernel() : X0(0), R0(0), C0(NULL) {}
  ~Kernel() {}

  void P2P(Cell *Ci, Cell *Cj) const {
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

  void P2M(Cell *C, real &Rmax) const {
    int c = C - C0;
    for( int b=C->LEAF; b<C->LEAF+C->NCLEAF; b++ ) {
      vec3 dist;
      for( int d=0; d<3; d++ ) dist[d] = C->X[d] - Jbodies[b][d];
      real R = std::sqrt(norm(dist));
      if( R > Rmax ) Rmax = R;
      real M[LTERM];
      M[0] = Jbodies[b][3];
      real tmp = M[0] * dist[0];
      Multipole[c][0] += M[0];
      Multipole[c][1] += dist[0] * tmp / 2;
      Multipole[c][2] += dist[1] * tmp;
      Multipole[c][3] += dist[2] * tmp;
      tmp = M[0] * dist[1];
      Multipole[c][4] += dist[1] * tmp / 2;
      Multipole[c][5] += dist[2] * tmp;
      Multipole[c][6] += M[0] * dist[2] * dist[2] / 2;
    }
    C->RCRIT = std::min(C->R,Rmax);
  }

  void M2M(Cell *Ci, real &Rmax) const {
    int ci = Ci - C0;
    for( int cj=Ci->CHILD; cj<Ci->CHILD+Ci->NCHILD; cj++ ) {
      Cell *Cj = C0 + cj;
      vec3 dist = Ci->X - Cj->X;
      real R = std::sqrt(norm(dist)) + Cj->RCRIT;
      if( R > Rmax ) Rmax = R;
      vecM M = Multipole[cj];
      for( int i=1; i<7; ++i ) Multipole[ci][i] += M[i];
      real tmp = M[0] * dist[0];
      Multipole[ci][0] += M[0];
      Multipole[ci][1] += dist[0] * tmp / 2;
      Multipole[ci][2] += dist[1] * tmp;
      Multipole[ci][3] += dist[2] * tmp;
      tmp = M[0] * dist[1];
      Multipole[ci][4] += dist[1] * tmp / 2;
      Multipole[ci][5] += dist[2] * tmp;
      Multipole[ci][6] += M[0] * dist[2] * dist[2] / 2;
    }
    Ci->RCRIT = std::min(Ci->R,Rmax);
  }

  void M2L(Cell *Ci, Cell *Cj) const {
    int ci = Ci - C0;
    int cj = Cj - C0;
    vec3 dist = Ci->X - Cj->X;
    real invR2 = 1 / norm(dist);
    real invR  = Multipole[ci][0] * Multipole[cj][0] * std::sqrt(invR2);
    real C[LTERM];
    getCoef(C,dist,invR2,invR);
    sumM2L(Local[ci],C,Multipole[cj]);
  }

  void L2L(Cell *Ci) const {
    int ci = Ci - C0;
    int cj = Ci->PARENT;
    Cell *Cj = C0 + cj;
    vec3 dist = Ci->X - Cj->X;
    real C[LTERM];
    for( int i=0; i<LTERM; ++i ) Local[ci][i] /= Multipole[ci][0];
    for( int i=0; i<LTERM; ++i ) Local[ci][i] += Local[cj][i];
    C[0] = Local[cj][1] *dist[0] + Local[cj][2] *dist[1] + Local[cj][3] *dist[2];
    C[1] = Local[cj][4] *dist[0] + Local[cj][5] *dist[1] + Local[cj][6] *dist[2];
    C[2] = Local[cj][5] *dist[0] + Local[cj][7] *dist[1] + Local[cj][8] *dist[2];
    C[3] = Local[cj][6] *dist[0] + Local[cj][8] *dist[1] + Local[cj][9] *dist[2];
    C[4] = Local[cj][10]*dist[0] + Local[cj][11]*dist[1] + Local[cj][12]*dist[2];
    C[5] = Local[cj][11]*dist[0] + Local[cj][13]*dist[1] + Local[cj][14]*dist[2];
    C[6] = Local[cj][12]*dist[0] + Local[cj][14]*dist[1] + Local[cj][15]*dist[2];
    C[7] = Local[cj][13]*dist[0] + Local[cj][16]*dist[1] + Local[cj][17]*dist[2];
    C[8] = Local[cj][14]*dist[0] + Local[cj][17]*dist[1] + Local[cj][18]*dist[2];
    C[9] = Local[cj][15]*dist[0] + Local[cj][18]*dist[1] + Local[cj][19]*dist[2];
    for( int d=0; d<10; d++ ) Local[ci][d] += C[d];
    C[0] = (C[1]*dist[0] + C[2]*dist[1] + C[3]*dist[2]) / 2;
    C[1] = (C[4]*dist[0] + C[5]*dist[1] + C[6]*dist[2]) / 2;
    C[2] = (C[5]*dist[0] + C[7]*dist[1] + C[8]*dist[2]) / 2;
    C[3] = (C[6]*dist[0] + C[8]*dist[1] + C[9]*dist[2]) / 2;
    for( int d=0; d<4; d++ ) Local[ci][d] += C[d];
    Local[ci][0] += (dist[0]*C[1]+dist[1]*C[2]+dist[2]*C[3]) / 3;
  }

  void L2P(Cell *Ci) const {
    for( int b=Ci->LEAF; b<Ci->LEAF+Ci->NCLEAF; b++ ) {
      int ci = Ci - C0;
      real C[LTERM];
      vec3 dist;
      for( int d=0; d<3; d++ ) dist[d] = Jbodies[b][d] - Ci->X[d];
      for( int d=0; d<4; d++ ) Ibodies[b][d] /= Jbodies[b][3];
      for( int d=0; d<4; d++ ) Ibodies[b][d] += Local[ci][d];
      C[0] = Local[ci][1] *dist[0] + Local[ci][2] *dist[1] + Local[ci][3] *dist[2];
      C[1] = Local[ci][4] *dist[0] + Local[ci][5] *dist[1] + Local[ci][6] *dist[2];
      C[2] = Local[ci][5] *dist[0] + Local[ci][7] *dist[1] + Local[ci][8] *dist[2];
      C[3] = Local[ci][6] *dist[0] + Local[ci][8] *dist[1] + Local[ci][9] *dist[2];
      C[4] = Local[ci][10]*dist[0] + Local[ci][11]*dist[1] + Local[ci][12]*dist[2];
      C[5] = Local[ci][11]*dist[0] + Local[ci][13]*dist[1] + Local[ci][14]*dist[2];
      C[6] = Local[ci][12]*dist[0] + Local[ci][14]*dist[1] + Local[ci][15]*dist[2];
      C[7] = Local[ci][13]*dist[0] + Local[ci][16]*dist[1] + Local[ci][17]*dist[2];
      C[8] = Local[ci][14]*dist[0] + Local[ci][17]*dist[1] + Local[ci][18]*dist[2];
      C[9] = Local[ci][15]*dist[0] + Local[ci][18]*dist[1] + Local[ci][19]*dist[2];
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
