#ifndef kernel_h
#define kernel_h
#include "types.h"

template<int nx, int ny, int nz>
struct Index {
  static const int  M = Index<nx,ny+1,nz-1>::M + 1;
  static const int  I = Index<nx,ny+1,nz-1>::I + 1;
  static const real F = Index<nx,ny,nz-1>::F * nz;
};

template<int nx, int ny>
struct Index<nx,ny,0> {
  static const int  M = Index<nx+1,0,ny-1>::M + 1;
  static const int  I = Index<nx+1,0,ny-1>::I + 1;
  static const real F = Index<nx,ny-1,0>::F * ny;
};

template<int nx>
struct Index<nx,0,0> {
  static const int  M = Index<0,0,nx-1>::M + 1;
  static const int  I = Index<0,0,nx-1>::I + 1;
  static const real F = Index<nx-1,0,0>::F * nx;
};

template<>
struct Index<2,0,0> {
  static const int  M = 1;
  static const int  I = 4;
  static const real F = 2;
};

template<>
struct Index<0,0,1> {
  static const int  M = -1;
  static const int  I = 3;
  static const real F = 1;
};

template<>
struct Index<0,1,0> {
  static const int  M = -1;
  static const int  I = 2;
  static const real F = 1;
};

template<>
struct Index<1,0,0> {
  static const int  M = -1;
  static const int  I = 1;
  static const real F = 1;
};

template<>
struct Index<0,0,0> {
  static const int  M = 0;
  static const int  I = 0;
  static const real F = 1;
};

template<int nx, int ny, int nz>
struct Terms {
  static inline void power(real *C, const vect &dist) {
    Terms<nx,ny+1,nz-1>::power(C,dist);
    C[Index<nx,ny,nz>::I] = C[Index<nx,ny,nz-1>::I] * dist[2] / nz;
  }
};

template<int nx, int ny>
struct Terms<nx,ny,0> {
  static inline void power(real *C, const vect &dist) {
    Terms<nx+1,0,ny-1>::power(C,dist);
    C[Index<nx,ny,0>::I] = C[Index<nx,ny-1,0>::I] * dist[1] / ny;
  }
};

template<int nx>
struct Terms<nx,0,0> {
  static inline void power(real *C, const vect &dist) {
    Terms<0,0,nx-1>::power(C,dist);
    C[Index<nx,0,0>::I] = C[Index<nx-1,0,0>::I] * dist[0] / nx;
  }
};

template<>
struct Terms<0,0,0> {
  static inline void power(real*, const vect&) {}
};


template<int nx, int ny, int nz, int kx=nx, int ky=ny, int kz=nz>
struct M2MSum {
  static inline real kernel(const real *C, const real *M) {
    return M2MSum<nx,ny,nz,kx,ky,kz-1>::kernel(C,M)
         + C[Index<nx-kx,ny-ky,nz-kz>::I]*M[Index<kx,ky,kz>::M];
  }
};

template<int nx, int ny, int nz, int kx, int ky>
struct M2MSum<nx,ny,nz,kx,ky,0> {
  static inline real kernel(const real *C, const real *M) {
    return M2MSum<nx,ny,nz,kx,ky-1,nz>::kernel(C,M)
         + C[Index<nx-kx,ny-ky,nz>::I]*M[Index<kx,ky,0>::M];
  }
};

template<int nx, int ny, int nz, int kx>
struct M2MSum<nx,ny,nz,kx,0,0> {
  static inline real kernel(const real *C, const real *M) {
    return M2MSum<nx,ny,nz,kx-1,ny,nz>::kernel(C,M)
         + C[Index<nx-kx,ny,nz>::I]*M[Index<kx,0,0>::M];
  }
};

template<int nx, int ny, int nz>
struct M2MSum<nx,ny,nz,0,0,1> {
  static inline real kernel(const real*, const real*) { return 0; }
};

template<int nx, int ny, int nz>
struct M2MSum<nx,ny,nz,0,1,0> {
  static inline real kernel(const real*, const real*) { return 0; }
};

template<int nx, int ny, int nz>
struct M2MSum<nx,ny,nz,1,0,0> {
  static inline real kernel(const real*, const real*) { return 0; }
};

template<int nx, int ny, int nz>
struct M2MSum<nx,ny,nz,0,0,0> {
  static inline real kernel(const real*, const real*) { return 0; }
};

template<int nx, int ny, int nz, int kx=0, int ky=0, int kz=P-nx-ny-nz>
struct LocalSum {
  static inline real kernel(const real *C, const real *L) {
    return LocalSum<nx,ny,nz,kx,ky+1,kz-1>::kernel(C,L)
         + C[Index<kx,ky,kz>::I] * L[Index<nx+kx,ny+ky,nz+kz>::I];
  }
};

template<int nx, int ny, int nz, int kx, int ky>
struct LocalSum<nx,ny,nz,kx,ky,0> {
  static inline real kernel(const real *C, const real *L) {
    return LocalSum<nx,ny,nz,kx+1,0,ky-1>::kernel(C,L)
         + C[Index<kx,ky,0>::I] * L[Index<nx+kx,ny+ky,nz>::I];
  }
};

template<int nx, int ny, int nz, int kx>
struct LocalSum<nx,ny,nz,kx,0,0> {
  static inline real kernel(const real *C, const real *L) {
    return LocalSum<nx,ny,nz,0,0,kx-1>::kernel(C,L)
         + C[Index<kx,0,0>::I] * L[Index<nx+kx,ny,nz>::I];
  }
};

template<int nx, int ny, int nz>
struct LocalSum<nx,ny,nz,0,0,0> {
  static inline real kernel(const real*, const real*) { return 0; }
};


template<int nx, int ny, int nz>
struct Upward {
  static inline void M2M(real *MI, const real *C, const real *MJ) {
    Upward<nx,ny+1,nz-1>::M2M(MI,C,MJ);
    MI[Index<nx,ny,nz>::M] += M2MSum<nx,ny,nz>::kernel(C,MJ);
  }
};

template<int nx, int ny>
struct Upward<nx,ny,0> {
  static inline void M2M(real *MI, const real *C, const real *MJ) {
    Upward<nx+1,0,ny-1>::M2M(MI,C,MJ);
    MI[Index<nx,ny,0>::M] += M2MSum<nx,ny,0>::kernel(C,MJ);
  }
};

template<int nx>
struct Upward<nx,0,0> {
  static inline void M2M(real *MI, const real *C, const real *MJ) {
    Upward<0,0,nx-1>::M2M(MI,C,MJ);
    MI[Index<nx,0,0>::M] += M2MSum<nx,0,0>::kernel(C,MJ);
  }
};

template<>
struct Upward<0,0,1> {
  static inline void M2M(real*, const real*, const real*) {}
};

template<>
struct Upward<0,1,0> {
  static inline void M2M(real*, const real*, const real*) {}
};

template<>
struct Upward<1,0,0> {
  static inline void M2M(real*, const real*, const real*) {}
};

template<>
struct Upward<0,0,0> {
  static inline void M2M(real*, const real*, const real*) {}
};


template<int nx, int ny, int nz>
struct Downward {
  static inline void L2L(real *LI, const real *C, const real *LJ) {
    Downward<nx,ny+1,nz-1>::L2L(LI,C,LJ);
    LI[Index<nx,ny,nz>::I] += LocalSum<nx,ny,nz>::kernel(C,LJ);
  }
};

template<int nx, int ny>
struct Downward<nx,ny,0> {
  static inline void L2L(real *LI, const real *C, const real *LJ) {
    Downward<nx+1,0,ny-1>::L2L(LI,C,LJ);
    LI[Index<nx,ny,0>::I] += LocalSum<nx,ny,0>::kernel(C,LJ);
  }
};

template<int nx>
struct Downward<nx,0,0> {
  static inline void L2L(real *LI, const real *C, const real *LJ) {
    Downward<0,0,nx-1>::L2L(LI,C,LJ);
    LI[Index<nx,0,0>::I] += LocalSum<nx,0,0>::kernel(C,LJ);
  }
};

template<>
struct Downward<0,0,0> {
  static inline void L2L(real*, const real*, const real*) {}
};

class Kernel {
protected:
  vect   X0;
  real   R0;
  Cell  *C0;

  real (*Ibodies)[4];
  real (*Jbodies)[4];
  real (*Multipole)[MTERM];
  real (*Local)[LTERM];

private:
  inline void getCoef(real *C, const vect &dist, real &invR2, const real &invR) const {
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
      vect F0 = 0;
      for( int bj=Cj->LEAF; bj<Cj->LEAF+Cj->NDLEAF; ++bj ) {
        vect dX;
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
    for( int b=C->LEAF; b<C->LEAF+C->NCLEAF; b++ ) {
      vect dist;
      for( int d=0; d<3; d++ ) dist[d] = C->X[d] - Jbodies[b][d];
      real R = std::sqrt(norm(dist));
      if( R > Rmax ) Rmax = R;
      real M[LTERM];
      M[0] = Jbodies[b][3];
      Terms<0,0,P-1>::power(M,dist);
      Multipole[C-C0][0] += M[0];
      for( int i=1; i<MTERM; ++i ) Multipole[C-C0][i] += M[i+3];
    }
    C->RCRIT = std::min(C->R,Rmax);
  }

  void M2M(Cell *Ci, real &Rmax) const {
    for( Cell *Cj=C0+Ci->CHILD; Cj<C0+Ci->CHILD+Ci->NCHILD; ++Cj ) {
      vect dist = Ci->X - Cj->X;
      real R = std::sqrt(norm(dist)) + Cj->RCRIT;
      if( R > Rmax ) Rmax = R;
      real M[MTERM];
      real C[LTERM];
      C[0] = 1;
      Terms<0,0,P-1>::power(C,dist);
      for( int i=0; i<MTERM; ++i ) M[i] = Multipole[Cj-C0][i];
      Multipole[Ci-C0][0] += C[0] * M[0];
      for( int i=1; i<MTERM; ++i ) Multipole[Ci-C0][i] += C[i+3] * M[0];
      Upward<0,0,P-1>::M2M(Multipole[Ci-C0],C,M);
    }
    Ci->RCRIT = std::min(Ci->R,Rmax);
  }

  void M2L(Cell *Ci, Cell *Cj) const {
    vect dist = Ci->X - Cj->X;
    real invR2 = 1 / norm(dist);
    real invR  = Multipole[Ci-C0][0] * Multipole[Cj-C0][0] * std::sqrt(invR2);
    real C[LTERM];
    getCoef(C,dist,invR2,invR);
    sumM2L(Local[Ci-C0],C,Multipole[Cj-C0]);
  }

  void L2L(Cell *Ci) const {
    Cell *Cj = C0 + Ci->PARENT;
    vect dist = Ci->X - Cj->X;
    real C[LTERM];
    C[0] = 1;
    Terms<0,0,P>::power(C,dist);

    for( int i=0; i<LTERM; ++i ) Local[Ci-C0][i] /= Multipole[Ci-C0][0];
    for( int i=0; i<LTERM; ++i ) Local[Ci-C0][i] += Local[Cj-C0][i];
    for( int i=1; i<LTERM; ++i ) Local[Ci-C0][0] += C[i] * Local[Cj-C0][i];
    Downward<0,0,P-1>::L2L(Local[Ci-C0],C,Local[Cj-C0]);
  }

  void L2P(Cell *Ci) const {
    for( int b=Ci->LEAF; b<Ci->LEAF+Ci->NCLEAF; b++ ) {
      real C[LTERM];
      vect dist;
      for( int d=0; d<3; d++ ) dist[d] = Jbodies[b][d] - Ci->X[d];
      for( int d=0; d<4; d++ ) Ibodies[b][d] /= Jbodies[b][3];
      for( int d=0; d<4; d++ ) Ibodies[b][d] += Local[Ci-C0][d];
      C[0] = Local[Ci-C0][1] *dist[0] + Local[Ci-C0][2] *dist[1] + Local[Ci-C0][3] *dist[2];
      C[1] = Local[Ci-C0][4] *dist[0] + Local[Ci-C0][5] *dist[1] + Local[Ci-C0][6] *dist[2];
      C[2] = Local[Ci-C0][5] *dist[0] + Local[Ci-C0][7] *dist[1] + Local[Ci-C0][8] *dist[2];
      C[3] = Local[Ci-C0][6] *dist[0] + Local[Ci-C0][8] *dist[1] + Local[Ci-C0][9] *dist[2];
      C[4] = Local[Ci-C0][10]*dist[0] + Local[Ci-C0][11]*dist[1] + Local[Ci-C0][12]*dist[2];
      C[5] = Local[Ci-C0][11]*dist[0] + Local[Ci-C0][13]*dist[1] + Local[Ci-C0][14]*dist[2];
      C[6] = Local[Ci-C0][12]*dist[0] + Local[Ci-C0][14]*dist[1] + Local[Ci-C0][15]*dist[2];
      C[7] = Local[Ci-C0][13]*dist[0] + Local[Ci-C0][16]*dist[1] + Local[Ci-C0][17]*dist[2];
      C[8] = Local[Ci-C0][14]*dist[0] + Local[Ci-C0][17]*dist[1] + Local[Ci-C0][18]*dist[2];
      C[9] = Local[Ci-C0][15]*dist[0] + Local[Ci-C0][18]*dist[1] + Local[Ci-C0][19]*dist[2];
      for( int d=0; d<4; d++ ) Ibodies[b][d] += C[d];
      C[0] = (C[1]*dist[0] + C[3]*dist[2] + C[2]*dist[1]) / 2;
      C[1] = (C[4]*dist[0] + C[6]*dist[2] + C[5]*dist[1]) / 2;
      C[2] = (C[5]*dist[0] + C[8]*dist[2] + C[7]*dist[1]) / 2;
      C[3] = (C[6]*dist[0] + C[9]*dist[2] + C[8]*dist[1]) / 2;
      for( int d=0; d<4; d++ ) Ibodies[b][d] += C[d];
      Ibodies[b][0] += (dist[0]*C[1]+dist[1]*C[2]+dist[2]*C[3]) / 3;
    }
  }
};

#endif
