#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <sys/time.h>
#include "vec.h"
using boost::math::cyl_bessel_k;
using boost::math::tgamma;

typedef float real_t;
const int P = 8;
const real_t NU = 1.5;
const real_t SIGMA = 2;
const int MTERM = P*(P+1)*(P+2)/6;
const int LTERM = (P+1)*(P+2)*(P+3)/6;

typedef vec<3,real_t> vec3;
typedef vec<MTERM,real_t> vecM;
typedef vec<LTERM,real_t> vecL;

template<typename T, int nx, int ny, int nz>
struct Index {
  static const int                I = Index<T,nx,ny+1,nz-1>::I + 1;
  static const unsigned long long F = Index<T,nx,ny,nz-1>::F * nz;
};

template<typename T, int nx, int ny>
struct Index<T,nx,ny,0> {
  static const int                I = Index<T,nx+1,0,ny-1>::I + 1;
  static const unsigned long long F = Index<T,nx,ny-1,0>::F * ny;
};

template<typename T, int nx>
struct Index<T,nx,0,0> {
  static const int                I = Index<T,0,0,nx-1>::I + 1;
  static const unsigned long long F = Index<T,nx-1,0,0>::F * nx;
};

template<typename T>
struct Index<T,0,0,0> {
  static const int                I = 0;
  static const unsigned long long F = 1;
};


template<int kx, int ky , int kz, int d>
struct DerivativeTerm {
  static inline real_t kernel(const vecL &C, const vec3 &dX) {
    return dX[d] * C[Index<vecL,kx,ky,kz>::I];
  }
};

template<int kx, int ky , int kz>
struct DerivativeTerm<kx,ky,kz,-1> {
  static inline real_t kernel(const vecL &C, const vec3&) {
    return C[Index<vecL,kx,ky,kz>::I];
  }
};


template<int nx, int ny, int nz, int kx=nx, int ky=ny, int kz=nz, int flag=5>
struct DerivativeSum {
  static const int nextflag = 5 - (kz < nz || kz == 1);
  static const int dim = kz == (nz-1) ? -1 : 2;
  static const int n = nx + ny + nz;
  static inline real_t loop(const vecL &C, const vec3 &dX) {
    return DerivativeSum<nx,ny,nz,nx,ny,kz-1,nextflag>::loop(C,dX)
      - DerivativeTerm<nx,ny,kz-1,dim>::kernel(C,dX);
  }
};

template<int nx, int ny, int nz, int kx, int ky, int kz>
struct DerivativeSum<nx,ny,nz,kx,ky,kz,4> {
  static const int nextflag = 3 - (ny == 0);
  static inline real_t loop(const vecL &C, const vec3 &dX) {
    return DerivativeSum<nx,ny,nz,nx,ny,nz,nextflag>::loop(C,dX);
  }
};

template<int nx, int ny, int nz, int kx, int ky, int kz>
struct DerivativeSum<nx,ny,nz,kx,ky,kz,3> {
  static const int nextflag = 3 - (ky < ny || ky == 1);
  static const int dim = ky == (ny-1) ? -1 : 1;
  static const int n = nx + ny + nz;
  static inline real_t loop(const vecL &C, const vec3 &dX) {
    return DerivativeSum<nx,ny,nz,nx,ky-1,nz,nextflag>::loop(C,dX)
      - DerivativeTerm<nx,ky-1,nz,dim>::kernel(C,dX);
  }
};

template<int nx, int ny, int nz, int kx, int ky, int kz>
struct DerivativeSum<nx,ny,nz,kx,ky,kz,2> {
  static const int nextflag = 1 - (nx == 0);
  static inline real_t loop(const vecL &C, const vec3 &dX) {
    return DerivativeSum<nx,ny,nz,nx,ny,nz,nextflag>::loop(C,dX);
  }
};

template<int nx, int ny, int nz, int kx, int ky, int kz>
struct DerivativeSum<nx,ny,nz,kx,ky,kz,1> {
  static const int nextflag = 1 - (kx < nx || kx == 1);
  static const int dim = kx == (nx-1) ? -1 : 0;
  static const int n = nx + ny + nz;
  static inline real_t loop(const vecL &C, const vec3 &dX) {
    return DerivativeSum<nx,ny,nz,kx-1,ny,nz,nextflag>::loop(C,dX)
      - DerivativeTerm<kx-1,ny,nz,dim>::kernel(C,dX);
  }
};

template<int nx, int ny, int nz, int kx, int ky, int kz>
struct DerivativeSum<nx,ny,nz,kx,ky,kz,0> {
  static inline real_t loop(const vecL&, const vec3&) {
    return 0;
  }
};

template<int nx, int ny, int nz, int kx, int ky>
struct DerivativeSum<nx,ny,nz,kx,ky,0,5> {
  static inline real_t loop(const vecL &C, const vec3 &dX) {
    return DerivativeSum<nx,ny,nz,nx,ny,0,4>::loop(C,dX);
  }
};


template<int nx, int ny, int nz, int kx=nx, int ky=ny, int kz=nz>
struct MultipoleSum {
  static inline real_t kernel(const vecL &C, const vecM &M) {
    return MultipoleSum<nx,ny,nz,kx,ky,kz-1>::kernel(C,M)
      + C[Index<vecL,nx-kx,ny-ky,nz-kz>::I]*M[Index<vecM,kx,ky,kz>::I];
  }
};

template<int nx, int ny, int nz, int kx, int ky>
struct MultipoleSum<nx,ny,nz,kx,ky,0> {
  static inline real_t kernel(const vecL &C, const vecM &M) {
    return MultipoleSum<nx,ny,nz,kx,ky-1,nz>::kernel(C,M)
      + C[Index<vecL,nx-kx,ny-ky,nz>::I]*M[Index<vecM,kx,ky,0>::I];
  }
};

template<int nx, int ny, int nz, int kx>
struct MultipoleSum<nx,ny,nz,kx,0,0> {
  static inline real_t kernel(const vecL &C, const vecM &M) {
    return MultipoleSum<nx,ny,nz,kx-1,ny,nz>::kernel(C,M)
      + C[Index<vecL,nx-kx,ny,nz>::I]*M[Index<vecM,kx,0,0>::I];
  }
};

template<int nx, int ny, int nz>
struct MultipoleSum<nx,ny,nz,0,0,0> {
  static inline real_t kernel(const vecL&, const vecM&) { return 0; }
};


template<int nx, int ny, int nz, typename T, int kx=0, int ky=0, int kz=P-nx-ny-nz>
struct LocalSum {
  static inline real_t kernel(const T &M, const vecL &L) {
    return LocalSum<nx,ny,nz,T,kx,ky+1,kz-1>::kernel(M,L)
      + M[Index<T,kx,ky,kz>::I] * L[Index<vecL,nx+kx,ny+ky,nz+kz>::I];
  }
};

template<int nx, int ny, int nz, typename T, int kx, int ky>
struct LocalSum<nx,ny,nz,T,kx,ky,0> {
  static inline real_t kernel(const T &M, const vecL &L) {
    return LocalSum<nx,ny,nz,T,kx+1,0,ky-1>::kernel(M,L)
      + M[Index<T,kx,ky,0>::I] * L[Index<vecL,nx+kx,ny+ky,nz>::I];
  }
};

template<int nx, int ny, int nz, typename T, int kx>
struct LocalSum<nx,ny,nz,T,kx,0,0> {
  static inline real_t kernel(const T &M, const vecL &L) {
    return LocalSum<nx,ny,nz,T,0,0,kx-1>::kernel(M,L)
      + M[Index<T,kx,0,0>::I] * L[Index<vecL,nx+kx,ny,nz>::I];
  }
};

template<int nx, int ny, int nz, typename T>
struct LocalSum<nx,ny,nz,T,0,0,0> {
  static inline real_t kernel(const T&, const vecL&) { return 0; }
};


template<int nx, int ny, int nz>
struct Kernels {
  static inline void power(vecL &C, const vec3 &dX) {
    Kernels<nx,ny+1,nz-1>::power(C,dX);
    C[Index<vecL,nx,ny,nz>::I] = C[Index<vecL,nx,ny,nz-1>::I] * dX[2] / nz;
  }
  static inline void scale(vecL &C) {
    Kernels<nx,ny+1,nz-1>::scale(C);
    C[Index<vecL,nx,ny,nz>::I] *= Index<vecL,nx,ny,nz>::F;
  }
  static inline void M2M(vecM &MI, const vecL &C, const vecM &MJ) {
    Kernels<nx,ny+1,nz-1>::M2M(MI,C,MJ);
    MI[Index<vecM,nx,ny,nz>::I] += MultipoleSum<nx,ny,nz>::kernel(C,MJ);
  }
  static inline void M2L(vecL &L, const vecL &C, const vecM &M) {
    Kernels<nx,ny+1,nz-1>::M2L(L,C,M);
    L[Index<vecL,nx,ny,nz>::I] += LocalSum<nx,ny,nz,vecM>::kernel(M,C);
  }
  static inline void L2L(vecL &LI, const vecL &C, const vecL &LJ) {
    Kernels<nx,ny+1,nz-1>::L2L(LI,C,LJ);
    LI[Index<vecL,nx,ny,nz>::I] += LocalSum<nx,ny,nz,vecL>::kernel(C,LJ);
  }
};

template<int nx, int ny>
struct Kernels<nx,ny,0> {
  static inline void power(vecL &C, const vec3 &dX) {
    Kernels<nx+1,0,ny-1>::power(C,dX);
    C[Index<vecL,nx,ny,0>::I] = C[Index<vecL,nx,ny-1,0>::I] * dX[1] / ny;
  }
  static inline void scale(vecL &C) {
    Kernels<nx+1,0,ny-1>::scale(C);
    C[Index<vecL,nx,ny,0>::I] *= Index<vecL,nx,ny,0>::F;
  }
  static inline void M2M(vecM &MI, const vecL &C, const vecM &MJ) {
    Kernels<nx+1,0,ny-1>::M2M(MI,C,MJ);
    MI[Index<vecM,nx,ny,0>::I] += MultipoleSum<nx,ny,0>::kernel(C,MJ);
  }
  static inline void M2L(vecL &L, const vecL &C, const vecM &M) {
    Kernels<nx+1,0,ny-1>::M2L(L,C,M);
    L[Index<vecL,nx,ny,0>::I] += LocalSum<nx,ny,0,vecM>::kernel(M,C);
  }
  static inline void L2L(vecL &LI, const vecL &C, const vecL &LJ) {
    Kernels<nx+1,0,ny-1>::L2L(LI,C,LJ);
    LI[Index<vecL,nx,ny,0>::I] += LocalSum<nx,ny,0,vecL>::kernel(C,LJ);
  }
};

template<int nx>
struct Kernels<nx,0,0> {
  static inline void power(vecL &C, const vec3 &dX) {
    Kernels<0,0,nx-1>::power(C,dX);
    C[Index<vecL,nx,0,0>::I] = C[Index<vecL,nx-1,0,0>::I] * dX[0] / nx;
  }
  static inline void scale(vecL &C) {
    Kernels<0,0,nx-1>::scale(C);
    C[Index<vecL,nx,0,0>::I] *= Index<vecL,nx,0,0>::F;
  }
  static inline void M2M(vecM &MI, const vecL &C, const vecM &MJ) {
    Kernels<0,0,nx-1>::M2M(MI,C,MJ);
    MI[Index<vecM,nx,0,0>::I] += MultipoleSum<nx,0,0>::kernel(C,MJ);
  }
  static inline void M2L(vecL &L, const vecL &C, const vecM &M) {
    Kernels<0,0,nx-1>::M2L(L,C,M);
    L[Index<vecL,nx,0,0>::I] += LocalSum<nx,0,0,vecM>::kernel(M,C);
  }
  static inline void L2L(vecL &LI, const vecL &C, const vecL &LJ) {
    Kernels<0,0,nx-1>::L2L(LI,C,LJ);
    LI[Index<vecL,nx,0,0>::I] += LocalSum<nx,0,0,vecL>::kernel(C,LJ);
  }
};

template<>
struct Kernels<0,0,0> {
  static inline void power(vecL&, const vec3&) {}
  static inline void scale(vecL&) {}
  static inline void M2M(vecM&, const vecL&, const vecM&) {}
  static inline void M2L(vecL&, const vecL&, const vecM&) {}
  static inline void L2L(vecL&, const vecL&, const vecL&) {}
};


template<int np, int nx, int ny, int nz>
struct Kernels2 {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    static const int n = nx + ny + nz;
    Kernels2<np,nx,ny+1,nz-1>::derivative(C,G,dX,coef);
    C[Index<vecL,nx,ny,nz>::I] = DerivativeSum<nx,ny,nz>::loop(G,dX) / n * coef;
  }
};

template<int np, int nx, int ny>
struct Kernels2<np,nx,ny,0> {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    static const int n = nx + ny;
    Kernels2<np,nx+1,0,ny-1>::derivative(C,G,dX,coef);
    C[Index<vecL,nx,ny,0>::I] = DerivativeSum<nx,ny,0>::loop(G,dX) / n * coef;
  }
};

template<int np, int nx>
struct Kernels2<np,nx,0,0> {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    static const int n = nx;
    Kernels2<np,0,0,nx-1>::derivative(C,G,dX,coef);
    C[Index<vecL,nx,0,0>::I] = DerivativeSum<nx,0,0>::loop(G,dX) / n * coef;
  }
};

template<int np>
struct Kernels2<np,0,0,0> {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    Kernels2<np-1,0,0,np-1>::derivative(G,C,dX,coef);
    static const real_t c = std::sqrt(2 * NU);
    real_t R = c * std::sqrt(norm(dX));
    real_t zR = (-0.577216-log(R/2)) * (R<0.413) + 1 * (R>=0.413);
    static const real_t u = NU - P + np;
    static const real_t gu = tgamma(1-u) / tgamma(u);
    static const real_t aum = std::abs(u-1);
    static const real_t gaum = 1 / tgamma(aum);
    static const real_t au = std::abs(u);
    static const real_t gau = 1 / tgamma(au);
    if (aum < 1e-12) {
      G[0] = cyl_bessel_k(0,R) / zR;
    } else {
      G[0] = std::pow(R/2,aum) * 2 * cyl_bessel_k(aum,R) * gaum;
    }
    if (au < 1e-12) {
      C[0] = cyl_bessel_k(0,R) / zR;
    } else {
      C[0] = std::pow(R/2,au) * 2 * cyl_bessel_k(au,R) * gau;
    }
    real_t hu = 0;
    if (u > 1) {
      hu = 0.5 / (u-1);
    } else if (NU == 0) {
      hu = zR;
    } else if (u > 0 && u < 1) {
      hu = std::pow(R/2,2*u-2) / 2 * gu;
    } else if (u == 0) {
      hu = 1 / (R * R * zR);
    } else {
      hu = -2 * u / (R * R);
    }
    coef = c * c * hu;
  }
};

template<>
struct Kernels2<0,0,0,0> {
  static inline void derivative(vecL&, vecL&, const vec3&, const real_t&) {}
};


template<int PP>
inline void getCoef(vecL &C, const vec3 &dX) {
  real_t coef;
  vecL G;
  Kernels2<PP,0,0,PP>::derivative(C,G,dX,coef);
  Kernels<0,0,PP>::scale(C);
}


real_t get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return real_t(tv.tv_sec+tv.tv_usec*1e-6);
}

void matern(int ni, int nj, vec3 * XiL, vec3 XLM, vec3 * XMj, real_t * SRC, real_t * f) {
  real_t temp = std::pow(2,NU-1) * tgamma(NU);
  for (int i=0; i<ni; i++) {
    f[i] = 0;
    for (int j=0; j<nj; j++) {
      vec3 dX = (XiL[i] + XLM + XMj[j]) / SIGMA;
      real_t R = sqrt(norm(dX) * 2 * NU);
      if (R < 1e-12) {
        f[i] += SRC[j];
      } else {
        f[i] += SRC[j] * std::pow(R,NU) * cyl_bessel_k(NU,R) / temp;
      }
    }
  }
}

vecM P2M(vec3 XMj, real_t SRC) {
  vecM M;
  vecL C = 0;
  C[0] = SRC;
  Kernels<0,0,P-1>::power(C,XMj/SIGMA);
  for (int i=0; i<MTERM; i++) M[i] = C[i];
  return M;
}

vecL M2L(vec3 XLM, vecM M) {
  vecL C;
  getCoef<P>(C,XLM/SIGMA);
  vecL L = C * M[0];
  for (int i=1; i<MTERM; i++) L[0] += M[i] * C[i];
  Kernels<0,0,P-1>::M2L(L,C,M);
  return L;
}

real_t L2P(vec3 XiL, vecL L) {
  vecL C;
  C[0] = 1;
  Kernels<0,0,P>::power(C,XiL/SIGMA);
  real_t f = 0;
  for (int i=0; i<LTERM; i++) f += C[i] * L[i];
  return f;
}

int main() {
  const vec3 XLM = 2 * M_PI;
  const int ni = 30;
  const int nj = 30;
  vec3 * XiL = new vec3 [ni];
  vec3 * XMj = new vec3 [nj];
  real_t * SRC = new real_t [nj];
  real_t * f = new real_t [ni];

  for (int i=0; i<ni; i++) {
    XiL[i][0] = (drand48() - .5) * M_PI;
    XiL[i][1] = (drand48() - .5) * M_PI;
    XiL[i][2] = (drand48() - .5) * M_PI;
  }

  for (int i=0; i<nj; i++) {
    XMj[i][0] = (drand48() - .5) * M_PI;
    XMj[i][1] = (drand48() - .5) * M_PI;
    XMj[i][2] = (drand48() - .5) * M_PI;
    SRC[i] = drand48();
  }

  matern(ni,nj,XiL,XLM,XMj,SRC,f);

  vecM M = 0;
  for (int j=0; j<nj; j++) {
    M += P2M(XMj[j],SRC[j]);
  }

  vecL L = M2L(XLM,M);

  real_t dif = 0, val = 0;
  for (int i=0; i<ni; i++) {
    real_t f2 = L2P(XiL[i], L);
    dif += (f[i] - f2) * (f[i] - f2);
    val += f[i] * f[i];
  }
  std::cout << sqrt(dif/val) << std::endl;

  delete[] f;
  delete[] SRC;
  delete[] XMj;
  delete[] XiL;
}
