#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <sys/time.h>
#include "vec.h"
using boost::math::binomial_coefficient;
using boost::math::cyl_bessel_k;
using boost::math::tgamma;

const int P = 6;
const double NU = 1.5;
const int LTERM = (P+1)*(P+2)*(P+3)/6;

typedef double real_t;
typedef vec<3,real_t> vec3;
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
    return -C[Index<vecL,kx,ky,kz>::I];
  }
};


template<int nx, int ny, int nz, int kx=nx, int ky=ny, int kz=nz, int flag=5>
struct DerivativeSum {
  static const int nextflag = 5 - (kz < nz || kz == 1);
  static const int dim = kz == (nz-1) ? -1 : 2;
  static const int n = nx + ny + nz;
  static inline real_t loop(const vecL &C, const vec3 &dX) {
    return DerivativeSum<nx,ny,nz,nx,ny,kz-1,nextflag>::loop(C,dX)
      + DerivativeTerm<nx,ny,kz-1,dim>::kernel(C,dX);
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
      + DerivativeTerm<nx,ky-1,nz,dim>::kernel(C,dX);
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
      + DerivativeTerm<kx-1,ny,nz,dim>::kernel(C,dX);
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

template<int np, int nx, int ny, int nz>
struct Kernels {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    static const int n = nx + ny + nz;
    Kernels<np,nx,ny+1,nz-1>::derivative(C,G,dX,coef);
    C[Index<vecL,nx,ny,nz>::I] = DerivativeSum<nx,ny,nz>::loop(G,dX) / n * coef;
  }
};

template<int np, int nx, int ny>
struct Kernels<np,nx,ny,0> {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    static const int n = nx + ny;
    Kernels<np,nx+1,0,ny-1>::derivative(C,G,dX,coef);
    C[Index<vecL,nx,ny,0>::I] = DerivativeSum<nx,ny,0>::loop(G,dX) / n * coef;
  }
};

template<int np, int nx>
struct Kernels<np,nx,0,0> {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    static const int n = nx;
    Kernels<np,0,0,nx-1>::derivative(C,G,dX,coef);
    C[Index<vecL,nx,0,0>::I] = DerivativeSum<nx,0,0>::loop(G,dX) / n * coef;
  }
};

template<int np>
struct Kernels<np,0,0,0> {
  static inline void derivative(vecL &C, vecL &G, const vec3 &dX, real_t &coef) {
    Kernels<np-1,0,0,np-1>::derivative(G,C,dX,coef);
    static const double c = std::sqrt(2 * NU);
    double R = c * std::sqrt(norm(dX));
    double zR = (-0.577216-log(R/2)) * (R<0.413) + 1 * (R>=0.413);
    static const double u = NU - P - 1 + np;
    static const double gu = tgamma(1-u) / tgamma(u);
    static const double aum = std::abs(u-1);
    static const double gaum = 1 / tgamma(aum);
    static const double au = std::abs(u);
    static const double gau = 1 / tgamma(au);
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
    double hu = 0;
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
struct Kernels<0,0,0,0> {
  static inline void derivative(vecL&, vecL&, const vec3&, const real_t&) {}
};


template<int PP>
inline void getCoef2(vecL &C, const vec3 &dX) {
  double coef;
  vecL G;
  Kernels<PP,0,0,PP>::derivative(C,G,dX,coef);
}

typedef vec<3,double> vec3;
vec3 make_vec3(double a, double b, double c) {
  vec3 v;
  v[0] = a;
  v[1] = b;
  v[2] = c;
  return v;
}

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

void matern(int ni, int nj, vec3 * XiL, vec3 XLM, vec3 * XjM, double * f) {
  double temp = powf(2,NU-1) * tgamma(NU);
  for (int i=0; i<ni; i++) {
    f[i] = 0;
    for (int j=0; j<nj; j++) {
      vec3 dX = (XiL[i] + XLM - XjM[j]);
      double R = sqrt(norm(dX) * 2 * NU);
      f[i] += powf(R,NU) * cyl_bessel_k(NU,R) / temp;
      if (R < 1e-12) f[i] += 1;
    }
  }
}

void getCoef(vec3 XLM, double *** G, double *** H) {
  double c = sqrt(2 * NU);
  double R = c * sqrt(norm(XLM));
  double zR = (-0.577216-log(R/2)) * (R<0.413) + 1 * (R>=0.413);
  for (int ip=1; ip<P+2; ip++) {
    double u = NU - P - 2 + ip;
    double au = std::abs(u);
    if (au < 1e-12) {
      H[0][0][0] = cyl_bessel_k(0,R) / zR;
    } else {
      H[0][0][0] = powf(R/2,au) * 2 * cyl_bessel_k(au,R) / tgamma(au);
    }

    u = NU - P - 1 + ip;
    au = std::abs(u);
    if (au < 1e-12) {
      G[0][0][0] = cyl_bessel_k(0,R) / zR;
    } else {
      G[0][0][0] = powf(R/2,au) * 2 * cyl_bessel_k(au,R) / tgamma(au);
    }
    double hu = 0;
    if (u > 1) {
      hu = 0.5 / (u-1);
    } else if (NU == 0) {
      hu = zR;
    } else if (u > 0 && u < 1) {
      hu = powf(R/2,2*u-2) / 2 * tgamma(1-u) / tgamma(u);
    } else if (u == 0) {
      hu = 1 / (R * R * zR);
    } else {
      hu = -2 * u / (R * R);
    }
    for (int sumi=1; sumi<=ip; sumi++) {
      for (int ix=sumi; ix>=0; ix--) {
	for (int iz=0; iz<=sumi-ix; iz++) {
	  int iy = sumi - ix - iz;
          double part1, part2, part3;
          if (ix > 1) {
	    part1 = XLM[0] * H[ix-1][iy][iz] - H[ix-2][iy][iz];
          } else if (ix == 1) {
            part1 = XLM[0] * H[ix-1][iy][iz];
          } else {
            part1 = 0;
          }
          if (iy > 1) {
	    part2 = XLM[1] * H[ix][iy-1][iz] - H[ix][iy-2][iz];
          } else if (iy == 1) {
            part2 = XLM[1] * H[ix][iy-1][iz];
          } else {
            part2 = 0;
          }
          if (iz > 1) {
	    part3 = XLM[2] * H[ix][iy][iz-1] - H[ix][iy][iz-2];
          } else if (iz == 1) {
            part3 = XLM[2] * H[ix][iy][iz-1];
          } else {
            part3 = 0;
          }
	  G[ix][iy][iz] = (part1 + part2 + part3) * c * c * hu / sumi;
	}
      }
    }
    for (int sumi=1; sumi<=ip; sumi++) {
      for (int ix=sumi; ix>=0; ix--) {
        for (int iz=0; iz<=sumi-ix; iz++) {
          int iy = sumi - ix - iz;
          H[ix][iy][iz] = G[ix][iy][iz];
	}
      }
    }
  }
}

double M2P(double *** G, vec3 XiL, vec3 XjM) {
  double f = 0;
  for (int sumi=0; sumi<=P; sumi++) {
    for (int ix=sumi; ix>=0; ix--) {
      for (int iz=0; iz<=sumi-ix; iz++) {
	int iy = sumi - ix - iz;
	for (int sumj=0; sumj<=P; sumj++) {
	  for (int jx=sumj; jx>=0; jx--) {
	    for (int jz=0; jz<=sumj-jx; jz++) {
              int jy = sumj - jx - jz;
              double mom = 
		binomial_coefficient<double>(jx+ix,ix) * powf(-XiL[0],ix) * powf(XjM[0],jx) *
		binomial_coefficient<double>(jy+iy,iy) * powf(-XiL[1],iy) * powf(XjM[1],jy) *
		binomial_coefficient<double>(jz+iz,iz) * powf(-XiL[2],iz) * powf(XjM[2],jz);
              f += G[jx+ix][jy+iy][jz+iz] * mom;
	    }
	  }
	}
      }
    }
  }
  return f;
}

int main() {
  const vec3 XLM = make_vec3(0.7,0.3,0.4);
  const double ri = 0.2;
  const double rj = 0.4;
  const int ni = 30;
  const int nj = 30;
  vec3 * XiL = new vec3 [ni];
  vec3 * XjM = new vec3 [nj];
  double * f = new double [ni];
  double *** G = new double ** [2*P+2];
  double *** H = new double ** [2*P+2];
  for (int i=0; i<2*P+2; i++) {
    G[i] = new double * [2*P+2];
    H[i] = new double * [2*P+2];
    for (int j=0; j<2*P+2; j++) {
      G[i][j] = new double [2*P+2];
      H[i][j] = new double [2*P+2];
    }
  }

  double RLM = sqrt(XLM[0]*XLM[0]+XLM[1]*XLM[1]+XLM[2]*XLM[2]);
  for (int i=0; i<ni; i++) {
    XiL[i][0] = (i*2*ri/(ni-1) - ri) * XLM[0] / RLM;
    XiL[i][1] = (i*2*ri/(ni-1) - ri) * XLM[1] / RLM;
    XiL[i][2] = (i*2*ri/(ni-1) - ri) * XLM[2] / RLM;
  }
  for (int i=0; i<nj; i++) {
    XjM[i][0] = (i*2*rj/(nj-1) - rj) * XLM[0] / RLM;
    XjM[i][1] = (i*2*rj/(nj-1) - rj) * XLM[1] / RLM;
    XjM[i][2] = (i*2*rj/(nj-1) - rj) * XLM[2] / RLM;
  }

  matern(ni,nj,XiL,XLM,XjM,f);

  getCoef(XLM,G,H);
  vecL C;
  getCoef2<P+1>(C,XLM);

  double dif = 0, val = 0;
  for (int sumi=0,ic=0; sumi<=P; sumi++) {
    for (int ix=sumi; ix>=0; ix--) {
      for (int iz=0; iz<=sumi-ix; iz++,ic++) {
	int iy = sumi - ix - iz;
	//std::cout << ix << " " << iy << " " << iz << " " << G[ix][iy][iz] << " " << C[ic] << std::endl;
        dif += (G[ix][iy][iz] - C[ic]) * (G[ix][iy][iz] - C[ic]);
        val += G[ix][iy][iz] * G[ix][iy][iz];
      }
    }
  }
  std::cout << sqrt(dif/val) << std::endl;

  dif = val = 0;
  for (int i=0; i<ni; i++) {
    double f2 = 0;
    for (int j=0; j<nj; j++) {
      f2 += M2P(G, XiL[i], XjM[j]);
    }
    dif += (f[i] - f2) * (f[i] - f2);
    val += f[i] * f[i];
  }
  std::cout << sqrt(dif/val) << std::endl;

  for (int i=0; i<2*P+2; i++) {
    for (int j=0; j<2*P+2; j++) {
      delete[] G[i][j];
      delete[] H[i][j];
    }
    delete[] G[i];
    delete[] H[i];
  }
  delete[] G;
  delete[] H;
  delete[] f;
  delete[] XjM;
  delete[] XiL;
}
