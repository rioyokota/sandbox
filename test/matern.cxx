#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <sys/time.h>
#include "vec.h"
using boost::math::binomial_coefficient;
using boost::math::cyl_bessel_k;
using boost::math::tgamma;

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

void matern(int ni, int nj, vec3 * XiL, vec3 XLM, vec3 * XjM, double nu, double ** f) {
  double temp = powf(2,nu-1) * tgamma(nu);
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      vec3 dX = (XiL[i] + XLM - XjM[j]);
      double R = sqrt(norm(dX) * 2 * nu);
      f[i][j] = powf(R,nu) * cyl_bessel_k(nu,R) / temp;
      if (R < 1e-12) f[i][j] = 1;
    }
  }
}

void getCoef(vec3 XLM, double nu, int p, double *** G, double *** H) {
  double c = sqrt(2 * nu);
  double R = c * sqrt(norm(XLM));
  double zR = (-0.577216-log(R/2)) * (R<0.413) + 1 * (R>=0.413);
  for (int ip=1; ip<p+2; ip++) {
    double u = nu - p - 2 + ip;
    double au = std::abs(u);
    if (au < 1e-12) {
      H[0][0][0] = cyl_bessel_k(0,R) / zR;
    } else {
      H[0][0][0] = powf(R/2,au) * 2 * cyl_bessel_k(au,R) / tgamma(au);
    }

    u = nu - p - 1 + ip;
    au = std::abs(u);
    double hu;
    if (u > 1) {
      hu = 0.5 / (u-1);
    } else if (nu == 0) {
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
    if (au < 1e-12) {
      G[0][0][0] = cyl_bessel_k(0,R) / zR;
    } else {
      G[0][0][0] = powf(R/2,au) * 2 * cyl_bessel_k(au,R) / tgamma(au);
    }
    for (int ix=0; ix<=p; ix++) {
      for (int iy=0; iy<=p; iy++) {
        for (int iz=0; iz<=p; iz++) {
          H[ix][iy][iz] = G[ix][iy][iz];
	}
      }
    }
  }
}

double M2P(int p, double *** G, vec3 XiL, vec3 XjM) {
  double f = 0;
  for (int sumi=0; sumi<=p; sumi++) {
    for (int ix=sumi; ix>=0; ix--) {
      for (int iz=0; iz<=sumi-ix; iz++) {
	int iy = sumi - ix - iz;
	for (int sumj=0; sumj<=p; sumj++) {
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
  const double nu = 1.5;
  const int p = 3;
  const vec3 XLM = make_vec3(0.7,0.3,0.4);
  const double ri = 0.2;
  const double rj = 0.4;
  const int ni = 30;
  const int nj = 30;
  vec3 * XiL = new vec3 [ni];
  vec3 * XjM = new vec3 [nj];
  double ** f = new double * [ni];
  for (int i=0; i<ni; i++) f[i] = new double [nj];
  double *** G = new double ** [2*p+2];
  double *** H = new double ** [2*p+2];
  for (int i=0; i<2*p+2; i++) {
    G[i] = new double * [2*p+2];
    H[i] = new double * [2*p+2];
    for (int j=0; j<2*p+2; j++) {
      G[i][j] = new double [2*p+2];
      H[i][j] = new double [2*p+2];
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

  matern(ni,nj,XiL,XLM,XjM,nu,f);

  getCoef(XLM,nu,2*p,G,H);

  double dif = 0, val = 0;
  for (int i=0; i<ni; i++) {
    for (int j=0; j<nj; j++) {
      double f2 = M2P(p, G, XiL[i], XjM[j]);
      dif += (f[i][j] - f2) * (f[i][j] - f2);
      val += f[i][j] * f[i][j];
    }
  }
  std::cout << sqrt(dif/val) << std::endl;

  for (int i=0; i<2*p+2; i++) {
    for (int j=0; j<2*p+2; j++) {
      delete[] G[i][j];
      delete[] H[i][j];
    }
    delete[] G[i];
    delete[] H[i];
  }
  delete[] G;
  delete[] H;
  for (int i=0; i<ni; i++) delete[] f[i];
  delete[] f;
  delete[] XjM;
  delete[] XiL;
}
