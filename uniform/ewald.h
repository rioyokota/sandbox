#include <cassert>
#include <cmath>
#include "types.h"

class Ewald {
private:
  real R0;
  real X0[3];

private:
  inline int getKey(int *ix, int level) const {
    int id = 0;
    for (int lev=0; lev<level; lev++) {
      for_3d id += ((ix[d] >> lev) & 1) << (3 * lev + d);
    }
    return id;
  }

  inline void getIndex(int *ix, int index) const {
    for_3d ix[d] = 0;
    int d = 0, level = 0;
    while (index != 0) {
      ix[d] += (index & 1) * (1 << level);
      index >>= 1;
      d = (d+1) % 3;
      if (d == 0) level++;
    }
  }
  
  void dft(int numWaves, int numBodies, real scale, real *waveRe, real *waveIm,
	   real (*waveK)[3], real (*Jbodies)[4]) {
#pragma omp parallel for
    for (int w=0; w<numWaves; w++) { 
      waveRe[w] = waveIm[w] = 0;
      for (int b=0; b<numBodies; b++) {
	real th = 0;
	for_3d th += waveK[w][d] * Jbodies[b][d] * scale;
	waveRe[w] += Jbodies[b][3] * cos(th);
	waveIm[w] += Jbodies[b][3] * sin(th);
      }
    }
  }

  void idft(int numWaves, int numBodies, real scale, real *waveRe, real *waveIm,
	    real (*waveK)[3], real (*Ibodies)[4], real (*Jbodies)[4]) {
#pragma omp parallel for
    for (int b=0; b<numBodies; b++) {
      for (int w=0; w<numWaves; w++) {
	real th = 0;
	for_3d th += waveK[w][d] * Jbodies[b][d] * scale;
	real dtmp = waveRe[w] * sin(th) - waveIm[w] * cos(th);
	Ibodies[b][0] += waveRe[w] * cos(th) + waveIm[w] * sin(th);
	for_3d Ibodies[b][d+1] -= dtmp * waveK[w][d] * scale;
      }
    }
  }

  void P2PEwald(int ibegin, int iend, int jbegin, int jend, real *Xperiodic,
		real cutoff, real alpha, real (*Ibodies)[4], real (*Jbodies)[4]) const {
    for (int i=ibegin; i<iend; i++) {
      real Po = 0, Fx = 0, Fy = 0, Fz = 0;
      for (int j=jbegin; j<jend; j++) {
	real dist[3];
	for_3d dist[d] = Jbodies[i][d] - Jbodies[j][d] - Xperiodic[d];
	real R2 = dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2];
	if (0 < R2 && R2 < cutoff * cutoff) {
	  real R2s = R2 * alpha * alpha;
	  real Rs = sqrtf(R2s);
	  real invRs = 1 / Rs;
	  real invR2s = invRs * invRs;
	  real invR3s = invR2s * invRs;
	  real dtmp = Jbodies[j][3] * (M_2_SQRTPI * exp(-R2s) * invR2s + erfc(Rs) * invR3s);
	  dtmp *= alpha * alpha * alpha;
	  Po += Jbodies[j][3] * erfc(Rs) * invRs * alpha;
	  Fx += dist[0] * dtmp;
	  Fy += dist[1] * dtmp;
	  Fz += dist[2] * dtmp;
	}
      }
      Ibodies[i][0] += Po;
      Ibodies[i][1] -= Fx;
      Ibodies[i][2] -= Fy;
      Ibodies[i][3] -= Fz;
    }
  }
  
public:
  void init(real _R0, real _X0[3]) {
    R0 = _R0;
    for_3d X0[d] = _X0[d];
  }
  
  void dipoleCorrection(int numBodies, real cycle, real (*Ibodies)[4], real (*Jbodies)[4]) {
    real dipole[3] = {0, 0, 0};
    for (int i=0; i<numBodies; i++) {
      for_3d dipole[d] += (Jbodies[i][d] - X0[d]) * Jbodies[i][3];
    }
    real norm = dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2];
    real coef = 4 * M_PI / (3 * cycle * cycle * cycle);
    for (int i=0; i<numBodies; i++) {
      Ibodies[i][0] -= coef * norm / numBodies / Jbodies[i][3];
      for_3d Ibodies[i][d+1] -= coef * dipole[d];
    }
  }

  int initWaves(int ksize, real *waveRe, real *waveIm, real (*waveK)[3]) {
    int numWaves = 0;
    for (int l=0; l<=ksize; l++) {
      int mmin = -ksize;
      if (l==0) mmin = 0;
      for (int m=mmin; m<=ksize; m++) {
	int nmin = -ksize;
	if (l==0 && m==0) nmin=1;
	for (int n=nmin; n<=ksize; n++) {
	  real k2 = l * l + m * m + n * n;
	  if (k2 <= ksize * ksize) {
	    waveK[numWaves][0] = l;
	    waveK[numWaves][1] = m;
	    waveK[numWaves][2] = n;
	    waveRe[numWaves] = waveIm[numWaves] = 0;
	    numWaves++;
	  }
	}
      }
    }
    return numWaves;
  }
  
  void ewald(int numBodies, int maxLevel, real cycle, real (*Ibodies2)[4],
	     real (*Jbodies)[4], int (*Leafs)[2]) {
    const int ksize = 11;
    const real alpha = 10 / cycle;
    const real sigma = .25 / M_PI;
    const real cutoff = 10;
    const real scale = 2 * M_PI / cycle;
    const real coef = .5 / M_PI / M_PI / sigma / cycle;
    const real coef2 = scale * scale / (4 * alpha * alpha);
    int numLeafs = 1 << 3 * maxLevel;
    int numWaves = 4. / 3 * M_PI * ksize * ksize * ksize;
    real *waveRe = new real [numWaves];
    real *waveIm = new real [numWaves];
    real (*waveK)[3] = new real [numWaves][3]();
    numWaves = initWaves(ksize, waveRe, waveIm, waveK);
    assert(numWaves < 4. / 3 * M_PI * ksize * ksize * ksize);
    dft(numWaves, numBodies, scale, waveRe, waveIm, waveK, Jbodies);
#pragma omp parallel for
    for (int w=0; w<numWaves; w++) {
      real k2 = 0;
      for_3d k2 += waveK[w][d] * waveK[w][d];
      real factor = coef * exp(-k2 * coef2) / k2;
      waveRe[w] *= factor;
      waveIm[w] *= factor;
    }
    idft(numWaves, numBodies, scale, waveRe, waveIm, waveK, Ibodies2, Jbodies);
    delete[] waveRe;
    delete[] waveIm;
    delete[] waveK;

    int nunit = 1 << maxLevel;
    int nmin = -nunit;
    int nmax = 2 * nunit - 1;
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      int ix[3] = {0, 0, 0};
      getIndex(ix,i);
      int jxmin[3];
      for_3d jxmin[d] = MAX(nmin, ix[d] - 2);
      int jxmax[3];
      for_3d jxmax[d] = MIN(nmax, ix[d] + 2);
      int jx[3];
      for (jx[2]=jxmin[2]; jx[2]<=jxmax[2]; jx[2]++) {
	for (jx[1]=jxmin[1]; jx[1]<=jxmax[1]; jx[1]++) {
	  for (jx[0]=jxmin[0]; jx[0]<=jxmax[0]; jx[0]++) {
	    int jxp[3];
	    for_3d jxp[d] = (jx[d] + nunit) % nunit;
	    int j = getKey(jxp,maxLevel);
	    real Xperiodic[3] = {0, 0, 0};
	    for_3d jxp[d] = (jx[d] + nunit) / nunit;
	    for_3d Xperiodic[d] = (jxp[d] - 1) * 2 * R0;
	    P2PEwald(Leafs[i][0],Leafs[i][1],Leafs[j][0],Leafs[j][1],Xperiodic,
		     cutoff,alpha,Ibodies2,Jbodies);
	  }
	}
      }
    }
    for (int i=0; i<numBodies; i++) {
      Ibodies2[i][0] -= M_2_SQRTPI * Jbodies[i][3] * alpha;
    }
  }
};
