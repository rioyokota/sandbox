#include <cassert>
#include <cstdlib>
#include <cstdio>
#include "kernels.h"

class Ewald {
private:
  real R0;
  real X0[3];

public:
  void init(real r0, real x0[3]) {
    R0 = r0;
    for_3d X0[d] = x0[d];
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
};

class Fmm : public Kernel {
private:
  void sort(real (*bodies)[4], real (*bodies2)[4], int *key) const {
    int Imax = key[0];
    int Imin = key[0];
    for( int i=0; i<numBodies; i++ ) {
      Imax = MAX(Imax,key[i]);
      Imin = MIN(Imin,key[i]);
    }
    int numBucket = Imax - Imin + 1;
    int *bucket = new int [numBucket];
    for (int i=0; i<numBucket; i++) bucket[i] = 0;
    for (int i=0; i<numBodies; i++) bucket[key[i]-Imin]++;
    for (int i=1; i<numBucket; i++) bucket[i] += bucket[i-1];
    for (int i=numBodies-1; i>=0; i--) {
      bucket[key[i]-Imin]--;
      int inew = bucket[key[i]-Imin];
      for_4d bodies2[inew][d] = bodies[i][d];
    }
    delete[] bucket;
  }

public:
  void allocate(int NumBodies, int MaxLevel, int NumNeighbors, int NumImages) {
    numBodies = NumBodies;
    maxLevel = MaxLevel;
    numNeighbors = NumNeighbors;
    numImages = NumImages;
    numCells = ((1 << 3 * (maxLevel + 1)) - 1) / 7;
    numLeafs = 1 << 3 * maxLevel;
    Ibodies = new real [numBodies][4]();
    Ibodies2 = new real [numBodies][4]();
    Jbodies = new real [numBodies][4]();
    Multipole = new real [numCells][MTERM]();
    Local = new real [numCells][LTERM]();
    Leafs = new int [numLeafs][2]();
    for (int i=0; i<numCells; i++) {
      for_m Multipole[i][m] = 0;
      for_l Local[i][l] = 0;
    }
    for (int i=0; i<numLeafs; i++) {
      Leafs[i][0] = Leafs[i][1] = 0;
    }
  }

  void deallocate() {
    delete[] Ibodies;
    delete[] Ibodies2;
    delete[] Jbodies;
    delete[] Multipole;
    delete[] Local;
    delete[] Leafs;
  }

  void initBodies(real cycle) {
    int ix[3] = {0, 0, 0};
    R0 = cycle * .5;
    for_3d X0[d] = R0;
    srand48(0);
    real average = 0;
    for (int i=0; i<numBodies; i++) {
      Jbodies[i][0] = 2 * R0 * (drand48() + ix[0]);
      Jbodies[i][1] = 2 * R0 * (drand48() + ix[1]);
      Jbodies[i][2] = 2 * R0 * (drand48() + ix[2]);
      Jbodies[i][3] = (drand48() - .5) / numBodies;
      average += Jbodies[i][3];
    }
    average /= numBodies;
    for (int i=0; i<numBodies; i++) {
      Jbodies[i][3] -= average;
    }
  }

  void sortBodies() const {
    int *key = new int [numBodies];
    real diameter = 2 * R0 / (1 << maxLevel);
    int ix[3] = {0, 0, 0};
    for (int i=0; i<numBodies; i++) {
      for_3d ix[d] = int((Jbodies[i][d] + R0 - X0[d]) / diameter);
      key[i] = getKey(ix,maxLevel);
    }
    sort(Jbodies,Ibodies,key);
    for (int i=0; i<numBodies; i++) {
      for_4d Jbodies[i][d] = Ibodies[i][d];
      for_4d Ibodies[i][d] = 0;
    }
    delete[] key;
  }

  void fillLeafs() const {
    real diameter = 2 * R0 / (1 << maxLevel);
    int ix[3] = {0, 0, 0};
    for_3d ix[d] = int((Jbodies[0][d] + R0 - X0[d]) / diameter);
    int ileaf = getKey(ix,maxLevel,false);
    Leafs[ileaf][0] = 0;
    for (int i=0; i<numBodies; i++) {
      for_3d ix[d] = int((Jbodies[i][d] + R0 - X0[d]) / diameter);
      int inew = getKey(ix,maxLevel,false);
      if (ileaf != inew) {
        Leafs[ileaf][1] = Leafs[inew][0] = i;
        ileaf = inew;
      }
    }
    Leafs[ileaf][1] = numBodies;
    for (int i=0; i<numLeafs; i++) {
      if (Leafs[i][1] == Leafs[i][0]) printf("Warning: Cell %d is empty.\n",i);
    }
  }

  void direct() {
    real Ibodies3[4], Jbodies2[4], dX[3];
    int range = (pow(3,numImages) - 1) / 2;
    for (int i=0; i<100; i++) {
      for_4d Ibodies3[d] = 0;
      for_4d Jbodies2[d] = Jbodies[i][d];
      int jx[3];
      for (jx[2]=-range; jx[2]<=range; jx[2]++) {
	for (jx[1]=-range; jx[1]<=range; jx[1]++) {
	  for (jx[0]=-range; jx[0]<=range; jx[0]++) {	
	    for (int j=0; j<numBodies; j++) {
	      for_3d dX[d] = Jbodies2[d] - Jbodies[j][d] - jx[d] * 2 * R0;
	      real R2 = dX[0] * dX[0] + dX[1] * dX[1] + dX[2] * dX[2];
	      real invR2 = R2 == 0 ? 0 : 1.0 / R2;
	      real invR = Jbodies[j][3] * sqrtf(invR2);
	      for_3d dX[d] *= invR2 * invR;
	      Ibodies3[0] += invR;
	      Ibodies3[1] -= dX[0];
	      Ibodies3[2] -= dX[1];
	      Ibodies3[3] -= dX[2];
	    }
	  }
	}
      }
      for_4d Ibodies2[i][d] = Ibodies3[d];
    }
  }

  /*
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
  */
  
  inline void getIndex2(int *ix, int index) const {
    for_3d ix[d] = 0;
    int d = 0, level = 0;
    while (index != 0) {
      ix[d] += (index & 1) * (1 << level);
      index >>= 1;
      d = (d+1) % 3;
      if (d == 0) level++;
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
      getIndex2(ix,i);
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
	    int j = getKey(jxp,maxLevel,false);
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
  
  void verify(int numTargets, real & potDif, real & potNrm, real & accDif, real & accNrm) {
    real potSum = 0, potSum2 = 0;
    for (int i=0; i<numTargets; i++) {
      potSum += Ibodies[i][0] * Jbodies[i][3];
      potSum2 += Ibodies2[i][0] * Jbodies[i][3];
    }
    potDif = (potSum - potSum2) * (potSum - potSum2);
    potNrm = potSum2 * potSum2;
    for (int i=0; i<numTargets; i++) {
      for_3d accDif += (Ibodies[i][d+1] - Ibodies2[i][d+1]) * (Ibodies[i][d+1] - Ibodies2[i][d+1]);
      for_3d accNrm += (Ibodies2[i][d+1] * Ibodies2[i][d+1]);
    }
  }
};
