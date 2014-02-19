#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>

// 1-D Fast Multipole Method code by Rio Yokota Jan. 10 2014
// Calculates p[i] = sum_{j=0}^{N} q[j]/(x[i]-y[j]) for i=1,N (i!=j)

// Get wall clock time
double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  const int nx = 40; // Number of target points
  const int ny = 40; // Number of source points
  const int P = 4; // Order of multipole expansions
  const int pointsPerLeaf = 4; // Number of points per leaf cell
  const float eps = 1e-6; // Epsilon
  // Allocate memory
  int * ix = new int [nx]; // Index
  int * iy = new int [ny]; // Index
  float * x = new float [nx]; // Target coordinates
  float * y = new float [ny]; // Source coordinates
  float * p = new float [nx]; // Target values
  float * w = new float [ny]; // Source values
  // Initialize variables
  for (int i=0; i<nx; i++) {
    x[i] = drand48(); // Random number [0,1]
    p[i] = 0;
  }
  for (int i=0; i<ny; i++) {
    y[i] = drand48(); // Random number [0,1]
    w[i] = drand48();
  }
  double tic = get_time();
  // Min and Max of x
  float xmin = x[0], xmax = x[0]; // Initialize xmin, xmax
  for (int i=1; i<nx; i++) {
    xmin = x[i] < xmin ? x[i] : xmin; // xmin = min(x[i],xmin)
    xmax = x[i] > xmax ? x[i] : xmax; // xmax = max(x[i],xmax)
  }
  for (int i=1; i<ny; i++) {
    xmin = y[i] < xmin ? y[i] : xmin; // xmin = min(y[i],xmin)
    xmax = y[i] > xmax ? y[i] : xmax; // xmax = max(y[i],xmax)
  }
  xmin -= eps; // Give some leeway to avoid points on boundary
  xmax += eps;
  // Assign cell index to points
  const int n = std::max(nx,ny); // Larger of the two
  const int numLevels = log(n/pointsPerLeaf) / M_LN2; // Depth of binary tree
  const int maxLevel = numLevels - 1; // Level starts from 0
  const int numLeafs = 1 << maxLevel; // Cells at the bottom of binary tree
  const int numCells = 2 * numLeafs; // Total number of cells in binary tree
  const float leafSize = (xmax - xmin) / numLeafs; // Length of the leaf cell
  for (int i=0; i<nx; i++)
    ix[i] = (x[i] - xmin) / leafSize; // Group points according to leaf cell's ix
  // Allocate multipole & local expansions
  float (*m)[P] = new float [numCells][P]; // Multipole expansion coefficients
  float (*l)[P] = new float [numCells][P]; // Local expansion coefficients
  for (int i=0; i<numCells; i++)
    for (int n=0; n<P; n++) m[i][n] = l[i][n] = 0;
  // P2M
  int offset = ((1 << maxLevel) - 1);
  for (int i=0; i<nx; i++) {
    const float xCell = leafSize * (ix[i] + .5) + xmin;
    const float dx = x[i] - xCell;
    float powX = 1.0;
    m[ix[i]+offset][0] += w[i];
    for (int n=1; n<P; n++) {
      powX *= dx;
      m[ix[i]+offset][n] -= w[i] * powX / n;
    }
  }
  // M2M
  for (int level=maxLevel; level>2; level--) {
    const int joffset = ((1 << level) - 1);
    const int ioffset = ((1 << (level-1)) - 1);
    for (int jcell=0; jcell<(1<<level); jcell++) {
      const int icell = jcell / 2;
      const float dx = (xmax - xmin) / (1 << (level+1)) * (1 - 2 * (jcell & 1));
      const float invX = 1.0 / dx;
      m[icell+ioffset][0] += m[jcell+joffset][0];
      float powXn = 1.0;
      for (int n=1; n<P; n++) {
	powXn *= dx;
	m[icell+ioffset][n] -= m[jcell+joffset][0] * powXn / n;
	float powXnk = powXn;
	float Cnk = 1.0;
	for (int k=1; k<=n; k++) {
	  powXnk *= invX;
	  m[icell+ioffset][n] += m[jcell+joffset][k] * powXnk * Cnk;
	  Cnk *= float(n - k) / k;
	}
      }
    }
  }
  // M2L
  for (int level=2; level<numLevels; level++) {
    const int cellsPerLevel = 1 << level;
    offset = ((1 << level) - 1);
    for (int icell=0; icell<cellsPerLevel; icell++) {
      for (int jcell=0; jcell<cellsPerLevel; jcell++) {
	if(jcell < icell-1 || icell+1 < jcell) {
	  const float dx = (icell - jcell) * (xmax - xmin) / cellsPerLevel;
	  const float invX = 1.0 / dx;
	  float powXn = 1.0;
	  l[icell+offset][0] += m[jcell+offset][0] * log(fabs(dx));
	  for (int k=1; k<P; k++) {
	    powXn *= invX;
	    l[icell+offset][0] += m[jcell+offset][k] * powXn;
	  }
	  powXn = 1;
	  for (int n=1; n<P; n++) {
	    powXn *= -invX;
	    l[icell+offset][n] -= m[jcell+offset][0] * powXn / n;
	    float powXnk = powXn;
	    float Cnk = 1.0;
	    for (int k=1; k<P; k++) {
	      powXnk *= invX;
	      l[icell+offset][n] += m[jcell+offset][k] * powXnk * Cnk;
	      Cnk *= float(n + 1) / k;
	    }
	  }
	}
      }
    }
  }
  // L2L
  for (int level=3; level<numLevels; level++) {
    const int joffset = ((1 << (level-1)) - 1);
    const int ioffset = ((1 << level) - 1);
    for (int icell=0; icell<(1<<level); icell++) {
      const int jcell = icell / 2;
      const float dx = (xmax - xmin) / (1 << (level+1)) * (1 - 2 * (icell & 1));
      for (int n=0; n<P; n++) {
	float powX = 1.0;
	float Cnk = 1.0;
	for (int k=n; k<P; k++) {
	  l[icell+ioffset][n] += l[jcell+joffset][k] * powX * Cnk;
	  powX *= dx;
	  Cnk *= float(k + 1) / (k - n + 1);
	}
      }
    }
  }
  // L2P
  offset = ((1 << maxLevel) - 1);
  for (int i=0; i<nx; i++) {
    const float xCell = leafSize * (ix[i] + .5) + xmin;
    const float dx = x[i] - xCell;
    float powX = 1;
    for (int n=0; n<P; n++) {
      p[i] -= l[ix[i]+offset][n] * powX;
      powX *= dx;
    }
  }
  // P2P
  for (int i=0; i<nx; i++) {
    for (int j=0; j<nx; j++) {
      if(abs(ix[i] - ix[j]) < 2) {
	const float R = fabs(x[i] - x[j]);
	const float logR = R != 0 ? log(R) : 0;
	p[i] -= w[j] * logR;
      }
    }
  }

  double toc = get_time();
  std::cout << "FMM    : " << toc-tic << std::endl;
  // Direct summation
  double dif = 0, nrm = 0;
  for (int i=0; i<nx; i++) {
    float pi = 0;
    for (int j=0; j<nx; j++) {
      const float R = fabs(x[i] - x[j]);
      const float logR = R != 0 ? log(R) : 0;
      pi -= w[j] * logR;
    }
    dif += (p[i] - pi) * (p[i] - pi);
    nrm += pi * pi;
  }
  tic = get_time();
  std::cout << "Direct : " << tic-toc << std::endl;
  std::cout << "Error  : " << sqrt(dif/nrm) << std::endl;
  // Free memory
  delete[] ix;
  delete[] iy;
  delete[] x;
  delete[] y;
  delete[] p;
  delete[] w;
  delete[] m;
  delete[] l;
}
