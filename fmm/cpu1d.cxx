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
  const int N = 40; // Number of points
  const int P = 10; // Order of multipole expansions
  const int pointsPerLeaf = 4; // Number of points per leaf cell
  const float eps = 1e-6; // Epsilon
  // Allocate memory
  int * ix = new int [N]; // Index
  int * iy = new int [N]; // Index
  float * x = new float [N]; // Target coordinates
  float * y = new float [N]; // Source coordinates
  float * p = new float [N]; // Target values
  float * w = new float [N]; // Source values
  // Initialize variables
  for (int i=0; i<N; i++) {
    x[i] = (i + 0.5) / N; // Random number [0,1]
    p[i] = 0;
  }
  for (int i=0; i<N; i++) {
    y[i] = (i + 0.5) / N; // Random number [0,1]
    w[i] = 1;
  }
  double tic = get_time();
  // Min and Max of x
  float xmin = x[0], xmax = x[0]; // Initialize xmin, xmax
  for (int i=1; i<N; i++) {
    xmin = x[i] < xmin ? x[i] : xmin; // xmin = min(x[i],xmin)
    xmax = x[i] > xmax ? x[i] : xmax; // xmax = max(x[i],xmax)
  }
  for (int i=1; i<N; i++) {
    xmin = y[i] < xmin ? y[i] : xmin; // xmin = min(y[i],xmin)
    xmax = y[i] > xmax ? y[i] : xmax; // xmax = max(y[i],xmax)
  }
  xmin -= eps; // Give some leeway to avoid points on boundary
  xmax += eps;
  // Assign cell index to points
  const int numLevels = log(N/pointsPerLeaf) / M_LN2; // Depth of binary tree
  const int maxLevel = numLevels - 1; // Level starts from 0
  const int numLeafs = 1 << maxLevel; // Cells at the bottom of binary tree
  const int numCells = 2 * numLeafs; // Total number of cells in binary tree
  const float leafSize = (xmax - xmin) / numLeafs; // Length of the leaf cell
  for (int i=0; i<N; i++)
    ix[i] = (x[i] - xmin) / leafSize; // Group points according to leaf cell's ix
  // Allocate multipole & local expansions
  float (*m)[P] = new float [numCells][P]; // Multipole expansion coefficients
  float (*l)[P] = new float [numCells][P]; // Local expansion coefficients
  for (int i=0; i<numCells; i++)
    for (int n=0; n<P; n++) m[i][n] = l[i][n] = 0;
  // P2M
  int offset = ((1 << maxLevel) - 1);
  for (int i=0; i<N; i++) {
    const float xCell = leafSize * (ix[i] + .5) + xmin;
    const float dx = xCell - x[i];
    float M[P] = {0};
    M[0] = w[i];
    for (int n=1; n<P; n++) {
      M[n] = M[n-1] * dx / n;
    }
    for (int n=0; n<P; n++) {
      m[ix[i]+offset][n] += M[n];
    }
  }
  // M2M
  for (int level=maxLevel; level>=3; level--) {
    const int joffset = ((1 << level) - 1);
    const int ioffset = ((1 << (level-1)) - 1);
    for (int jcell=0; jcell<(1<<level); jcell++) {
      const int icell = jcell / 2;
      const float dx = (xmax - xmin) / (1 << (level+1)) * (1 - 2 * (jcell & 1));
      float C[P];
      C[0] = 1;
      for (int n=1; n<P; n++) {
	C[n] = C[n-1] * dx / n;
      }
      for (int n=0; n<P; n++) {
	for (int k=0; k<=n; k++) {
	  m[icell+ioffset][n] += C[n-k] * m[jcell+joffset][k];
	}
      }
    }
  }
  // M2L
  for (int level=2; level<=maxLevel; level++) {
    const int cellsPerLevel = 1 << level;
    offset = ((1 << level) - 1);
    for (int icell=0; icell<cellsPerLevel; icell++) {
      for (int jcell=0; jcell<cellsPerLevel; jcell++) {
	if(jcell < icell-1 || icell+1 < jcell) {
	  const float dx = (icell - jcell) * (xmax - xmin) / cellsPerLevel;
	  const float invR2 = 1 / (dx * dx);
	  const float invR = m[jcell+offset][0] * sqrtf(invR2);
	  float C[P];
	  C[0] = invR;
	  C[1] = -dx * C[0] * invR2;
	  for (int n=2; n<P; n++) {
	    C[n] = ((1 - 2 * n) * dx * C[n-1] + (1 - n) * C[n-2]) / n * invR2;
	  }
	  float fact = 1;
	  for (int n=1; n<P; n++) {
	    fact *= n;
	    C[n] *= fact;
	  }
	  for (int i=0; i<P; i++) l[icell+offset][i] += C[i];
	  for (int n=0; n<P; n++) {
	    for (int k=1; k<P-n; k++) {
	      l[icell+offset][n] += m[jcell+offset][k] * C[n+k];
	    }
	  }
	}
      }
    }
  }
  // L2L
  for (int level=3; level<=maxLevel; level++) {
    const int joffset = ((1 << (level-1)) - 1);
    const int ioffset = ((1 << level) - 1);
    for (int icell=0; icell<(1<<level); icell++) {
      const int jcell = icell / 2;
      const float dx = (xmax - xmin) / (1 << (level+1)) * (2 * (icell & 1) - 1);
      float C[P];
      C[0] = 1;
      for (int n=1; n<P; n++) {
	C[n] = C[n-1] * dx / n;
      }
      for (int n=0; n<P; n++) l[icell+ioffset][n] += l[jcell+joffset][n];
      for (int n=0; n<P; n++) {
	for (int k=1; k<P-n; k++) {
	  l[icell+ioffset][n] += C[k] * l[jcell+joffset][n+k];
	}
      }
    }
  }
  // L2P
  offset = ((1 << maxLevel) - 1);
  for (int i=0; i<N; i++) {
    const float xCell = leafSize * (ix[i] + .5) + xmin;
    const float dx = x[i] - xCell;
    float C[P];
    C[0] = 1;
    for (int n=1; n<P; n++) {
      C[n] = C[n-1] * dx / n;
    }
    for (int n=0; n<P; n++) {
      p[i] += l[ix[i]+offset][n] * C[n];
    }
  }
  // P2P
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      if(abs(ix[i] - ix[j]) < 2 && i != j) {
	const float R = fabs(x[i] - x[j]);
	p[i] += w[j] / R;
      }
    }
    printf("%d %f\n",i,p[i]);
  }

  double toc = get_time();
  std::cout << "FMM    : " << toc-tic << std::endl;
  // Direct summation
  double dif = 0, nrm = 0;
  for (int i=0; i<N; i++) {
    float pi = 0;
    for (int j=0; j<N; j++) {
      if (i != j) {
	const float R = fabs(x[i] - x[j]);
	pi += w[j] / R;
      }
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
