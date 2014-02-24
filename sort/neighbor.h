#include <cassert>
#include <cmath>
#include <omp.h>

class Neighbor {
private:
  static const int NUM_PER_CELL = 10;                            // Number of points per leaf cell
  static const int OMP_NUM_THREADS = 32;                         // Number of OpenMP threads
  const int N;                                                   // Number of points
  const int LEVEL;                                               // Number of tree levels
  double X0[3];                                                  // Center of domain
  double R0;                                                     // Radius of domain
  double XMIN[3];                                                // Minimum of X
  double SIZE;                                                   // Diameter of leaf cell
  int *KEY;                                                      // Morton key
  int *KBUFFER;                                                  // Morton key buffer
  double *XBUFFER[3];                                            // X buffer
  int *PERMUTATION;                                              // Permutation index
  int *MAP;                                                      // Map from Morton key to cell index
  int *BEGIN;                                                    // Begin index of point in each cell
  int *END;                                                      // End index of point in each cell

  void getIndex3(double &x, double &y, double &z, int *iX) {
    iX[0] = int( (x - XMIN[0]) / SIZE );
    iX[1] = int( (y - XMIN[1]) / SIZE );
    iX[2] = int( (z - XMIN[2]) / SIZE );
  }

  int getMorton(int *iX) {
    int key = 0;
    for (int l=0; l<LEVEL; l++) {
      for (int d=0; d!=3; d++) {
        key += iX[d] % 2 << (3 * l + d);
        iX[d] >>= 1;
      }
    }
    return key;
  }

  int getNumKeys() {
    int numKey = 1;
    int oldKey = KEY[0];
    for (int i=0; i<N; i++) {
      if (KEY[i] != oldKey) {
        oldKey = KEY[i];
        numKey++;
      }
    }
    return numKey;
  }

public:
  Neighbor(int n) : N(n), LEVEL(N >= NUM_PER_CELL ? 1 + int( log(N / NUM_PER_CELL) / M_LN2 / 3 ) : 0) {}
  ~Neighbor() {}

  void getBounds(double **X) {
    double Xmin[3], Xmax[3];
    for (int d=0; d<3; d++) Xmin[d] = Xmax[d] = X[d][0];
    for (int i=1; i<N; i++) {
      for (int d=0; d<3; d++) {
	Xmin[d] = Xmin[d] < X[d][i] ? Xmin[0] : X[d][i];
	Xmax[d] = Xmax[d] > X[d][i] ? Xmax[d] : X[d][i];
      }
    }
    R0 = 0;
    for (int d=0; d<3; d++) {
      double R = (Xmax[d] - Xmin[d]) * .5;
      R0 = R0 > R ? R0 : R;
      X0[d] = (Xmax[d] + Xmin[d]) * .5;
    }
    R0 *= 1.000001;
    SIZE = R0 / (1 << (LEVEL-1));
    for (int d=0; d<3; d++) XMIN[d] = X0[d] - R0;
  }

  void setKeys(double **X) {
    KEY = new int [N];
#pragma omp parallel for
    for (int i=0; i<N; i++) {
      int iX[3];
      getIndex3(X[0][i],X[1][i],X[2][i],iX);
      KEY[i] = getMorton(iX);
    }
  }

  void buffer(double **X) {
    KBUFFER = new int [N];
    for (int d=0; d<3; d++) XBUFFER[d] = new double [N];
    PERMUTATION = new int [N];
#pragma omp parallel for
    for (int i=0; i<N; i++) {
      KBUFFER[i] = KEY[i];
      PERMUTATION[i] = i;
    }
    for (int d=0; d<3; d++) {
#pragma omp parallel for
      for (int i=0; i<N; i++) XBUFFER[d][i] = X[d][i];
    }
  }

  void radixSort() {
    const int bitStride = 8;
    const int stride = 1 << bitStride;
    const int mask = stride - 1;
    int *ibuffer = new int [N];
    int *pbuffer = new int [N];
#pragma omp parallel
    assert(omp_get_num_threads() <= OMP_NUM_THREADS);
    int (*bucketPerThread)[stride] = new int [OMP_NUM_THREADS][stride]();
    int iMaxPerThread[OMP_NUM_THREADS] = {0};
#pragma omp parallel for
    for (int i=0; i<N; i++)
      if (KEY[i] > iMaxPerThread[omp_get_thread_num()])
	iMaxPerThread[omp_get_thread_num()] = KEY[i];
    int iMax = 0;
    for (int i=0; i<OMP_NUM_THREADS; i++)
      if (iMaxPerThread[i] > iMax) iMax = iMaxPerThread[i];
    while (iMax > 0) {
      int bucket[stride] = {0};
      for (int t=0; t<OMP_NUM_THREADS; t++)
	for (int i=0; i<stride; i++)
	  bucketPerThread[t][i] = 0;
#pragma omp parallel for
      for (int i=0; i<N; i++)
	bucketPerThread[omp_get_thread_num()][KEY[i] & mask]++;
      for (int t=0; t<OMP_NUM_THREADS; t++)
	for (int i=0; i<stride; i++)
	  bucket[i] += bucketPerThread[t][i];
      for (int i=1; i<stride; i++)
	bucket[i] += bucket[i-1];
      for (int i=N-1; i>=0; i--)
	pbuffer[i] = --bucket[KEY[i] & mask];
#pragma omp parallel for
      for (int i=0; i<N; i++)
	ibuffer[pbuffer[i]] = PERMUTATION[i];
#pragma omp parallel for
      for (int i=0; i<N; i++)
	PERMUTATION[i] = ibuffer[i];
#pragma omp parallel for
      for (int i=0; i<N; i++)
	ibuffer[pbuffer[i]] = KEY[i];
#pragma omp parallel for
      for (int i=0; i<N; i++)
	KEY[i] = ibuffer[i] >> bitStride;
      iMax >>= bitStride;
    }
    delete[] ibuffer;
    delete[] pbuffer;
    delete[] bucketPerThread;
  }

  void permute(double **X) {
#pragma omp parallel for
    for (int i=0; i<N; i++) {
      KEY[i] = KBUFFER[PERMUTATION[i]];
    }
    for (int d=0; d<3; d++) {
#pragma omp parallel for
      for (int i=0; i<N; i++) X[d][i] = XBUFFER[d][PERMUTATION[i]];
    }
    delete[] KBUFFER;
    for (int d=0; d<3; d++) delete[] XBUFFER[d];
    delete[] PERMUTATION;
  }

  void setRange() {
    int numKey = getNumKeys();
    int maxKey = 1 << (3*LEVEL);
    MAP = new int [maxKey];
    BEGIN = new int [numKey];
    END = new int [numKey];
    int oldKey = KEY[0];
    for (int i=0; i<maxKey; i++) MAP[i] = -1;
    int cell = 0;
    MAP[oldKey] = cell;
    BEGIN[cell] = 0;
    for (int i=0; i<N; i++) {
      if (KEY[i] != oldKey) {
	END[cell] = i;
	cell++;
	oldKey = KEY[i];
	MAP[oldKey] = cell;
	BEGIN[cell] = i;
      }
    }
    END[cell] = N;
  }

  void getNeighbor(double *Xi, double **Xj, double &R) {
    int iX[3],jX[3],ijX[3];
    getIndex3(Xi[0],Xi[1],Xi[2],iX);
    int irange = int(R/SIZE + .5);
    int numNeighbors = 0;
    for (jX[0]=-irange; jX[0]<=irange; jX[0]++) {
      for (jX[1]=-irange; jX[1]<=irange; jX[1]++) {
	for (jX[2]=-irange; jX[2]<=irange; jX[2]++) {
	  for (int d=0; d<3; d++) ijX[d] = iX[d] + jX[d];
	  int key = getMorton(ijX);
	  int cell = MAP[key];
	  if (cell != -1) {
	    for (int j=BEGIN[cell]; j<END[cell]; j++) {
	      double R2 = 0;
	      for (int d=0; d<3; d++)
		R2 += (Xi[d] - Xj[d][j]) * (Xi[d] - Xj[d][j]);
	      if (R2 < R*R) numNeighbors++;
	    }
	  }
	}
      }
    }
    std::cout << numNeighbors << std::endl;
  }
};
