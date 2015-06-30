#include <cmath>
#include "types.h"

class Kernel {
public:
  int maxLevel;
  int numBodies;
  int numCells;
  int numLeafs;
  int numNeighbors;

  real X0;
  real R0;
  real (*Ibodies)[2];
  real (*Ibodies2)[2];
  real (*Jbodies)[2];
  real (*Multipole)[PP];
  real (*Local)[PP];
  int (*Leafs)[2];

private:
  void P2PSum(int ibegin, int iend, int jbegin, int jend) const {
    for (int i=ibegin; i<iend; i++) {
      real Po = 0, Fx = 0;
      for (int j=jbegin; j<jend; j++) {
	real dx = Jbodies[i][0] - Jbodies[j][0];
	real R2 = dx * dx;
	real invR2 = R2 == 0 ? 0 : 1.0 / R2;
	real invR = Jbodies[j][1] * sqrt(invR2);
	real invR3 = invR2 * invR;
	Po += invR;
	Fx += dx * invR3;
      }
      Ibodies[i][0] += Po;
      Ibodies[i][1] -= Fx;
    }
  }
  
public:
  void P2P() const {
    int nunit = 1 << maxLevel;
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      int jmin = MAX(0, i - numNeighbors);
      int jmax = MIN(nunit - 1, i + numNeighbors);
      for (int j=jmin; j<=jmax; j++) {
	P2PSum(Leafs[i][0],Leafs[i][1],Leafs[j][0],Leafs[j][1]);
      }
    }
  }

  void P2M() const {
    int levelOffset = ((1 << maxLevel) - 1);
    real R = R0 / (1 << maxLevel);
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      real center = X0 - R0 + (2 * i + 1) * R;
      for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
        real dx = center - Jbodies[j][0];
        real M[PP];
        M[0] = Jbodies[j][1];
	for (int n=1; n<PP; n++) {
	  M[n] = M[n-1] * dx / n;
	}
	for (int n=0; n<PP; n++) {
	  Multipole[i+levelOffset][n] += M[n];
	}
      }
    }
  }

  void M2M() const {
    for (int lev=maxLevel; lev>0; lev--) {
      int childOffset = ((1 << lev) - 1);
      int parentOffset = ((1 << (lev - 1)) - 1);
      real radius = R0 / (1 << lev);
#pragma omp parallel for schedule(static, 4)
      for (int i=0; i<(1 << lev); i++) {
        int c = i + childOffset;
        int p = (i >> 1) + parentOffset;
        real dx = (1 - (i & 1) * 2) * radius;
        real C[PP];
        C[0] = 1;
	for (int n=1; n<PP; n++) {
	  C[n] = C[n-1] * dx / n;
	}
	for (int n=0; n<PP; n++) {
	  for (int k=0; k<=n; k++) {
	    Multipole[p][n] += C[n-k] * Multipole[c][k];
	  }
	}
      }
    }
  }

  void M2L() const {
    for (int lev=1; lev<=maxLevel; lev++) {
      int levelOffset = ((1 << lev) - 1);
      int nunit = 1 << lev;
      real diameter = 2 * R0 / (1 << lev);
#pragma omp parallel for
      for (int i=0; i<(1 << lev); i++) {
        real L[PP];
	for (int n=0; n<PP; n++) {
	  L[n] = 0;
	}
        int jmin =  MAX(0, (i >> 1) - numNeighbors) << 1;
        int jmax = (MIN((nunit >> 1) - 1, (i >> 1) + numNeighbors) << 1) + 1;
	for (int j=jmin; j<=jmax; j++) {
	  if(j < i-numNeighbors || i+numNeighbors < j) {
	    real dx = (i - j) * diameter;
	    real invR2 = 1. / (dx * dx);
	    real invR  = sqrt(invR2);
	    real C[PP];
	    C[0] = invR;
	    C[1] = -dx * C[0] * invR2;
	    for (int n=2; n<PP; n++) {
	      C[n] = ((1 - 2 * n) * dx * C[n-1] + (1 - n) * C[n-2]) / n * invR2;
	    }
	    float fact = 1;
	    for (int n=1; n<PP; n++) {
	      fact *= n;
	      C[n] *= fact;
	    }
	    for (int k=0; k<PP; k++) {
	      L[0] += Multipole[j+levelOffset][k] * C[k];
	    }
	    for (int n=1; n<PP; n++) {
	      for (int k=0; k<PP-n; k++) {
		L[n] += Multipole[j+levelOffset][k] * C[n+k];
	      }
	    }
	  }
	}
	for (int n=0; n<PP; n++) {
	  Local[i+levelOffset][n] += L[n];
	}
      }
    }
  }

  void L2L() const {
    for (int lev=1; lev<=maxLevel; lev++) {
      int childOffset = ((1 << lev) - 1);
      int parentOffset = ((1 << (lev - 1)) - 1);
      real radius = R0 / (1 << lev);
#pragma omp parallel for
      for (int i=0; i<(1 << lev); i++) {
        int c = i + childOffset;
        int p = (i >> 1) + parentOffset;
        real dx = ((i & 1) * 2 - 1) * radius;
        real C[PP];
        C[0] = 1;
	for (int n=1; n<PP; n++) {
	  C[n] = C[n-1] * dx / n;
	}
	for (int n=0; n<PP; n++) {
	  for (int k=n; k<PP; k++) {
	    Local[c][n] += C[k-n] * Local[p][k];
	  }
	}
      }
    }
  }

  void L2P() const {
    int levelOffset = ((1 << maxLevel) - 1);
    real R = R0 / (1 << maxLevel);
#pragma omp parallel for
    for (int i=0; i<numLeafs; i++) {
      real center = X0 - R0 + (2 * i + 1) * R;
      real L[PP];
      for (int n=0; n<PP; n++) {
	L[n] = Local[i+levelOffset][n];
      }
      for (int j=Leafs[i][0]; j<Leafs[i][1]; j++) {
        real dx = Jbodies[j][0] - center;
        real C[PP];
        C[0] = 1;
	for (int n=1; n<PP; n++) {
	  C[n] = C[n-1] * dx / n;
	}
        for (int l=0; l<PP; l++) Ibodies[j][0] += C[l] * L[l];
	for (int l=0; l<PP-1; l++) Ibodies[j][1] += C[l] * L[l+1];
      }
    }
  }
};
