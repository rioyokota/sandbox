#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "vec.h"

typedef double real_t;
typedef vec<2,real_t> vec2;

struct Body {
  vec2 X;
  int IBODY;
};
typedef std::vector<Body> Bodies;
typedef Body* B_iter;

B_iter initBodies(int numBodies, real_t R0) {
  B_iter B0 = (B_iter) malloc(sizeof(Body) * numBodies);
  for (B_iter B=B0; B!=B0+numBodies; B++) {
    real_t theta = 2 * M_PI * drand48();
    real_t R = R0 * drand48();
    B->X[0] = R * cos(theta);
    B->X[1] = R * sin(theta);
  }
  return B0;
}

inline int nearest(B_iter B, B_iter C0, int numCluster) {
  int index = B->IBODY;
  real_t R2min = HUGE_VAL;
  for (int i=0; i<numCluster; i++) {
    B_iter C = C0 + i;
    vec2 dX = B->X - C->X;
    real_t R2 = norm(dX);
    if (R2min > R2) {
      R2min = R2;
      index = i;
    }
  }
  return index;
}

void initCluster(B_iter B0, int numBodies, B_iter C0, int numCluster) {
  for (int c=0; c<numCluster; c++) {
    C0[c] = B0[rand() % numBodies];
  }
  for (B_iter B=B0; B<B0+numBodies; B++) {
    B->IBODY = nearest(B, C0, numCluster);
  }
}

B_iter setCluster(B_iter B0, int numBodies, int numCluster) {
  B_iter C0 = (B_iter) malloc(sizeof(Body) * numCluster);
  initCluster(B0, numBodies, C0, numCluster);
  int changed;
  do {
    for (B_iter C=C0; C!=C0+numCluster; C++) {
      C->IBODY = 0;
      C->X = 0;
    }
    for (B_iter B=B0; B!=B0+numBodies; B++) {
      B_iter C = C0 + B->IBODY;
      C->IBODY++;
      C->X += B->X;
    }
    for (B_iter C=C0; C!=C0+numCluster; C++) {
      C->X /= C->IBODY;
      C->IBODY = C-C0;
    }
    changed = 0;
    for (B_iter B=B0; B!=B0+numBodies; B++) {
      int index = nearest(B, C0, numCluster);
      if (index != B->IBODY) {
	changed++;
	B->IBODY = index;
      }
    }
  } while (changed > (numBodies >> 10));
  return C0;
}

void writeBodies(B_iter B0, int numBodies, B_iter C0, int numCluster) {
  FILE *fid = fopen("kmeans.dat","w");
  for (B_iter C=C0; C!=C0+numCluster; C++) {
    fprintf(fid, "%ld %g %g\n", C-C0, C->X[0], C->X[1]);
    for (B_iter B=B0; B!=B0+numBodies; B++) {
      if (B->IBODY != C-C0) continue;
      fprintf(fid, "%ld %g %g\n", C-C0, B->X[0], B->X[1]);
    }
  }
  fclose(fid);
}

int main() {
  const int numBodies = 100000;
  const int numClusters = 14;
  const real_t R0 = 10;
  B_iter B0 = initBodies(numBodies, R0);
  B_iter C0 = setCluster(B0, numBodies, numClusters);
  writeBodies(B0, numBodies, C0, numClusters);
  free(B0); free(C0);
}
