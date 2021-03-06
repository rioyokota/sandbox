#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <vector>
#include "vec.h"

typedef double real_t;
typedef vec<3,real_t> vec3;

struct Body {
  vec3 X;
  int I;
  bool operator<(const Body& B) const {return I < B.I;}
};
typedef std::vector<Body> Bodies;
typedef Bodies::iterator B_iter;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

Bodies initBodies(int numBodies, real_t R0) {
  Bodies bodies(numBodies);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    real_t R = R0 * drand48();
    real_t theta = 2 * M_PI * drand48();
    real_t phi = M_PI * drand48();
    B->X[0] = R * cos(theta) * sin(phi);
    B->X[1] = R * sin(theta) * sin(phi);
    B->X[2] = R * cos(phi)*0;
  }
  return bodies;
}

inline int nearest(B_iter B, B_iter C0, int numClusters) {
  int index = B->I;
  real_t R2min = HUGE_VAL;
  for (int c=0; c<numClusters; c++) {
    B_iter C = C0 + c;
    vec3 dX = B->X - C->X;
    real_t R2 = norm(dX);
    if (R2min > R2) {
      R2min = R2;
      index = c;
    }
  }
  return index;
}

void initCluster(Bodies & bodies, Bodies & clusters) {
  for (B_iter C=clusters.begin(); C!=clusters.end(); C++) {
    *C = bodies[rand() % bodies.size()];
  }
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->I = nearest(B, clusters.begin(), clusters.size());
  }
}

Bodies setCluster(Bodies & bodies, int numClusters) {
  Bodies clusters(numClusters);
  initCluster(bodies, clusters);
  B_iter C0 = clusters.begin();
  unsigned changed;
  do {
    for (B_iter C=clusters.begin(); C!=clusters.end(); C++) {
      C->I = 0;
      C->X = 0;
    }
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
      B_iter C = C0 + B->I;
      C->I++;
      C->X += B->X;
    }
    for (B_iter C=clusters.begin(); C!=clusters.end(); C++) {
      C->X /= C->I;
      C->I = C-C0;
    }
    changed = 0;
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
      int index = nearest(B, C0, numClusters);
      if (index != B->I) {
	changed++;
	B->I = index;
      }
    }
  } while (changed > (bodies.size() >> 10));
  return clusters;
}

void writeBodies(Bodies bodies, Bodies clusters) {
  FILE *fid = fopen("kmeans.dat","w");
  int index = -1;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    if (B->I != index) {
      B_iter C = clusters.begin()+B->I;
      fprintf(fid, "%d %g %g %g\n", C->I, C->X[0], C->X[1], C->X[2]);
      assert(B->I == C->I);
      index = B->I;
    }
    fprintf(fid, "%d %g %g %g\n", B->I, B->X[0], B->X[1], B->X[2]);
  }
  fclose(fid);
}

int main() {
  const int numBodies = 100000;
  const int numClusters = 64;
  const real_t R0 = 10;
  double tic = get_time();
  Bodies bodies = initBodies(numBodies, R0);
  double toc = get_time();
  printf("initBodies   : %lf s\n",toc-tic);
  Bodies clusters = setCluster(bodies, numClusters);
  tic = get_time();
  printf("setCluster   : %lf s\n",tic-toc);
  std::sort(bodies.begin(),bodies.end());
  toc = get_time();
  printf("sortBodies   : %lf s\n",toc-tic);
  writeBodies(bodies, clusters);
  tic = get_time();
  printf("writeBodies  : %lf s\n",tic-toc);
}
