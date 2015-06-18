#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Body {
  double x, y;
  int group;
};
typedef std::vector<Body> Bodies;
typedef Body* B_iter;

B_iter initBodies(int numBodies, double R0) {
  B_iter B0 = (B_iter) malloc(sizeof(Body) * numBodies);
  for (B_iter B=B0; B!=B0+numBodies; B++) {
    double theta = 2 * M_PI * drand48();
    double R = R0 * drand48();
    B->x = R * cos(theta);
    B->y = R * sin(theta);
  }
  return B0;
}

inline int nearest(B_iter B, B_iter C0, int numCluster) {
  int min_i = B->group;
  double min_d = HUGE_VAL;
  for (int i=0; i<numCluster; i++) {
    B_iter C = C0 + i;
    double dx = B->x - C->x;
    double dy = B->y - C->y;
    double d = dx * dx + dy * dy;
    if (min_d > d) {
      min_d = d;
      min_i = i;
    }
  }
  return min_i;
}

void initCluster(B_iter B0, int numBodies, B_iter C0, int numCluster) {
  for (int c=0; c<numCluster; c++) {
    C0[c] = B0[rand() % numBodies];
  }
  for (B_iter B=B0; B<B0+numBodies; B++) {
    B->group = nearest(B, C0, numCluster);
  }
}

B_iter setCluster(B_iter B0, int numBodies, int numCluster) {
  int min_i;
  B_iter C0 = (B_iter) malloc(sizeof(Body) * numCluster);
  initCluster(B0, numBodies, C0, numCluster);
  int changed;
  do {
    for (B_iter c=C0; c!=C0+numCluster; c++) {
      c->group = 0;
      c->x = 0;
      c->y = 0;
    }
    for (B_iter p=B0; p!=B0+numBodies; p++) {
      B_iter c = C0 + p->group;
      c->group++;
      c->x += p->x;
      c->y += p->y;
    }
    for (B_iter c=C0; c!=C0+numCluster; c++) {
      c->x /= c->group;
      c->y /= c->group;
    }

    changed = 0;
    /* find closest C0roid of each B_iter */
    for (B_iter p=B0; p!=B0+numBodies; p++) {
      min_i = nearest(p, C0, numCluster);
      if (min_i != p->group) {
	changed++;
	p->group = min_i;
      }
    }
  } while (changed > (numBodies >> 10)); /* stop when 99.9% of B_iters are good */

  for (int i=0; i<numCluster; i++) {
    B_iter c = C0+i;
    c->group = i;
  }

  return C0;
}

void writeBodies(B_iter B0, int numBodies, B_iter C0, int numCluster) {
  FILE *fid = fopen("kmeans.dat","w");
  for (B_iter C=C0; C!=C0+numCluster; C++) {
    fprintf(fid, "%ld %g %g\n", C-C0, C->x, C->y);
    for (B_iter B=B0; B!=B0+numBodies; B++) {
      if (B->group != C-C0) continue;
      fprintf(fid, "%ld %g %g\n", C-C0, B->x, B->y);
    }
  }
  fclose(fid);
}

int main() {
  const int numBodies = 100000;
  const int numClusters = 14;
  const double R0 = 10;
  B_iter B0 = initBodies(numBodies, R0);
  B_iter C0 = setCluster(B0, numBodies, numClusters);
  writeBodies(B0, numBodies, C0, numClusters);
  free(B0); free(C0);
}
