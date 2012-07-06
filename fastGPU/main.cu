#include "octree.h"

int main() {
  for( int it=0; it<25; it++ ) {
    uint numBodies = uint(pow(10,(it+24)/8.0));
    octree *tree = new octree(numBodies);
    printf("N     : %d\n",numBodies);
    for( uint i=0; i<numBodies; i++ ) {
      tree->bodyPos[i][3]  = 1. / numBodies;
      tree->bodyPos[i][0]  = drand48();
      tree->bodyPos[i][1]  = drand48();
      tree->bodyPos[i][2]  = drand48();
    }
    tree->bodyPos.h2d();
    tree->iterate(); 
    double tic = tree->get_time();
    tree->direct();
    double toc = tree->get_time();
    tree->bodyAcc.d2h();
    tree->bodyAcc2.d2h();
    float diff1 = 0, norm1 = 0, diff2 = 0, norm2 = 0;
    for( uint i=0; i<numBodies/100; i++ ) {
      vec4 fapprox = tree->bodyAcc[i];
      vec4 fdirect = tree->bodyAcc2[i];
      diff1 += (fapprox[3] - fdirect[3]) * (fapprox[3] - fdirect[3]);
      diff2 += norm(make_vec3(fapprox - fdirect));
      norm1 += fdirect[3] * fdirect[3];
      norm2 += norm(make_vec3(fdirect));
    }
    printf("Direct: %lf\n",toc-tic);
    printf("P Err : %f\n",sqrtf(diff1/norm1));
    printf("F Err : %f\n",sqrtf(diff2/norm2));
    delete tree;
  }
  return 0;
}
