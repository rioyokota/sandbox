#include "octree.h"

int main() {
  for( int it=0; it<25; it++ ) {
    uint numBodies = uint(pow(10,(it+24)/8.0));
    octree *tree = new octree(numBodies);
    printf("N     : %d\n",numBodies);
    for( uint i=0; i<numBodies; i++ ) {
      tree->Body_X[i][0]  = drand48();
      tree->Body_X[i][1]  = drand48();
      tree->Body_X[i][2]  = drand48();
      tree->Body_SRC[i]  = 1. / numBodies;
    }
    tree->Body_X.h2d();
    tree->Body_SRC.h2d();
    tree->iterate(); 
    double tic = tree->get_time();
    tree->direct();
    double toc = tree->get_time();
    tree->Body_TRG.d2h();
    tree->Body2_TRG.d2h();
    float diff1 = 0, norm1 = 0, diff2 = 0, norm2 = 0;
    for( uint i=0; i<numBodies/100; i++ ) {
      vec4 fapprox = tree->Body_TRG[i];
      vec4 fdirect = tree->Body2_TRG[i];
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
