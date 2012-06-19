#include <openacc.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

void P2Phost(float4 *target, float4 *source, int ni, int nj, float eps2) {
  for( int i=0; i<ni; i++ ) {
    float4 t;
    for( int j=0; j<nj; j++ ) {
      dx = source[j].x - target[i].x;
      dy = source[j].y - target[i].y;
      dz = source[j].z - target[i].z;
      R2 = dx * dx + dy * dy + dz * dz + eps2;
      invR = 1. / sqrtf(R2);
      t.w += source[j].w * invR;
      invR3 = invR * invR * invR * source[j].w;
      t.x += dx * invR3;
      t.y += dy * invR3;
      t.z += dz * invR3;
    }
    target[i] = t;
  }
}
