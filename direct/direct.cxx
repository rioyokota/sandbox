#include <assert.h>
#include <cmath>
#include <stdio.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

void P2P(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for (int i=0; i<ni; i++) {
    float ax = 0;
    float ay = 0;
    float az = 0;
    float phi = 0;
    float xi = source[i].x;
    float yi = source[i].y;
    float zi = source[i].z;
    for (int j=0; j<nj; j++) {
      float dx = source[j].x - xi;
      float dy = source[j].y - yi;
      float dz = source[j].z - zi;
      float R2 = dx * dx + dy * dy + dz * dz + eps2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = source[j].w * invR * invR * invR;
      phi += source[j].w * invR;
      ax += dx * invR3;
      ay += dy * invR3;
      az += dz * invR3;
    }
    target[i].w = phi;
    target[i].x = ax;
    target[i].y = ay;
    target[i].z = az;
  }
}
