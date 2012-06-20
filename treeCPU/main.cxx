#include <iostream>
#include <fstream>
#include <cstdlib>
#include "octree.h"
#include "plummer.h"

using namespace std;

int main(){
#if 0
  int n = 1<<20;
  vector<float4> vec(n);
  for(int i=0; i<n; i++){
    vec[i].x = rand() / (1. + RAND_MAX) - 0.5f;
    vec[i].y = rand() / (1. + RAND_MAX) - 0.5f;
    vec[i].z = rand() / (1. + RAND_MAX) - 0.5f;
  }
  // cerr << vec[0].x << " "
  //    << vec[0].y << " "
  //    << vec[0].z << endl;
  Octree <4> octree(&vec[0], n);
  // cerr << vec[0].x << " "
  //    << vec[0].y << " "
  //    << vec[0].z << endl;
#endif
  int nbody = 1<<20;
  vector<float4> vec(nbody);
  Plummer plummer(nbody);
  for(int i=0; i<nbody; i++){
    vec[i].w = plummer.mass[i];
    vec[i].x = plummer.pos[i].x;
    vec[i].y = plummer.pos[i].y;
    vec[i].z = plummer.pos[i].z;
  }
  Octree <10> octree(&vec[0], nbody);
  return 0;
}
