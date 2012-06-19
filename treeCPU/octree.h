#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <omp.h>

#ifndef __FLOAT4
#define __FLOAT4
struct float4{
  float x, y, z, w;
#if 1
  // SSE optimization for STL sort
  const float4 operator = (const float4 &rhs) const{
    typedef float v4sf __attribute__ ((vector_size(16)));
    *(v4sf *)this = *(v4sf *)&rhs;
    return *this;
  }
#endif
} __attribute__ ((aligned(16)));
#endif

#include "wtime.h"
#include "morton_key.h"
#include "node.h"
#include "octsort.h"
#include "moments.h"



typedef float v4sf __attribute__ ((vector_size(16)));
template <int NP_PER_NODE = 2>
struct Octree{
  int n;
  std::vector<Key_index> key_index;
  std::vector<Node> node;
  std::vector<unsigned long long> key;
  std::vector<int> index;
  std::vector<float4> posm;
  // std::vector<float4> com;
  // std::vector<Quad> quad;
  Octree(/*const*/ float4 x[], int _n) :
    n(_n),
    key_index(n),
    node(n / NP_PER_NODE),
    key(n),
    index(n),
    posm(n)
  {
    double t0 = wtime();
// #pragma omp parallel for
    for(int i=0; i<n; i++){
      key_index[i] = Key_index(x[i], i);
    }
    double t1 = wtime();
    std::cout << "key gen: " << t1-t0 << std::endl;
#if 0
    std::sort(key_index.begin(), key_index.end(), Cmp_key_index());
#else
    octsort64 <57> ((int)key_index.size(), 
                &key_index[0], 
                &std::vector<Key_index>(n)[0]);
#endif
    double t2 = wtime();
    std::cout << "sort   : " << t2-t1 << std::endl;

#pragma omp parallel for
    for(int i=0; i<n; i++){
      key[i] = key_index[i].key;
      index[i] = key_index[i].index;
    }
    linear2morton((v4sf *)&posm[0], (v4sf *)x);
    // morton2linear(x, &posm[0]);
    double t3 = wtime();
    std::cout << "shuffle: " << t3-t2 << std::endl;

    Node::node_ptr() = &node[0];
    Node::node_count() = 1;
    Node::node_limit() = node.size();
    for(int i=0; i<n; i++){
      node[0].insert(i, &key[0], 60);
    }
    double t4 = wtime();
    std::cout << "build  : " << t4-t3 << std::endl;

    Moments moments(&node[0], Node::node_count(), &posm[0]);
    double t5 = wtime();
    std::cout << "upward : " << t5-t4 << std::endl;
    std::cout << "total  : " << t5-t0 << std::endl;
    std::cout << "node / ptcl " 
              << Node::node_count() << " / "
              << n << std::endl;
#if 0
    Com com;
    moments.quad[0] = Quad();
    for(int i=0; i<n; i++){
      const float4 &p = posm[i];
      moments.quad[0] += Quad(com, p.x, p.y, p.z, p.w);
    }
#endif
  }
  // gather
  template <typename T>
  void linear2morton(T dst[], const T src[]){
#pragma omp parallel for
    for(int i=0; i<n; i++){
      dst[i] = src[index[i]];
    }
  }
  // scatter
  template <typename T>
  void morton2linear(T dst[], const T src[]){
#pragma omp parallel for
    for(int i=0; i<n; i++){
      dst[index[i]] = src[i];
    }
  }
};
