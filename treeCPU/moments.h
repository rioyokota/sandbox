#include <iostream>
#include <cmath>
#include <vector>

#if 0 
struct float4{
  float x, y, z, w;
};
#include "node.h"
#endif

struct Com{
  float x, y, z, m;
  Com() {
    x = y = z = m = 0.0f;
  }
  // construct from a leaf
  Com(const float4 ptcl[], int np){
    x = y = z = m = 0.0f;
    for(int i=0; i<np; i++){
      const float4 &p = ptcl[i];
      x += p.x * p.w;
      y += p.y * p.w;
      z += p.z * p.w;
      m += p.w;
    }
    x *= (1./m);
    y *= (1./m);
    z *= (1./m);
  }
  // construct from a node
  Com(const Com cnode[]){
    x = y = z = m = 0.0f;
    for(int i=0; i<8; i++){
      const Com &c = cnode[i];
      x += c.x * c.m;
      y += c.y * c.m;
      z += c.z * c.m;
      m += c.m;
    }
    x *= (1./m);
    y *= (1./m);
    z *= (1./m);
  }
};

struct Quad{
  float xx, yy, zz, xy, yz, zx;
  Quad() {
    xx = yy = zz = xy = yz = zx = 0.0;
  }
  Quad operator += (const Quad &r){
    xx += r.xx;
    yy += r.yy;
    zz += r.zz;
    xy += r.xy;
    yz += r.yz;
    zx += r.zx;
    return *this;
  }
  Quad(const Com &com, float x, float y, float z, float m){
    x -= com.x;
    y -= com.y;
    z -= com.z;
    float dr2 = x*x + y*y + z*z;
    xx = m * (3.f*x*x - dr2);
    yy = m * (3.f*y*y - dr2);
    zz = m * (3.f*z*z - dr2);
    xy = (3.f*m)*x*y;
    yz = (3.f*m)*y*z;
    zx = (3.f*m)*z*x;
  }
  // construct from a leaf
  Quad(const Com &com, const float4 ptcl[], int np){
    xx = yy = zz = xy = yz = zx = 0.0;
    for(int i=0; i<np; i++){
      const float4 &p = ptcl[i];
      *this += Quad(com, p.x, p.y, p.z, p.w);
    }
  }
  // construct from a node
  Quad(const Com &com, const Com cnode[], const Quad qnode[]){
    xx = yy = zz = xy = yz = zx = 0.0;
    for(int i=0; i<8; i++){
      const Com &c = cnode[i];
      *this += Quad(com, c.x, c.y, c.z, c.m);
      *this += qnode[i];
    }
  }
};

struct Bound{
  float xmin, ymin, zmin;
  float xmax, ymax, zmax;
  Bound(){
    xmin = ymin = zmin = +HUGE;
    xmax = ymax = zmax = -HUGE;
  }
  // construct from a leaf
  Bound(const float4 ptcl[], int np){
    xmin = ymin = zmin = +HUGE;
    xmax = ymax = zmax = -HUGE;
    for(int i=0; i<np; i++){
      xmin = std::min(xmin, ptcl[i].x);
      xmax = std::max(xmax, ptcl[i].x);
      ymin = std::min(ymin, ptcl[i].y);
      ymax = std::max(ymax, ptcl[i].y);
      zmin = std::min(zmin, ptcl[i].z);
      zmax = std::max(zmax, ptcl[i].z);
    }
  }
  // construct from a node
  Bound(const Bound node[]){
    xmin = ymin = zmin = +HUGE;
    xmax = ymax = zmax = -HUGE;
    for(int i=0; i<8; i++){
      xmin = std::min(xmin, node[i].xmin);
      xmax = std::max(xmax, node[i].xmax);
      ymin = std::min(ymin, node[i].ymin);
      ymax = std::max(ymax, node[i].ymax);
      zmin = std::min(zmin, node[i].zmin);
      zmax = std::max(zmax, node[i].zmax);
    }
  }
  float4 centre() const {
    float4 ret;
    ret.x = 0.5 * (xmin + xmax);
    ret.y = 0.5 * (ymin + ymax);
    ret.z = 0.5 * (zmin + zmax);
    ret.w = std::max(xmax-xmin, std::max(ymax-ymin, zmax-zmin));
    return ret;
  }
};

struct Moments{
  std::vector<Com>   com;
  std::vector<Quad>  quad;
  std::vector<Bound> bound;
  std::vector<float4>  centre;

  Moments(const Node node[], int nnode, const float4 ptcl[]) :
    com(nnode),
    quad(nnode),
    bound(nnode),
    centre(nnode)
  {
    calc_momoents(node, 0, ptcl);
  }
  int np(const Node node[], int inode){
    const Node &n = node[inode];
    if(n.is_leaf()){
         return n.np;
    }else{
      int cfirst = n.ifirst;
      int nn = 0;
      for(int ic=0; ic<8; ic++){
        nn += np(node, cfirst+ic);
      }
      return nn;
    }
  }
  double mass(const Node node[], int inode, const float4 ptcl[]){
    const Node &n = node[inode];
    double m = 0.0;
    if(n.is_leaf()){
      int pfirst = n.ifirst;
      for(int i=0; i<n.np; i++){
        m += ptcl[pfirst + i].w;
      }
    }else{
      int cfirst = n.ifirst;
      for(int ic=0; ic<8; ic++){
        m += mass(node, cfirst+ic, ptcl);
      }
    }
    return m;
  }
  void calc_momoents(const Node node[], int inode, const float4 ptcl[]){
    int npd     = node[inode].np;
    if(node[inode].is_leaf()){
      int pfirst = node[inode].ifirst;
      if(npd > 0){
        com  [inode] = Com  (            &ptcl[pfirst], npd);
        quad [inode] = Quad (com[inode], &ptcl[pfirst], npd);
        bound[inode] = Bound(            &ptcl[pfirst], npd);
      }
    }else{
      int cfirst = node[inode].ifirst;
      if(npd < (1<<16)){
        for(int ic=0; ic<8; ic++){
          calc_momoents(node, cfirst+ic, ptcl);
        }
      }else{
#pragma omp parallel for
        for(int ic=0; ic<8; ic++){
          calc_momoents(node, cfirst+ic, ptcl);
        }
      }
      com  [inode] = Com  (&com[cfirst]);
      quad [inode] = Quad (com[inode], &com[cfirst], &quad[cfirst]);
      bound[inode] = Bound(&bound[cfirst]);
    }
    centre[inode] = bound[inode].centre();
  }
};
