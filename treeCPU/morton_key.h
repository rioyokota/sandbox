// 63-bit key generator
#include <cassert>
#ifndef __FLOAT4
#define __FLOAT4
struct float4{
  float x, y, z, w;
} __attribute__ ((aligned(16)));
#endif

template <typename REAL, int SCALE>
struct Morton_key{
  typedef unsigned long long key_t;
  key_t val;
  Morton_key() {}
  Morton_key(float4 p)
  {
    val = key_gen(p.x, p.y, p.z);
  }
  static key_t key_gen(REAL x, REAL y, REAL z)
  {
    static const key_t table[128] = {
      #include "key_table"
    };
    const REAL scale = REAL(1<<(20-SCALE));
    int xi = int(x * scale);
    int yi = int(y * scale);
    int zi = int(z * scale);
    assert(((-1<<21) < xi) && (xi < (1<<21)));
    assert(((-1<<21) < yi) && (yi < (1<<21)));
    assert(((-1<<21) < zi) && (zi < (1<<21)));
    key_t xkey = (table[xi&127]) | (table[(xi>>7)&127] << 21) | (table[(xi>>14)&127] << 42);
    key_t ykey = (table[yi&127]) | (table[(yi>>7)&127] << 21) | (table[(yi>>14)&127] << 42);
    key_t zkey = (table[zi&127]) | (table[(zi>>7)&127] << 21) | (table[(zi>>14)&127] << 42);
    return (xkey<<2) | (ykey<<1) | zkey;
  }
  key_t key(){
    return val;
  }
  operator key_t(){
    return val;
  }
  bool operator< (const Morton_key &rhs) const{
    return val < rhs.val;
  }
};

struct Key_index{
  Morton_key <float, 8> key;
  int index;
  int pad;
  Key_index() {}
  Key_index(float4 p, int i) :
    key(p),
    index(i)
  {}
#if 1
  // SSE optimization for STL sort
  const Key_index operator = (const Key_index &rhs) const{
    typedef float v4sf __attribute__ ((vector_size(16)));
    *(v4sf *)this = *(v4sf *)&rhs;
    return *this;
  }
#endif
  bool operator == (const Key_index &rhs) const {
    return (index == rhs.index) && (key.val == rhs.key.val);
  }
  friend std::ostream &operator << (std::ostream &os, const Key_index ki){
    os << ki.index << " " << ki.key.val;
    return os;
  }
};

struct Cmp_key_index{
  bool operator() (const Key_index &lhs, const Key_index &rhs){
    return lhs.key < rhs.key;
  }
};

#if 1
  // SSE optimization for STL sort
namespace std{
  template <>
  inline void iter_swap <std::vector<Key_index>::iterator, std::vector<Key_index>::iterator>
    (std::vector<Key_index>::iterator a, std::vector<Key_index>::iterator b){

    typedef float v4sf __attribute__ ((vector_size(16)));
    v4sf *ap = (v4sf *)&(*a);
    v4sf *bp = (v4sf *)&(*b);
    v4sf v0 = *ap;
    v4sf v1 = *bp;
    *ap = v1;
    *bp = v0;
  }
}
#endif
