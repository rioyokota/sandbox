#ifndef simd_h
#define simd_h
#include <iostream>
#include <immintrin.h>

class fvec8 {
private:
  __m256 data;
public:
  fvec8() {}                                                       // Default constructor
  fvec8(const float v) {                                           // Copy constructor scalar
    data = _mm256_set1_ps(v);
  }
  fvec8(const __m256 v) {                                          // Copy constructor SIMD register
    data = v;
  }
  fvec8(const fvec8 &v) {                                          // Copy constructor vector
    data = v.data;
  }
  fvec8(const float a, const float b, const float c, const float d,
        const float e, const float f, const float g, const float h) {// Copy constructor (component-wise)
    data = _mm256_setr_ps(a,b,c,d,e,f,g,h);
  }
  ~fvec8(){}                                                       // Destructor
  const fvec8 &operator=(const float v) {                          // Scalar assignment
    data = _mm256_set1_ps(v);
    return *this;
  }
  const fvec8 &operator=(const fvec8 &v) {                         // Vector assignment
    data = v.data;
    return *this;
  }
  const fvec8 &operator+=(const fvec8 &v) {                        // Vector compound assignment (add)
    data = _mm256_add_ps(data,v.data);
    return *this;
  }
  const fvec8 &operator-=(const fvec8 &v) {                        // Vector compound assignment (subtract)
    data = _mm256_sub_ps(data,v.data);
    return *this;
  }
  const fvec8 &operator*=(const fvec8 &v) {                        // Vector compound assignment (multiply)
    data = _mm256_mul_ps(data,v.data);
    return *this;
  }
  const fvec8 &operator/=(const fvec8 &v) {                        // Vector compound assignment (divide)
    data = _mm256_div_ps(data,v.data);
    return *this;
  }
  fvec8 operator+(const fvec8 &v) const {                          // Vector arithmetic (add)
    return fvec8(_mm256_add_ps(data,v.data));
  }
  fvec8 operator-(const fvec8 &v) const {                          // Vector arithmetic (subtract)
    return fvec8(_mm256_sub_ps(data,v.data));
  }
  fvec8 operator*(const fvec8 &v) const {                          // Vector arithmetic (multiply)
    return fvec8(_mm256_mul_ps(data,v.data));
  }
  fvec8 operator/(const fvec8 &v) const {                          // Vector arithmetic (divide)
    return fvec8(_mm256_div_ps(data,v.data));
  }
  float &operator[](int i) {                                       // Indexing (lvalue)
    return ((float*)&data)[i];
  }
  const float &operator[](int i) const {                           // Indexing (rvalue)
    return ((float*)&data)[i];
  }
  friend std::ostream &operator<<(std::ostream &s, const fvec8 &v) {// Component-wise output stream
    for (int i=0; i<8; i++) s << v[i] << ' ';
    return s;
  }
  friend fvec8 rsqrt(const fvec8 &v) {                             // reciprocal square root
    return fvec8(_mm256_rsqrt_ps(v.data));
  }
};

#endif
