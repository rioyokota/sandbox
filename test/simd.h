#ifndef simd_h
#define simd_h
#include <iostream>
#include <immintrin.h>
class fvec8 {
private:
  __m256 ymm;
public:
  fvec8() {}                                                       // Default constructor
  fvec8(const float a) {                                           // Copy constructor scalar
    ymm = _mm256_set1_ps(a);
  }
  fvec8(const __m256 a) {                                          // Copy constructor vector (__m256)
    ymm = a;
  }
  fvec8(const fvec8 &a) {                                          // Copy constructor vector (fvec8)
    ymm = a.ymm;
  }
  fvec8(const float a, const float b, const float c, const float d,
        const float e, const float f, const float g, const float h) {// Copy constructor (component-wise)
    ymm = _mm256_setr_ps(a,b,c,d,e,f,g,h);
  }
  ~fvec8(){}                                                       // Destructor
  const fvec8 &operator=(const float a) {                          // Scalar assignment
    ymm = _mm256_set1_ps(a);
    return *this;
  }
  const fvec8 &operator=(const __m256 a) {                         // Vector assignment (__m256)
    ymm = a;
    return *this;
  }
  const fvec8 &operator=(const fvec8 &a) {                         // Vector assignment (fvec8)
    ymm = a.ymm;
    return *this;
  }
  const fvec8 &operator+=(const fvec8 &a) {                        // Vector compound assignment (add)
    ymm = _mm256_add_ps(ymm,a.ymm);
    return *this;
  }
  const fvec8 &operator-=(const fvec8 &a) {                        // Vector compound assignment (subtract)
    ymm = _mm256_sub_ps(ymm,a.ymm);
    return *this;
  }
  const fvec8 &operator*=(const fvec8 &a) {                        // Vector compound assignment (multiply)
    ymm = _mm256_mul_ps(ymm,a.ymm);
    return *this;
  }
  const fvec8 &operator/=(const fvec8 &a) {                        // Vector compound assignment (divide)
    ymm = _mm256_div_ps(ymm,a.ymm);
    return *this;
  }
  fvec8 operator+(const fvec8 &a) const {                          // Vector arithmetic (add)
    return fvec8(_mm256_add_ps(ymm,a.ymm));
  }
  fvec8 operator-(const fvec8 &a) const {                          // Vector arithmetic (subtract)
    return fvec8(_mm256_sub_ps(ymm,a.ymm));
  }
  fvec8 operator*(const fvec8 &a) const {                          // Vector arithmetic (multiply)
    return fvec8(_mm256_mul_ps(ymm,a.ymm));
  }
  fvec8 operator/(const fvec8 &a) const {                          // Vector arithmetic (divide)
    return fvec8(_mm256_div_ps(ymm,a.ymm));
  }
  float &operator[](int i) {                                       // Indexing (lvalue)
    return ((float*)&ymm)[i];
  }
  const float &operator[](int i) const {                           // Indexing (rvalue)
    return ((float*)&ymm)[i];
  }
  friend std::ostream &operator<<(std::ostream &s, const fvec8 &a) {// Component-wise output stream
    for (int i=0; i<8; i++) s<<((float*)&a)[i]<<' ';
    return s;
  }
  friend fvec8 rsqrt(const fvec8 &a) {
    return fvec8(_mm256_rsqrt_ps(a.ymm));
  }
};

#endif
