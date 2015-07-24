#pragma once
#include <complex>
#include "kahan.h"
#include "macros.h"
#include <stdint.h>
#include "vec.h"
#include <vector>

//! Basic type definitions
#if FP64
typedef double               real_t;                            //!< Floating point type is double precision
const real_t EPS = 1e-16;                                       //!< Double precision epsilon
#else
typedef float                real_t;                            //!< Floating point type is single precision
const real_t EPS = 1e-8;                                        //!< Single precision epsilon
#endif
typedef std::complex<real_t> complex_t;                         //!< Complex type
typedef vec<3,real_t>        vec3;                              //!< Vector of 3 real_t types

//! SIMD vector types for MIC, AVX, and SSE
const int NSIMD = SIMD_BYTES / sizeof(real_t);                  //!< SIMD vector length (SIMD_BYTES defined in macros.h)
typedef vec<NSIMD,real_t> simdvec;                              //!< SIMD vector type

//! Kahan summation types (Achieves quasi-double precision using single precision types)
#if KAHAN
typedef kahan<real_t>  kreal_t;                                 //!< Floating point type with Kahan summation
typedef vec<3,kreal_t> kvec3;                                   //!< Vector of 3 floating point types with Kahan summaiton
typedef kahan<simdvec> ksimdvec;                                //!< SIMD vector type with Kahan summation
#else
typedef real_t         kreal_t;                                 //!< Floating point type
typedef vec<3,real_t>  kvec3;                                   //!< Vector of 3 floating point types
typedef simdvec        ksimdvec;                                //!< SIMD vector type
#endif

//! Multipole/local expansion coefficients
const int P = EXPANSION;                                        //!< Order of expansions
#if Cartesian
const int NTERM = P*(P+1)*(P+2)/6;                              //!< Number of Cartesian mutlipole/local terms
typedef vec<NTERM,real_t> vecP;                                 //!< Multipole/local coefficient type for Cartesian
#elif Spherical
const int NTERM = P*(P+1)/2;                                    //!< Number of Spherical multipole/local terms
typedef vec<NTERM,complex_t> vecP;                              //!< Multipole/local coefficient type for spherical
#endif
typedef std::vector<vecP> Coefs;                                //!< Vector of expansion coefficients

//! Min & max bounds, center, and radius of bounding box
struct Bounds {
  vec3 Xmin;                                                    //!< Minimum value of coordinates
  vec3 Xmax;                                                    //!< Maximum value of coordinates
  vec3 X;                                                       //!< Box center
  real_t R;                                                     //!< Box radius
};

//! Structure of bodies
struct Body {
  vec3 X;                                                       //!< Coordinates
  real_t q;                                                     //!< Charge
};
typedef std::vector<Body> Bodies;                               //!< Vector of bodies

//! Structure of fields
struct Field {
  kreal_t p;                                                    //!< Potential
  kvec3 F;                                                      //!< Force
};
typedef std::vector<Field> Fields;                              //!< Vector of fields

typedef std::vector<int> Ints;                                  //!< Vector of integers
typedef std::vector<uint64_t> Uint64s;                          //!< Vector of 64-bit unsigned integers
