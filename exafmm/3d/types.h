#ifndef types_h
#define types_h
#include <assert.h>                                             // Some compilers don't have cassert
#include <complex>
#include <stdint.h>
#include <vector>
#include "vec.h"

namespace exafmm {
  // Basic type definitions
#if EXAFMM_SINGLE
  typedef float real_t;                                         //!< Floating point type is single precision
  const real_t EPS = 1e-8f;                                     //!< Single precision epsilon
#else
  typedef double real_t;                                        //!< Floating point type is double precision
  const real_t EPS = 1e-16;                                     //!< Double precision epsilon
#endif
  typedef std::complex<real_t> complex_t;                       //!< Complex type
  const complex_t I(0.,1.);                                     //!< Imaginary unit

  typedef vec<3,int> ivec3;                                     //!< Vector of 3 int types
  typedef vec<3,real_t> vec3;                                   //!< Vector of 3 real_t types
  typedef vec<4,real_t> vec4;                                   //!< Vector of 4 real_t types
  typedef vec<3,float> fvec3;                                   //!< Vector of 3 float types
  typedef vec<3,complex_t> cvec3;                               //!< Vector of 3 complex_t types

  //! Center and radius of bounding box
  struct Box {
    vec3   X;                                                   //!< Box center
    real_t R;                                                   //!< Box radius
  };

  //! Min & max bounds of bounding box
  struct Bounds {
    vec3 Xmin;                                                  //!< Minimum value of coordinates
    vec3 Xmax;                                                  //!< Maximum value of coordinates
  };

  //! Structure of aligned source for SIMD
  struct Source {                                               //!< Base components of source structure
    vec3      X;                                                //!< Position
    real_t    SRC;                                              //!< Scalar real values
  };

  //! Structure of bodies
  struct Body : public Source {                                 //!< Base components of body structure
    int     IBODY;                                              //!< Initial body numbering for sorting back
    int     IRANK;                                              //!< Initial rank numbering for partitioning back
    int64_t ICELL;                                              //!< Cell index
    real_t  WEIGHT;                                             //!< Weight for partitioning
    vec4    TRG;                                                //!< Scalar+vector3 real values
  };
  typedef std::vector<Body> Bodies;                             //!< Vector of bodies
  typedef typename Bodies::iterator B_iter;                     //!< Iterator of body vector

  //! Base components of cells
  struct CellBase {
    int IPARENT;                                                //!< Index of parent cell
    int ICHILD;                                                 //!< Index of first child cell
    int NCHILD;                                                 //!< Number of child cells
    int IBODY;                                                  //!< Index of first body
    int NBODY;                                                  //!< Number of descendant bodies
    uint64_t ICELL;                                             //!< Cell index
    // real_t   WEIGHT;                                            //!< Weight for partitioning
    vec3     X;                                                 //!< Cell center
    real_t   R;                                                 //!< Cell radius
    B_iter   BODY;                                              //!< Iterator of first body
  };
  //! Structure of cells
  struct Cell : public CellBase {
    std::vector<complex_t> M;                                   //!< Multipole expansion coefs
    std::vector<complex_t> L;                                   //!< Local expansion coefs
    using CellBase::operator=;
  };
  typedef std::vector<Cell> Cells;                              //!< Vector of cells
  typedef std::vector<CellBase> CellBases;                      //!< Vector of cell bases
  typedef typename Cells::iterator C_iter;                      //!< Iterator of cell vector
  typedef typename CellBases::iterator CB_iter;                 //!< Iterator of cell vector
}
#endif
