#ifndef types_h
#define types_h
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "vec.h"

namespace exafmm {
  // Basic type definitions
  typedef double real_t;                                        //!< Floating point type
  typedef std::complex<real_t> complex_t;                       //!< Complex type
  const complex_t I(0.,1.);                                     //!< Imaginary unit

  typedef vec<3,real_t> vec3;                                   //!< Vector of 3 real_t types
  typedef vec<4,real_t> vec4;                                   //!< Vector of 4 real_t types

  //! Center and radius of bounding box
  struct Box {
    real_t X[3];                                                //!< Box center
    real_t R;                                                   //!< Box radius
  };

  //! Structure of bodies
  struct Body {                                                 //!< Base components of body structure
    vec3  X;                                                    //!< Position
    real_t SRC;                                                 //!< Scalar real values
    vec4 TRG;                                                   //!< Scalar+vector3 real values
  };
  typedef std::vector<Body> Bodies;                             //!< Vector of bodies

  //! Structure of cells
  struct Cell {
    int NCHILD;                                                 //!< Number of child cells
    int NBODY;                                                  //!< Number of descendant bodies
    Cell * CHILD;                                               //!< Pointer of first child cell
    std::vector<Cell>::iterator CHILD2;                         //!< Pointer of first child cell
    Body * BODY;                                                //!< Pointer of first body
    vec3 X;                                                     //!< Cell center
    real_t R;                                                   //!< Cell radius
    std::vector<complex_t> M;                                   //!< Multipole expansion coefs
    std::vector<complex_t> L;                                   //!< Local expansion coefs
  };
  typedef std::vector<Cell> Cells;                              //!< Vector of cells
  typedef typename Cells::iterator C_iter;                      //!< Iterator of cell vector
}
#endif
