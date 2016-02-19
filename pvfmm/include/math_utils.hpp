#include <cmath>
#include <ostream>

#ifndef _MATH_UTILS_
#define _MATH_UTILS_

namespace pvfmm{

template <class Real_t>
inline Real_t fabs(const Real_t f){return ::fabs(f);}

template <class Real_t>
inline Real_t sqrt(const Real_t a){return ::sqrt(a);}

template <class Real_t>
inline Real_t sin(const Real_t a){return ::sin(a);}

template <class Real_t>
inline Real_t cos(const Real_t a){return ::cos(a);}

template <class Real_t>
inline Real_t exp(const Real_t a){return ::exp(a);}

template <class Real_t>
inline Real_t log(const Real_t a){return ::log(a);}

template <class Real_t>
inline Real_t pow(const Real_t b, const Real_t e){return ::pow(b,e);}

}//end namespace



#ifdef PVFMM_QUAD_T

typedef PVFMM_QUAD_T QuadReal_t;

namespace pvfmm{
inline QuadReal_t atoquad(const char* str);
}

inline std::ostream& operator<<(std::ostream& output, const QuadReal_t q_);

#endif //PVFMM_QUAD_T

#endif //_MATH_UTILS_HPP_

