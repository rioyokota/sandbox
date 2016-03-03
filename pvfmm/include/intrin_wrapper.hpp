#ifndef _PVFMM_INTRIN_WRAPPER_HPP_
#define _PVFMM_INTRIN_WRAPPER_HPP_

namespace pvfmm{

template <class T>
inline T zero_intrin(){
  return (T)0;
}

template <class T, class Real_t>
inline T set_intrin(const Real_t& a){
  return a;
}

template <class T, class Real_t>
inline T load_intrin(Real_t const* a){
  return a[0];
}

template <class T, class Real_t>
inline T bcast_intrin(Real_t const* a){
  return a[0];
}

template <class T, class Real_t>
inline void store_intrin(Real_t* a, const T& b){
  a[0]=b;
}

template <class T>
inline T mul_intrin(const T& a, const T& b){
  return a*b;
}

template <class T>
inline T add_intrin(const T& a, const T& b){
  return a+b;
}

template <class T>
inline T sub_intrin(const T& a, const T& b){
  return a-b;
}

template <class T>
inline T rsqrt_approx_intrin(const T& r2){
  if(r2!=0) return 1.0/sqrtf(r2);
  return 0;
}

template <class T, class Real_t>
inline void rsqrt_newton_intrin(T& rinv, const T& r2, const Real_t& nwtn_const){
  rinv=rinv*(nwtn_const-r2*rinv*rinv);
}

#ifdef __SSE3__
template <>
inline __m128 zero_intrin(){
  return _mm_setzero_ps();
}

template <>
inline __m128d zero_intrin(){
  return _mm_setzero_pd();
}

template <>
inline __m128 set_intrin(const float& a){
  return _mm_set_ps1(a);
}

template <>
inline __m128d set_intrin(const double& a){
  return _mm_set1_pd(a);
}

template <>
inline __m128 load_intrin(float const* a){
  return _mm_load_ps(a);
}

template <>
inline __m128d load_intrin(double const* a){
  return _mm_load_pd(a);
}

template <>
inline __m128 bcast_intrin(float const* a){
  return _mm_set_ps1(a[0]);
}

template <>
inline __m128d bcast_intrin(double const* a){
  return _mm_load_pd1(a);
}

template <>
inline void store_intrin(float* a, const __m128& b){
  return _mm_store_ps(a,b);
}

template <>
inline void store_intrin(double* a, const __m128d& b){
  return _mm_store_pd(a,b);
}

template <>
inline __m128 mul_intrin(const __m128& a, const __m128& b){
  return _mm_mul_ps(a,b);
}

template <>
inline __m128d mul_intrin(const __m128d& a, const __m128d& b){
  return _mm_mul_pd(a,b);
}

template <>
inline __m128 add_intrin(const __m128& a, const __m128& b){
  return _mm_add_ps(a,b);
}

template <>
inline __m128d add_intrin(const __m128d& a, const __m128d& b){
  return _mm_add_pd(a,b);
}

template <>
inline __m128 sub_intrin(const __m128& a, const __m128& b){
  return _mm_sub_ps(a,b);
}

template <>
inline __m128d sub_intrin(const __m128d& a, const __m128d& b){
  return _mm_sub_pd(a,b);
}

template <>
inline __m128 rsqrt_approx_intrin(const __m128& r2){
  #define VEC_INTRIN          __m128
  #define RSQRT_INTRIN(a)     _mm_rsqrt_ps(a)
  #define CMPEQ_INTRIN(a,b)   _mm_cmpeq_ps(a,b)
  #define ANDNOT_INTRIN(a,b)  _mm_andnot_ps(a,b)
  return ANDNOT_INTRIN(CMPEQ_INTRIN(r2,zero_intrin<VEC_INTRIN>()),RSQRT_INTRIN(r2));
  #undef VEC_INTRIN
  #undef RSQRT_INTRIN
  #undef CMPEQ_INTRIN
  #undef ANDNOT_INTRIN
}

template <>
inline __m128d rsqrt_approx_intrin(const __m128d& r2){
  #define PD2PS(a) _mm_cvtpd_ps(a)
  #define PS2PD(a) _mm_cvtps_pd(a)
  return PS2PD(rsqrt_approx_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline void rsqrt_newton_intrin(__m128& rinv, const __m128& r2, const float& nwtn_const){
  #define VEC_INTRIN       __m128
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline void rsqrt_newton_intrin(__m128d& rinv, const __m128d& r2, const double& nwtn_const){
  #define VEC_INTRIN       __m128d
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

#endif



#ifdef __AVX__
template <>
inline __m256 zero_intrin(){
  return _mm256_setzero_ps();
}

template <>
inline __m256d zero_intrin(){
  return _mm256_setzero_pd();
}

template <>
inline __m256 set_intrin(const float& a){
  return _mm256_set_ps(a,a,a,a,a,a,a,a);
}

template <>
inline __m256d set_intrin(const double& a){
  return _mm256_set_pd(a,a,a,a);
}

template <>
inline __m256 load_intrin(float const* a){
  return _mm256_load_ps(a);
}

template <>
inline __m256d load_intrin(double const* a){
  return _mm256_load_pd(a);
}

template <>
inline __m256 bcast_intrin(float const* a){
  return _mm256_broadcast_ss(a);
}

template <>
inline __m256d bcast_intrin(double const* a){
  return _mm256_broadcast_sd(a);
}

template <>
inline void store_intrin(float* a, const __m256& b){
  return _mm256_store_ps(a,b);
}

template <>
inline void store_intrin(double* a, const __m256d& b){
  return _mm256_store_pd(a,b);
}

template <>
inline __m256 mul_intrin(const __m256& a, const __m256& b){
  return _mm256_mul_ps(a,b);
}

template <>
inline __m256d mul_intrin(const __m256d& a, const __m256d& b){
  return _mm256_mul_pd(a,b);
}

template <>
inline __m256 add_intrin(const __m256& a, const __m256& b){
  return _mm256_add_ps(a,b);
}

template <>
inline __m256d add_intrin(const __m256d& a, const __m256d& b){
  return _mm256_add_pd(a,b);
}

template <>
inline __m256 sub_intrin(const __m256& a, const __m256& b){
  return _mm256_sub_ps(a,b);
}

template <>
inline __m256d sub_intrin(const __m256d& a, const __m256d& b){
  return _mm256_sub_pd(a,b);
}

template <>
inline __m256 rsqrt_approx_intrin(const __m256& r2){
  #define VEC_INTRIN          __m256
  #define RSQRT_INTRIN(a)     _mm256_rsqrt_ps(a)
  #define CMPEQ_INTRIN(a,b)   _mm256_cmp_ps(a,b,_CMP_EQ_OS)
  #define ANDNOT_INTRIN(a,b)  _mm256_andnot_ps(a,b)
  return ANDNOT_INTRIN(CMPEQ_INTRIN(r2,zero_intrin<VEC_INTRIN>()),RSQRT_INTRIN(r2));
  #undef VEC_INTRIN
  #undef RSQRT_INTRIN
  #undef CMPEQ_INTRIN
  #undef ANDNOT_INTRIN
}

template <>
inline __m256d rsqrt_approx_intrin(const __m256d& r2){
  #define PD2PS(a) _mm256_cvtpd_ps(a)
  #define PS2PD(a) _mm256_cvtps_pd(a)
  return PS2PD(rsqrt_approx_intrin(PD2PS(r2)));
  #undef PD2PS
  #undef PS2PD
}

template <>
inline void rsqrt_newton_intrin(__m256& rinv, const __m256& r2, const float& nwtn_const){
  #define VEC_INTRIN       __m256
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

template <>
inline void rsqrt_newton_intrin(__m256d& rinv, const __m256d& r2, const double& nwtn_const){
  #define VEC_INTRIN       __m256d
  rinv=mul_intrin(rinv,sub_intrin(set_intrin<VEC_INTRIN>(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
  #undef VEC_INTRIN
}

#endif


template <class VEC, class Real_t>
inline VEC rsqrt_intrin2(VEC r2){
  VEC rinv=rsqrt_approx_intrin(r2);
  rsqrt_newton_intrin(rinv,r2,Real_t(3));
  rsqrt_newton_intrin(rinv,r2,Real_t(12));
  return rinv;
}

}

#endif //_PVFMM_INTRIN_WRAPPER_HPP_
