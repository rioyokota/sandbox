#ifndef _PVFMM_INTRIN_WRAPPER_HPP_
#define _PVFMM_INTRIN_WRAPPER_HPP_

namespace pvfmm{

#ifdef __AVX__
inline __m256 zero_intrin(const float){
  return _mm256_setzero_ps();
}

inline __m256d zero_intrin(const double){
  return _mm256_setzero_pd();
}

inline __m256 set_intrin(const float& a){
  return _mm256_set_ps(a,a,a,a,a,a,a,a);
}

inline __m256d set_intrin(const double& a){
  return _mm256_set_pd(a,a,a,a);
}

inline __m256 load_intrin(const float* a){
  return _mm256_load_ps(a);
}

inline __m256d load_intrin(const double* a){
  return _mm256_load_pd(a);
}

inline void store_intrin(float* a, const __m256& b){
  return _mm256_store_ps(a,b);
}

inline void store_intrin(double* a, const __m256d& b){
  return _mm256_store_pd(a,b);
}

inline __m256 mul_intrin(const __m256& a, const __m256& b){
  return _mm256_mul_ps(a,b);
}

inline __m256d mul_intrin(const __m256d& a, const __m256d& b){
  return _mm256_mul_pd(a,b);
}

inline __m256 add_intrin(const __m256& a, const __m256& b){
  return _mm256_add_ps(a,b);
}

inline __m256d add_intrin(const __m256d& a, const __m256d& b){
  return _mm256_add_pd(a,b);
}

inline __m256 sub_intrin(const __m256& a, const __m256& b){
  return _mm256_sub_ps(a,b);
}

inline __m256d sub_intrin(const __m256d& a, const __m256d& b){
  return _mm256_sub_pd(a,b);
}

inline __m128 rsqrt_approx_intrin(const __m128& r2){
  return _mm_andnot_ps(_mm_cmpeq_ps(r2,_mm_setzero_ps()),_mm_rsqrt_ps(r2));
}

inline __m256 rsqrt_approx_intrin(const __m256& r2){
  return _mm256_andnot_ps(_mm256_cmp_ps(r2,_mm256_setzero_ps(),_CMP_EQ_OS),_mm256_rsqrt_ps(r2));
}

inline __m256d rsqrt_approx_intrin(const __m256d& r2){
  return _mm256_cvtps_pd(rsqrt_approx_intrin(_mm256_cvtpd_ps(r2)));
}

inline void rsqrt_newton_intrin(__m256& rinv, const __m256& r2, const float& nwtn_const){
  rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
}

inline void rsqrt_newton_intrin(__m256d& rinv, const __m256d& r2, const double& nwtn_const){
  rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
}

#else
#ifdef __SSE3__
inline __m128 zero_intrin(const float){
  return _mm_setzero_ps();
}

inline __m128d zero_intrin(const double){
  return _mm_setzero_pd();
}

inline __m128 set_intrin(const float& a){
  return _mm_set1_ps(a);
}

inline __m128d set_intrin(const double& a){
  return _mm_set1_pd(a);
}

inline __m128 load_intrin(const float* a){
  return _mm_load_ps(a);
}

inline __m128d load_intrin(const double* a){
  return _mm_load_pd(a);
}

inline void store_intrin(float* a, const __m128& b){
  return _mm_store_ps(a,b);
}

inline void store_intrin(double* a, const __m128d& b){
  return _mm_store_pd(a,b);
}

inline __m128 mul_intrin(const __m128& a, const __m128& b){
  return _mm_mul_ps(a,b);
}

inline __m128d mul_intrin(const __m128d& a, const __m128d& b){
  return _mm_mul_pd(a,b);
}

inline __m128 add_intrin(const __m128& a, const __m128& b){
  return _mm_add_ps(a,b);
}

inline __m128d add_intrin(const __m128d& a, const __m128d& b){
  return _mm_add_pd(a,b);
}

inline __m128 sub_intrin(const __m128& a, const __m128& b){
  return _mm_sub_ps(a,b);
}

inline __m128d sub_intrin(const __m128d& a, const __m128d& b){
  return _mm_sub_pd(a,b);
}

inline __m128 rsqrt_approx_intrin(const __m128& r2){
  return _mm_andnot_ps(_mm_cmpeq_ps(r2,_mm_setzero_ps()),_mm_rsqrt_ps(r2));
}

inline __m128d rsqrt_approx_intrin(const __m128d& r2){
  return _mm_cvtps_pd(rsqrt_approx_intrin(_mm_cvtpd_ps(r2)));
}

inline void rsqrt_newton_intrin(__m128& rinv, const __m128& r2, const float& nwtn_const){
  rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
}

inline void rsqrt_newton_intrin(__m128d& rinv, const __m128d& r2, const double& nwtn_const){
  rinv=mul_intrin(rinv,sub_intrin(set_intrin(nwtn_const),mul_intrin(r2,mul_intrin(rinv,rinv))));
}

#endif //__SSE3__
#endif //__AVX__

inline Vec_t rsqrt_intrin2(Vec_t r2){
  Vec_t rinv=rsqrt_approx_intrin(r2);
  rsqrt_newton_intrin(rinv,r2,Real_t(3));
  rsqrt_newton_intrin(rinv,r2,Real_t(12));
  return rinv;
}

}

#endif //_PVFMM_INTRIN_WRAPPER_HPP_
