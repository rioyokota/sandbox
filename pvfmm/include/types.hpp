#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

#ifndef NULL
#define NULL 0
#endif

#ifndef NDEBUG
#define NDEBUG
#endif

#define MAX_DEPTH 62
#define MEM_ALIGN 64
#define DEVICE_BUFFER_SIZE 1024LL
#define CACHE_SIZE 64

#define UNUSED(x) (void)(x) // to ignore unused variable warning.

#if FLOAT
typedef float Real_t;
#if defined __AVX__
typedef __m256 Vec_t;
#elif defined __SSE3__
typedef __m128 Vec_t;
#else
typedef Real_t Vec_t;
#endif
#else
typedef double Real_t;
#if defined __AVX__
typedef __m256d Vec_t;
#elif defined __SSE3__
typedef __m128d Vec_t;
#else
typedef Real_t Vec_t;
#endif
#endif

#endif //_PVFMM_COMMON_HPP_
