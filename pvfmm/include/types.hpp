#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

#ifndef NULL
#define NULL 0
#endif

#ifndef NDEBUG
#define NDEBUG
#endif

#define MAX_DEPTH 15

#if MAX_DEPTH < 7
#define UINT_T uint8_t
#define  INT_T  int8_t
#elif MAX_DEPTH < 15
#define UINT_T uint16_t
#define  INT_T  int16_t
#elif MAX_DEPTH < 31
#define UINT_T uint32_t
#define  INT_T  int32_t
#elif MAX_DEPTH < 63
#define UINT_T uint64_t
#define  INT_T  int64_t
#endif

#define BC_LEVELS 60

#define RAD0 1.05 //Radius of upward equivalent (downward check) surface.
#define RAD1 2.95 //Radius of downward equivalent (upward check) surface.

#define MEM_ALIGN 64
#define DEVICE_BUFFER_SIZE 1024LL //in MB
#define V_BLK_CACHE 25 //in KB
#define PVFMM_MAX_COORD_HASH 2000

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
