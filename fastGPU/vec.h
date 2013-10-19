#ifndef vec_h
#define vec_h
#include <ostream>
#define NEWTON 1

template<int N>
struct Vec {
  __host__ __device__ __forceinline__
  static void sub(float * temp, const float * data, const float * v) {
    temp[N-1] = data[N-1] - v[N-1];
    Vec<N-1>::sub(temp, data, v);
  }
};

template<>
struct Vec<1> {
  __host__ __device__ __forceinline__
  static void sub(float * temp, const float * data, const float * v) {
    temp[0] = data[0] - v[0];
  }
};

//! Custom vector type for small vectors with template specialization for MIC, AVX, SSE intrinsics
template<int N, typename T>
  struct vec {
  private:
    T data[N];
  public:
    __host__ __device__ __forceinline__
    const vec &operator=(const vec &v) {                        // Vector assignment
#pragma unroll
      for (int i=0; i<N; i++) data[i] = v[i];
      return *this;
    }
    __host__ __device__ __forceinline__
    vec operator-(const vec &v) const {                         // Vector arithmetic (subtract)
      vec temp;
      Vec<N>::sub(temp,data,v);
      return temp;
    }
    __host__ __device__ __forceinline__
    T &operator[](int i) {                                      // Indexing (lvalue)
      assert(i < N);
      return data[i];
    }
    __host__ __device__ __forceinline__
    const T &operator[](int i) const {                          // Indexing (rvalue)
      assert(i < N);
      return data[i];
    }
    __host__ __device__ __forceinline__
    operator       T* ()       {return data;}                   // Type-casting (lvalue)
    __host__ __device__ __forceinline__
    operator const T* () const {return data;}                   // Type-casting (rvalue)
    __host__ __device__ __forceinline__
    friend T norm(const vec &v) {                               // L2 norm squared
      T temp = 0;
#pragma unroll
      for (int i=0; i<N; i++) temp += v[i] * v[i];
      return temp;
    }
  };

#endif
