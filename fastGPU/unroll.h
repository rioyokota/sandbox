#ifndef unroll_h
#define unroll_h

namespace Ops {
  template<typename T> struct Assign {
    __host__ __device__ __forceinline__
    const T operator() (T & a, const T & b) const {
      return a = b;
    }
  };
  template<typename T> struct Add {
    __host__ __device__ __forceinline__
    const T operator() (T & a, const T & b) const {
      return a += b;
    }
  };
  template<typename T> struct Sub {
    __host__ __device__ __forceinline__
    const T operator() (T & a, const T & b) const {
      return a -= b;
    }
  };
  template<typename T> struct Mul {
    __host__ __device__ __forceinline__
    const T operator() (T & a, const T & b) const {
      return a *= b;
    }
  };
  template<typename T> struct Div {
    __host__ __device__ __forceinline__
    const T operator() (T & a, const T & b) const {
      return a /= b;
    }
  };
  template<typename T> struct Gt {
    __host__ __device__ __forceinline__
    bool operator() (T & a, const T & b) const {
      return a >= b;
    }
  };
  template<typename T> struct Lt {
    __host__ __device__ __forceinline__
    bool operator() (T & a, const T & b) const {
      return a <= b;
    }
  };
  template<typename T> struct And {
    __host__ __device__ __forceinline__
    int operator() (int & a, const int & b) const {
      return a &= b;
    }
  };
  template<typename T> struct Or {
    __host__ __device__ __forceinline__
    int operator() (int & a, const int & b) const {
      return a |= b;
    }
  };
}

template<typename Op, typename T, int N>
  struct Unroll {
    __host__ __device__ __forceinline__
    static void loop(T * data, const T * v) {
      Op operation;
      operation(data[N-1], v[N-1]);
      Unroll<Op,T,N-1>::loop(data, v);
    }
    __host__ __device__ __forceinline__
    static void loop(T * data, const T v) {
      Op operation;
      operation(data[N-1], v);
      Unroll<Op,T,N-1>::loop(data, v);
    }
    __host__ __device__ __forceinline__
    static const T reduce(const T * v) {
      Op operation;
      return operation(const_cast<T*>(v)[N-1],Unroll<Op,T,N-1>::reduce(v));
    }
  };

template<typename Op, typename T>
  struct Unroll<Op,T,1> {
    __host__ __device__ __forceinline__
    static void loop(T * data, const T * v) {
      Op operation;
      operation(data[0], v[0]);
    }
    __host__ __device__ __forceinline__
    static void loop(T * data, const T v) {
      Op operation;
      operation(data[0], v);
    }
    __host__ __device__ __forceinline__
    static const T reduce(const T * v) {
      return v[0];
    }
};

#endif
