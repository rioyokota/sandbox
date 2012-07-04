#ifndef pair_h
#define pair_h

template<typename T1, typename T2>
struct Pair {
  T1 first;
  T2 second;
  __device__
  Pair() : first(T1()), second(T2()) {}
  __device__
  Pair(T1 f, T2 s) : first(f), second(s) {}
};
#endif
