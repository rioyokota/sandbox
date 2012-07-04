#ifndef stack_h
#define stack_h

template<int N, typename T>
class Stack {
private:
  T *TOP;
  T LIST[N];

public:
  __host__ __device__
  Stack() : TOP(LIST) {}

  __host__ __device__
  void push(T const &x) {
    *(TOP++) = x;
  }

  __host__ __device__
  T pop() {
    return *(--TOP);
  }

  __host__ __device__
  int size() {
    return TOP-LIST;
  }

  __host__ __device__
  bool empty() {
    return TOP-LIST == 0;
  }
};
#endif
