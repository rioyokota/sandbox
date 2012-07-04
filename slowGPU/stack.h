#ifndef stack_h
#define stack_h

template<int N, typename T>
class Stack {
private:
  T *TOP;
  T LIST[N];

public:
  __device__
  Stack() : TOP(LIST) {}

  __device__
  void push(T const &x) {
    *(TOP++) = x;
  }

  __device__
  T pop() {
    return *(--TOP);
  }

  __device__
  int size() {
    return TOP-LIST;
  }

  __device__
  bool empty() {
    return TOP-LIST == 0;
  }
};
#endif
