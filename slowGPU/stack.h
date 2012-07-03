#ifndef stack_h
#define stack_h

template<int N, typename T>
class Stack {
private:
  T *TOP;
  T LIST[N];

public:
  Stack() : TOP(LIST) {}

  void push(T const &x) {
    *(TOP++) = x;
  }

  T pop() {
    return *(--TOP);
  }

  bool empty() {
    return TOP-LIST == 0;
  }
};
#endif
