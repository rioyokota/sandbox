#include <cstdio>

extern "C" void vecref_( int[], int * );

int main() {
  int i, sum;
  int v[9] = {1,1,1,1,1,1,1,1,1};
  vecref_( v, &sum );
  printf("%d\n",sum);
}
