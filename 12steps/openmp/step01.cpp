#include <cstdlib>
#include <cstdio>
#include <ctime>
int main(int argc, char ** argv) {
  int n = atoi(argv[1]);
  double * a = new double [n];
  double * b = new double [n];
  std::clock_t tic = std::clock();
#pragma omp parallel for
  for (int i=1; i<n; i++) {
    b[i] = (a[i] + a[i-1]) / 2.0;
  }
  std::clock_t toc = std::clock();
  printf("%lf s\n",(toc-tic)/double(CLOCKS_PER_SEC));
  delete[] a;
  delete[] b;
}
