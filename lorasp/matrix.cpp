#include <cmath>
#include <fstream>
#include <iostream>

double kernel(int i, int j) {
  double r = std::abs(i - j);
  return exp(-r*r/10);
}

int main() {
  const int N = 100;
  const double eps = 1e-12;
  int nnz = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      double f = kernel(i,j);
      if (f > eps) {
        nnz++;
      }
    }
  }
  std::cout << nnz << "/" << N*N << std::endl;
  std::fstream fid("A.mtx", std::fstream::out);
  fid << N << " " << N << " " << nnz << std::endl;
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      double f = kernel(i,j);
      if (f > eps) {
        fid << i+1 << " " << j+1 << " " << f << std::endl;
      }
    }
  }
  fid.close();
}
