#include <iostream>

template<int nx, int ny, int nz>
struct TaylorIndex {
  static const int I = TaylorIndex<nx,ny+1,nz-1>::I + 1;
  static const int F = TaylorIndex<nx,ny,nz-1>::F * nz;
};

template<int nx, int ny>
struct TaylorIndex<nx,ny,0> {
  static const int I = TaylorIndex<nx+1,0,ny-1>::I + 1;
  static const int F = TaylorIndex<nx,ny-1,0>::F * ny;
};

template<int nx>
struct TaylorIndex<nx,0,0> {
  static const int I = TaylorIndex<0,0,nx-1>::I + 1;
  static const int F = TaylorIndex<nx-1,0,0>::F * nx;
};

template<>
struct TaylorIndex<0,0,0> {
  static const int I = 0;
  static const int F = 1;
};

int main() {
  std::cout << TaylorIndex<2,2,3>::F << std::endl;
}
