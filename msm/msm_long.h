#ifndef MSM_LONG_H_
#define MSM_LONG_H_

#include <map>
#include <iomanip>
#include <iterator>
#include "time_logger.h"
#include "system.h"
#include "ewald_short.h"
#include "ewald_long.h"

////////////////////////////////////////////////////////////////////////////////
class Utility{
public:
  static void read_parameter(const std::string& file,
    std::map<std::string, std::string>& parameter){
    std::ifstream in(file.c_str());

    if(!in){
      throw std::invalid_argument(file + " can not be opened.");
    }

    std::string line;

    while(std::getline(in, line)){
      std::istringstream ss(line);
      std::istream_iterator<std::string> it(ss), eos;
      if(it == eos){continue;}

      std::string key = *it;

      it++;

      if(key[0] == '#'){
        continue;
      }

      if(it == eos){
        throw std::invalid_argument(key + " : value is missing");
      }

      parameter[key] = *it;
    }
  }

};

////////////////////////////////////////////////////////////////////////////////
class Array{
public:
  // y = x + y
  static void add(int n, const real* x, real* y){
    for(int i = 0; i < n; i++){
      y[i] += x[i];
    }
  }

  // x *= a
  static void multiply(int n, real* x, real a){
    for(int i = 0; i < n; i++){
      x[i] *= a;
    }
  }

  // sum(x)
  static real sum(int n, const real* x){
    real s = 0;

    for(int i = 0; i < n; i++){
      s += x[i];
    }

    return s;
  }

  static real get_distance(int n, const real* r1, const real* r2){
    real dr2 = 0;

    for(int i = 0; i < n; i++){
      real dri = r2[i] - r1[i];
      dr2 += dri * dri;
    }

    return std::sqrt(dr2);
  }

  // binomial_cofficient(n,k) : 0 <= k <= n
  static void get_binomial_coefficient(int n, real* binom){
    if(n < 0){return;}

    binom[0] = 1;

    if(n == 0){return;}

    binom[n] = 1;

    if(n == 1){return;}

    // n! / (k! (n-k)!)
    for(int k = 1; k < n; k++){
      int k1 = std::min(k, n - k);
      real num = n - k1 + 1;
      real s = num;

      for(int i = 2; i <= k1; i++){
        num = num + 1;
        s *= (num / i);
      }

      binom[k] = s;
    }
  }

  static void print_separator(std::ostream& o){
    for(int i = 0; i < 80; i++){
      o << "=";
    }

    o << std::endl;
  }

  static void divide_integer(int n, int nthread, int* start, int* count){
    const int res = n % nthread;
    const int ni = (n - res) / nthread;

    std::fill_n(count, nthread, ni);

    for(int i = 0; i < res; i++){
      count[i]++;
    }

    start[0] = 0;

    for(int i = 1; i < nthread; i++){
      start[i] = start[i - 1] + count[i - 1];
    }
  }

  static void print_matrix(int m, int n, const real* A, std::ostream& o){
    for(int i = 0; i < m; i++){
      const real* Ai = &A[i * n];

      for(int j = 0; j < n; j++){
        o << " " << Ai[j];
      }

      o << std::endl;
    }
  }

  static void print_matrix(int m, int n, const real* x,
    int width, int precision, std::ostream& o){
    int ij = 0;

    for(int i = 0; i < m; i++){
      for(int j = 0; j < n; j++){
        o << std::right << std::scientific << std::setw(width)
          << std::setprecision(precision) << x[ij++];
      }

      o << std::endl;
    }
  }
/*
  static void print_array3d(int nx, int ny, int nz, const real* a,
    std::ostream& o){
    for(int i = 0; i < nx; i++){
      for(int j = 0; j < ny; j++){
        for(int k = 0; k < nz; k++){
          o << "[" << i << "][" << j << "][" << k << "] "
            << a[i * ny * nz + j * nz + k] << std::endl;
        }
      }
    }
  }
*/
  static void copy_index_value(int n, const int* index, const real* x, real* y){
    for(int i = 0; i < n; i++){
      y[i] = x[index[i]];
    }
  }

  static int get_periodic_index(int period, int i){
    if(i >= 0){return i % period;}

    // 0 <= n * period + i < period
    // -i <= n * period < period - i
    // (-i)/period <= n < 1 + (-i)/period
    int r = (-i) % period;

    if(r == 0){
      return 0;
    }
    else{
      return period - r;
    }
  }

  static void get_periodic_index(int period, int start, int count, int* index){
    if(count == 0){return;}

    int j = get_periodic_index(period, start);
    index[0] = j;

    for(int i = 1; i < count; i++){
      j++;

      if(j == period){
        j = 0;
      }

      index[i] = j;
    }
  }

  // one-sided kernel k to two-sided kernel K
  static void get_two_sided_kernel(int nk, const real* k, real* K){
    const int middle = nk - 1;
    int forward = middle + 1;
    int backward = middle - 1;
    K[middle] = k[0];

    for(int i = 1; i < nk; i++){
      K[forward++] = k[i];
      K[backward--] = k[i];
    }
  }

  // convolve(kx ky kz, f)
  static void get_periodic_convolution3d_symmetric_kernel(
    int nkx, const real* kx, int nky, const real* ky, int nkz, const real* kz,
    int nx, int ny, int nz, const real* f, real* kf){
    const int nKx = 2 * nkx - 1;
    real* Kx = new real[nKx];
    const int nKy = 2 * nky - 1;
    real* Ky = new real[nKy];
    const int nKz = 2 * nkz - 1;
    real* Kz = new real[nKz];

    get_two_sided_kernel(nkx, kx, Kx);
    get_two_sided_kernel(nky, ky, Ky);
    get_two_sided_kernel(nkz, kz, Kz);

    // kz * f : z-direction convolution
    const int nmzindex = nz + 2 * (nkz - 1);
    int* mzindex = new int[nmzindex];
    get_periodic_index(nz, - (nkz - 1), nmzindex, mzindex);
    real* fxy = new real[nmzindex];
    const int nxny = nx * ny;
    const int nynz = ny * nz;
    const real zero = 0;

    for(int mx = 0; mx < nx; mx++){
      const real* fx = &f[mx * nynz];
      real* kzfx = &kf[mx * ny];

      for(int my = 0; my < ny; my++){
        copy_index_value(nmzindex, mzindex, &fx[my * nz], fxy);
        real* kzfxy = &kzfx[my];

        for(int mz = 0; mz < nz; mz++){
          // xyz-order -> zxy-order
          kzfxy[mz * nxny] = std::inner_product(Kz, Kz + nKz, &fxy[mz], zero);
        }
      }
    }

    delete [] mzindex;
    delete [] fxy;


    // ky * (kz * f) : y-direction convolution
    const int nmyindex = ny + 2 * (nky - 1);
    int* myindex = new int[nmyindex];
    get_periodic_index(ny, - (nky - 1), nmyindex, myindex);
    real* kzfzx = new real[nmyindex];
    real* kykzf = new real[nx * ny * nz];
    const int nznx = nz * nx;

    // zxy-order
    for(int mz = 0; mz < nz; mz++){
      const real* kzfz = &kf[mz * nxny];
      real* kykzfz = &kykzf[mz * nx];

      for(int mx = 0; mx < nx; mx++){
        copy_index_value(nmyindex, myindex, &kzfz[mx * ny], kzfzx);
        real* kykzfzx = &kykzfz[mx];

        for(int my = 0; my < ny; my++){
          // zxy-order -> yzx-order
          kykzfzx[my * nznx] = std::inner_product(
            Ky, Ky + nKy, &kzfzx[my], zero);
        }
      }
    }

    delete [] myindex;
    delete [] kzfzx;


    // kx * (ky * (kz * f)) : x-direction convolution
    const int nmxindex = nx + 2 * (nkx - 1);
    int* mxindex = new int[nmxindex];
    get_periodic_index(nx, - (nkx - 1), nmxindex, mxindex);
    real* kykzfyz = new real[nmxindex];

    // yzx-order
    for(int my = 0; my < ny; my++){
      const real* kykzfy = &kykzf[my * nznx];
      real* kfy = &kf[my * nz];

      for(int mz = 0; mz < nz; mz++){
        copy_index_value(nmxindex, mxindex, &kykzfy[mz * nx], kykzfyz);
        real* kfyz = &kfy[mz];

        for(int mx = 0; mx < nx; mx++){
          // yzx-order -> xyz-order
          kfyz[mx * nynz] = std::inner_product(
            Kz, Kz + nKz, &kykzfyz[mx], zero);
        }
      }
    }

    delete [] mxindex;
    delete [] kykzfyz;

    delete [] Kx;
    delete [] Ky;
    delete [] Kz;
    delete [] kykzf;
  }

};

////////////////////////////////////////////////////////////////////////////////
class Matrix{
public:
  // Ax = b
  static void solve_gaussian_elimination(int n, const real* A,
    const real* b, real* x){
    real* B = new real[n * (n + 1)];

    for(int i = 0; i < n; i++){
      const real* Ai = &A[i * n];
      real* Bi = &B[i * (n + 1)];

      std::copy_n(Ai, n, Bi);
      Bi[n] = b[i];
    }

    for(int i = 0; i < n; i++){
      real* Bi = &B[i * (n + 1)];
      const real Bii_inv = 1.0 / Bi[i];

      for(int k = i + 1; k <= n; k++){
        Bi[k] *= Bii_inv;
      }

      Bi[i] = 1;

      for(int k = i + 1; k < n; k++){
        real* Bk = &B[k * (n + 1)];

        if(Bk[i] != 0){
          real Bki = Bk[i];

          for(int j = i + 1; j <= n; j++){
            Bk[j] -= Bki * Bi[j];
          }

          Bk[i] = 0;
        }

      }
    }

    for(int i = n - 1; i >= 0; i--){
      const real* Bi = &B[i * (n + 1)];
      real sum = 0;

      for(int j = i + 1; j < n; j++){
        sum += Bi[j] * x[j];
      }

      x[i] = Bi[n] - sum;
    }

    delete [] B;
  }

  static void test_solve_gaussian_elimination(std::ostream& o){
    const int n = 4;
    real* A = new real[n * n];
    real* b = new real[n];
    real* x0 = new real[n];
    real* x = new real[n];

    for(int i = 0; i < n; i++){
      x0[i] = i + 1;
    }

    for(int i = 0; i < n; i++){
      real* Ai = &A[i * n];

      Ai[i] = 2;

      if(i - 1 >= 0){
        Ai[i - 1] = 1;
      }

      if(i + 1 < n){
        Ai[i + 1] = 1;
      }
    }

    for(int i = 0; i < n; i++){
      real* Ai = &A[i * n];
      real sum = 0;

      for(int j = 0; j < n; j++){
        sum += Ai[j] * x0[j];
      }

      b[i] = sum;
    }

    o << "x0" << std::endl;
    Array::print_matrix(1, n, x0, o);

    o << "A" << std::endl;
    Array::print_matrix(n, n, A, o);

    o << "b" << std::endl;
    Array::print_matrix(1, n, b, o);

    solve_gaussian_elimination(n, A, b, x);

    o << "x" << std::endl;
    Array::print_matrix(1, n, x, o);

    delete [] A;
    delete [] b;
    delete [] x0;
    delete [] x;
  }

};

////////////////////////////////////////////////////////////////////////////////
class GaussianQuadrature{
public:
  // https://en.wikipedia.org/wiki/Gaussian_quadrature
  // https://pomax.github.io/bezierinfo/legendre-gauss.html
  static void get_gaussian_quadrature_point(int n, real* x, real* w){
    if(n == 1){
      x[0] = 0;
      w[0] = 2;
    }
    else if(n == 2){
      w[0] = 1.0000000000000000; x[0] = -0.5773502691896257;
      w[1] = 1.0000000000000000; x[1] = 0.5773502691896257;
    }
    else if(n == 3){
      w[0] = 0.5555555555555556; x[0] = -0.7745966692414834;
      w[1] = 0.8888888888888888; x[1] = 0.0000000000000000;
      w[2] = 0.5555555555555556; x[2] = 0.7745966692414834;
    }
    else if(n == 4){
      w[0] = 0.3478548451374538; x[0] = -0.8611363115940526;
      w[1] = 0.6521451548625461; x[1] = -0.3399810435848563;
      w[2] = 0.6521451548625461; x[2] = 0.3399810435848563;
      w[3] = 0.3478548451374538; x[3] = 0.8611363115940526;
    }
    else if(n == 5){
      w[0] = 0.2369268850561891; x[0] = -0.9061798459386640;
      w[1] = 0.4786286704993665; x[1] = -0.5384693101056831;
      w[2] = 0.5688888888888889; x[2] = 0.0000000000000000;
      w[3] = 0.4786286704993665; x[3] = 0.5384693101056831;
      w[4] = 0.2369268850561891; x[4] = 0.9061798459386640;
    }
    else if(n == 6){
      w[0] = 0.1713244923791704; x[0] = -0.9324695142031521;
      w[1] = 0.3607615730481386; x[1] = -0.6612093864662645;
      w[2] = 0.4679139345726910; x[2] = -0.2386191860831969;
      w[3] = 0.4679139345726910; x[3] =  0.2386191860831969;
      w[4] = 0.3607615730481386; x[4] =  0.6612093864662645;
      w[5] = 0.1713244923791704; x[5] =  0.9324695142031521;
    }
    else if(n == 7){
      w[0] = 0.1294849661688697; x[0] = -0.9491079123427585;
      w[1] = 0.2797053914892766; x[1] = -0.7415311855993945;
      w[2] = 0.3818300505051189; x[2] = -0.4058451513773972;
      w[3] = 0.4179591836734694; x[3] =  0.0000000000000000;
      w[4] = 0.3818300505051189; x[4] =  0.4058451513773972;
      w[5] = 0.2797053914892766; x[5] =  0.7415311855993945;
      w[6] = 0.1294849661688697; x[6] =  0.9491079123427585;
    }
    else if(n == 8){
      w[0] = 0.1012285362903763; x[0] = -0.9602898564975363;
      w[1] = 0.2223810344533745; x[1] = -0.7966664774136267;
      w[2] = 0.3137066458778873; x[2] = -0.5255324099163290;
      w[3] = 0.3626837833783620; x[3] = -0.1834346424956498;
      w[4] = 0.3626837833783620; x[4] =  0.1834346424956498;
      w[5] = 0.3137066458778873; x[5] =  0.5255324099163290;
      w[6] = 0.2223810344533745; x[6] =  0.7966664774136267;
      w[7] = 0.1012285362903763; x[7] =  0.9602898564975363;
    }
    else if(n == 9){
      w[0] = 0.0812743883615744; x[0] = -0.9681602395076261;
      w[1] = 0.1806481606948574; x[1] = -0.8360311073266358;
      w[2] = 0.2606106964029354; x[2] = -0.6133714327005904;
      w[3] = 0.3123470770400029; x[3] = -0.3242534234038089;
      w[4] = 0.3302393550012598; x[4] = 0.0000000000000000;
      w[5] = 0.3123470770400029; x[5] = 0.3242534234038089;
      w[6] = 0.2606106964029354; x[6] = 0.6133714327005904;
      w[7] = 0.1806481606948574; x[7] = 0.8360311073266358;
      w[8] = 0.0812743883615744; x[8] = 0.9681602395076261;
    }
    else if(n == 10){
      w[0] = 0.2955242247147529; x[0] = -0.1488743389816312;
      w[1] = 0.2955242247147529; x[1] = 0.1488743389816312;
      w[2] = 0.2692667193099963; x[2] = -0.4333953941292472;
      w[3] = 0.2692667193099963; x[3] = 0.4333953941292472;
      w[4] = 0.2190863625159820; x[4] = -0.6794095682990244;
      w[5] = 0.2190863625159820; x[5] = 0.6794095682990244;
      w[6] = 0.1494513491505806; x[6] = -0.8650633666889845;
      w[7] = 0.1494513491505806; x[7] = 0.8650633666889845;
      w[8] = 0.0666713443086881; x[8] = -0.9739065285171717;
      w[9] = 0.0666713443086881; x[9] = 0.9739065285171717;
    }
    else{
      throw std::invalid_argument("1 <= quadrature_point <= 10");
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
class Bspline{
public:
  // Bp(x)
  static real get_Bspline(real x, int p){
    if(p == 1){
      if(x >= 0 && x < 1){
         return 1;
      }
      else{
        return 0;
      }
    }
    else{
      const real p1 = p - 1.0;
      const real p1_inv = 1.0 / p1; // 1/(p-1)

      return p1_inv * (x * get_Bspline(x, p1)
        + (p - x) * get_Bspline(x - 1.0, p1));
    }
  }

  // dBp(x)/dx
//  static real get_Bspline_derivative(real x, int p){
//TODO p == 1
//    return get_Bspline(x, p - 1) - get_Bspline(x - 1, p - 1);
//  }

//TODO return value
  // Bp(x), dBp(x)/dx, p >= 2
  static void get_Bspline_value_derivative(real x, int p,
    real* value, real* deriv){
    real p1 = p - 1.0;
    real p1_inv = 1.0 / p1;
    real a = get_Bspline(x, p1);
    real b = get_Bspline(x - 1, p1);

//    *value = p1_inv * (x * a + (p - x) * b);
    *deriv = a - b;
    *value = p1_inv * (x * *deriv + p * b);
  }

  // Cp(x) = Bp(x + p / 2)
  // y = Cp*x -> y = A x
  // y is symmetric series : y[-i] = y[i]
  // 0 <= i, j <= n matrix is returned
  static void get_centered_Bspline_convolution_matrix(int p, int n, real* A){
    const real p_2 = p / 2.0;
    int xmax = std::floor(p_2);
    real* C = new real[xmax + 1];

    for(int i = 0; i <= xmax; i++){
      C[i] = get_Bspline(i + p_2, p);
    }

    std::fill_n(A, (n + 1) * (n + 1), 0);

    for(int i = 0; i <= n; i++){
      real* Ai = &A[i * (n + 1)];

      int jmin = std::ceil(i - p_2);
      int jmax = std::floor(i + p_2);

      for(int j = jmin; j <= jmax; j++){
        int j_index = std::abs(j);

        if(j_index <= n){
          int index = std::abs(i - j);
          Ai[j_index] += C[index];
        }
      }
    }

    delete [] C;
  }

  static void test_centered_Bspline_convolution_matrix(int p, int n,
    std::ostream& o){
    real* A = new real[(n + 1) * (n + 1)];
   
    get_centered_Bspline_convolution_matrix(p, n, A);

    o << "p " << p << std::endl;
    o << "n " << n << std::endl;
    o << "A" << std::endl;
    Array::print_matrix(n + 1, n + 1, A, o);

    delete [] A;
  }
/*
  // f(x) = sum_i c_i Bp(x + p / 2 - i)
  static real get_centered_Bspline_interpolation(real x, int p, int n, real* c){
    // 0 <= x + p / 2 - i < p
    // x - p / 2 < i <= x + p / 2
    int imin = std::ceil(x - p / 2.0);
    int imax = std::floor(x + p / 2.0);
    real y = 0;

    for(int i = imin; i <= imax; i++){
      int index = std::abs(i);

      y += c[index] * get_Bspline(x + p / 2.0 - i, p);
    }

    return y;
  }

  // f(x,y) = sum_ij a(i-j) Bp(x+p/2-i) Bp(y+p/2-j)
  static real get_centered_Bspline_interpolation2d(
    real x, real y, int p, int n, real* a){
    // 0 <= x + p / 2 - i < p
    // x - p / 2 < i <= x + p / 2
    int imin = std::ceil(x - p / 2.0);
    int imax = std::floor(x + p / 2.0);
    int jmin = std::ceil(y - p / 2.0);
    int jmax = std::floor(y + p / 2.0);
    real f = 0;

    for(int i = imin; i <= imax; i++){
      real Bi = get_Bspline(x + p / 2.0 - i, p);

      for(int j = jmin; j <= jmax; j++){
        int index = i - j;

        if(index < 0){
          index = -index;
        }

        f += a[index] * Bi * get_Bspline(y + p / 2.0 - j, p);
      }
    }

    return f;
  }

  // exp(-x^2/(2*sigma^2)) = sum_i c_i Bp(x + p/2 - i)
  static void test_gaussian_interpolation(int pmax, int n, real sigma,
    std::ostream& o){
    real* y = new real[n + 1];
    real* A = new real[(n + 1) * (n + 1)];
    real* c = new real[(n + 1) * pmax];

    for(int i = 0; i <= n; i++){
      real xi = i / sigma;
      y[i] = std::exp(- xi * xi / 2.0);
    }

    for(int p = 1; p <= pmax; p++){
      get_centered_Bspline_convolution_matrix(p, n, A);
      real* cp = &c[(p - 1) * (n + 1)];
      Matrix::solve_gaussian_elimination(n + 1, A, y, cp);
    }

    const real xmax = 10;
    const int nx = 201;
    const real dx = xmax / (nx - 1);

    o << "x exp(-(x/" << sigma << ")^2/2)";

    for(int p = 1; p <= pmax; p++){
      o << " p=" << p;
    }

    o << std::endl;

    for(int i = 0; i < nx; i++){
      real xi = i * dx;
      real x0 = xi / sigma;
      real y0 = std::exp(- x0 * x0 / 2.0);

      o << xi << " " << y0;

      for(int p = 1; p <= pmax; p++){
        real* cp = &c[(p - 1) * (n + 1)];
        real yinter = get_centered_Bspline_interpolation(xi, p, n, cp);

        o << " " << yinter;
      }

      o << std::endl;
    }

    delete [] A;
    delete [] y;
    delete [] c;
  }

  // exp(-(x-y)^2/(2*sigma^2)) = sum_ij c(i-j) Bp(x+p/2-i) Bp(y+p/2-j)
  static void test_gaussian_interpolation2d(int pmax, int n, real sigma,
    real y, std::ostream& o){
    real* g = new real[n + 1];
    real* A = new real[(n + 1) * (n + 1)];
    real* cp = new real[n + 1];
    real* a = new real[(n + 1) * pmax];

    for(int i = 0; i <= n; i++){
      real xi = i / sigma;
      g[i] = std::exp(- xi * xi / 2.0);
    }

    for(int p = 1; p <= pmax; p++){
      get_centered_Bspline_convolution_matrix(p, n, A);
      real* ap = &a[(p - 1) * (n + 1)];
      Matrix::solve_gaussian_elimination(n + 1, A, g, cp);
      Matrix::solve_gaussian_elimination(n + 1, A, cp, ap);
    }

    const real xmax = 10;
    const int nx = 201;
    const real dx = xmax / (nx - 1);

    o << "x exp(-((x-" << y << ")/" << sigma << ")^2/2)";

    for(int p = 1; p <= pmax; p++){
      o << " p=" << p;
    }

    o << std::endl;

    for(int i = 0; i < nx; i++){
      real xi = i * dx;
      real x0 = (xi - y) / sigma;
      real g0 = std::exp(- x0 * x0 / 2.0);

      o << xi << " " << g0;

      for(int p = 1; p <= pmax; p++){
        real* ap = &a[(p - 1) * (n + 1)];
        real f = get_centered_Bspline_interpolation2d(xi, y, p, n, ap);

        o << " " << f;
      }

      o << std::endl;
    }

    delete [] A;
    delete [] g;
    delete [] cp;
    delete [] a;
  }
*/
  // 0 <= x < 1
  // Bp(x+(p-1)), ..., Bp(x)
  static void get_reversed_Bspline(int p, real x, real* B){
    std::fill_n(B, p, 0);
    B[p - 1] = 1.0;

    for(int i = 1; i < p; i++){
      const real p_inv = 1.0 / i;
      real a = p_inv * (1.0 - x);

      for(int j = p - i; j < p; j++){
        B[j - 1] += a * B[j];
        B[j] *= 1.0 - a;
        a += p_inv;
      }
    }
  }

  // 0 <= x < 1
  // Bp(x+(p-1)),     ..., Bp(x)
  // dBp(x+(p-1))/dx, ..., dBp(x)/dx
  static void get_reversed_Bspline_value_derivative(int p, real x,
    real* B, real* dB){
    std::fill_n(B, p, 0);
    B[p - 1] = 1.0;

    for(int i = 1; i < p - 1; i++){
      const real p_inv = 1.0 / i;
      real a = p_inv * (1.0 - x);

      for(int j = p - i; j < p; j++){
        B[j - 1] += a * B[j];
        B[j] *= 1.0 - a;
        a += p_inv;
      }
    }

    std::fill_n(dB, p, 0);

    for(int j = 1; j < p; j++){
      dB[j - 1] -= B[j];
      dB[j] += B[j];
    }

    const real p_inv = 1.0 / (p - 1.0);
    real a = p_inv * (1.0 - x);

    for(int j = 1; j < p; j++){
      B[j - 1] += a * B[j];
      B[j] *= 1.0 - a;
      a += p_inv;
    }
  }

  static void get_periodic_interpolation3d_value_gradient(int p,
    int nx, int ny, int nz, real hx, real hy, real hz, const real* coef,
    int ntarget, const real* rtarget,
    real* value, real* gradient){
    const real hx_inv = 1.0 / hx;
    const real hy_inv = 1.0 / hy;
    const real hz_inv = 1.0 / hz;

    real* Cx = new real[p];
    real* dCx = new real[p];
    real* Cy = new real[p];
    real* dCy = new real[p];
    real* Cz = new real[p];
    real* dCz = new real[p];

    int* xindex = new int[nx + p - 1];
    int* yindex = new int[ny + p - 1];
    int* zindex = new int[nz + p - 1];
    Array::get_periodic_index(nx, - p / 2 + 1, nx + p - 1, xindex);
    Array::get_periodic_index(ny, - p / 2 + 1, ny + p - 1, yindex);
    Array::get_periodic_index(nz, - p / 2 + 1, nz + p - 1, zindex);

    const int nynz = ny * nz;

    for(int i = 0; i < ntarget; i++){
      const real* ri = &rtarget[3 * i];

      real xi = ri[0] * hx_inv;
      real yi = ri[1] * hy_inv;
      real zi = ri[2] * hz_inv;
      int mx0 = std::floor(xi);
      int my0 = std::floor(yi);
      int mz0 = std::floor(zi);

      get_reversed_Bspline_value_derivative(p, xi - mx0, Cx, dCx);
      get_reversed_Bspline_value_derivative(p, yi - my0, Cy, dCy);
      get_reversed_Bspline_value_derivative(p, zi - mz0, Cz, dCz);

      const int* mxindex = &xindex[mx0];
      const int* myindex = &yindex[my0];
      const int* mzindex = &zindex[mz0];

      real f = 0;
      real dx = 0;
      real dy = 0;
      real dz = 0;

      for(int jx = 0; jx < p; jx++){
        const real* cx = &coef[mxindex[jx] * nynz];

        for(int jy = 0; jy < p; jy++){
          real dCxCy = dCx[jx] * Cy[jy];
          real CxdCy = Cx[jx] * dCy[jy];
          real CxCy = Cx[jx] * Cy[jy];
          const real* cxy = &cx[myindex[jy] * nz];

          for(int jz = 0; jz < p; jz++){
            real cxyz = cxy[mzindex[jz]];

            f += cxyz * CxCy * Cz[jz];
            dx += cxyz * dCxCy * Cz[jz];
            dy += cxyz * CxdCy * Cz[jz];
            dz += cxyz * CxCy * dCz[jz];
          }
        }
      }

      value[i] = f;
      real* gi = &gradient[3 * i];
      gi[0] = dx * hx_inv;
      gi[1] = dy * hy_inv;
      gi[2] = dz * hz_inv;
    }

    delete [] Cx;
    delete [] dCx;
    delete [] Cy;
    delete [] dCy;
    delete [] Cz;
    delete [] dCz;
    delete [] xindex;
    delete [] yindex;
    delete [] zindex;
  }

};

////////////////////////////////////////////////////////////////////////////////
class MSMKernel{
public:
  virtual void set_grid_width(real hx, real hy, real hz) = 0;

  virtual void get_periodic_convolution(int l, int nx, int ny, int nz,
    const real* q, real* Kq) = 0;

//  virtual real get_level_0_potential(real r) const = 0;

//  virtual real get_level_0_force(real r) const = 0;

//  virtual real get_level_0_potential_force(real r, const real * dr,
//    real * force) const = 0;

  virtual real get_level_l_potential(int l, real r) const = 0;

  virtual void print_parameter(std::ostream& o) const = 0;

};

////////////////////////////////////////////////////////////////////////////////
class EwaldGaussianMixtureMSMKernel : public MSMKernel{
private:
  real sigma_;
  real cutoff_;
  int p_; // B-spline order
  int nquadrature_;

  int nKx_;
  int nKy_;
  int nKz_;
  real hx_;
  real hy_;
  real hz_;
  real* Kx_;
  real* Ky_;
  real* Kz_;

  real alpha_; // 1 / (sqrt(2) * sigma)
  real beta_; // (1 / (sqrt(2) sigma)) (2 / sqrt(pi))

  real* sigma_quadrature_;
  real* scale_;

  // Approximation of Ewald kernel g(r) by Gaussian quadrature
  // g(r) ~ 2/sqrt(pi) 1/(sqrt(2) sigma) sum_i wi exp(-r^2/(2 sigma[i]^2)) / 4
  // sigma[i] = (4 / (ui + 1)) sigma
  void set_quadrature(){
    real* u = new real[nquadrature_];
    real* w = new real[nquadrature_];

    GaussianQuadrature::get_gaussian_quadrature_point(nquadrature_, u, w);

    sigma_quadrature_ = new real[nquadrature_];
    scale_ = new real[nquadrature_];
    const real a = beta_ / 4.0;

    for(int i = 0; i < nquadrature_; i++){
      sigma_quadrature_[i] = sigma_ * 4.0 / (u[i] + 3.0);
      scale_[i] = std::cbrt(a * w[i]);
    }

    delete [] u;
    delete [] w;
  }

public:
  EwaldGaussianMixtureMSMKernel(real sigma, real cutoff, int p,
    int nquadrature){
    sigma_ = sigma;
    cutoff_ = cutoff;
    p_ = p;
    nquadrature_ = nquadrature;

    nKx_ = 0;
    nKy_ = 0;
    nKz_ = 0;
    hx_ = 0;
    hy_ = 0;
    hz_ = 0;
    Kx_ = nullptr;
    Ky_ = nullptr;
    Kz_ = nullptr;

    alpha_ = 1.0 / (std::sqrt(2) * sigma_);
    beta_ = alpha_ * 2.0 / std::sqrt(M_PI);

    set_quadrature();
  }

  ~EwaldGaussianMixtureMSMKernel(){
    if(Kx_ != nullptr){delete [] Kx_;}
    if(Ky_ != nullptr){delete [] Ky_;}
    if(Kz_ != nullptr){delete [] Kz_;}
    if(sigma_quadrature_ != nullptr){delete [] sigma_quadrature_;}
    if(scale_ != nullptr){delete [] scale_;}
  }

  // exp(-(x-x')^2/sigma^2)
  //  = sum_m sum_n K_(m-n) Bp(x/h + p/2 - m) Bp(x'/h + p/2 - n)
  // - nK <= x/h <= nK
  void get_gaussian_centered_Bspline_coefficient(int nK, real h, real* K){
//TODO create at constructor
    real* g = new real[nK + 1];
    real* A = new real[(nK + 1) * (nK + 1)];
    real* cp = new real[nK + 1];

//TODO compute at once
    Bspline::get_centered_Bspline_convolution_matrix(p_, nK, A);

    for(int i = 0; i < nquadrature_; i++){
      // exp(-x^2/(2 sigma^2)) = sum_j Kj Bp(x/h + p/2 - j))
      // x = h i
      // gi = exp(-(h i)^2/(2 sigma^2)) = sum_j Kj Bp(i + p/2 - j))
      // g = A K : Aij = Bp(i + p/2 - j)
      const real sigmai = sigma_quadrature_[i];
      real* Ki = &K[(nK + 1) * i];

      for(int j = 0; j <= nK; j++){
        const real xj = h * j / sigmai;
        g[j] = std::exp(- xj * xj / 2.0);
      }

      Matrix::solve_gaussian_elimination(nK + 1, A, g, cp);
      Matrix::solve_gaussian_elimination(nK + 1, A, cp, Ki);

      const real scale = scale_[i];

      for(int j = 0; j <= nK; j++){
        Ki[j] *= scale;
      }
    }

    delete [] A;
    delete [] g;
    delete [] cp;
  }

  void set_grid_width(real hx, real hy, real hz){
    if(hx != hx_){
      hx_ = hx;
      int nKx = std::ceil(2 * cutoff_ / hx_);

      if(nKx != nKx_){
        nKx_ = nKx;

        if(Kx_ != nullptr){delete [] Kx_;}

        Kx_ = new real[nquadrature_ * (nKx_ + 1)];
        get_gaussian_centered_Bspline_coefficient(nKx_, hx_, Kx_);
      }
    }

    if(hy != hy_){
      hy_ = hy;
      int nKy = std::ceil(2 * cutoff_ / hy_);

      if(nKy != nKy_){
        nKy_ = nKy;

        if(Ky_ != nullptr){delete [] Ky_;}

        Ky_ = new real[nquadrature_ * (nKy_ + 1)];
        get_gaussian_centered_Bspline_coefficient(nKy_, hy_, Ky_);
      }
    }

    if(hz != hz_){
      hz_ = hz;
      int nKz = std::ceil(2 * cutoff_ / hz_);

      if(nKz != nKz_){
        nKz_ = nKz;

        if(Kz_ != nullptr){delete [] Kz_;}

        Kz_ = new real[nquadrature_ * (nKz_ + 1)];
        get_gaussian_centered_Bspline_coefficient(nKz_, hz_, Kz_);
      }
    }
  }

  void get_periodic_convolution(int l,
    int nlx, int nly, int nlz, const real* q,
    real* Kq){
    const int nl = nlx * nly * nlz;
    real* Kqi = new real[nl];

    std::fill_n(Kq, nl, 0);

    for(int i = 0; i < nquadrature_; i++){
      Array::get_periodic_convolution3d_symmetric_kernel(
        nKx_, &Kx_[(nKx_ + 1) * i],
        nKy_, &Ky_[(nKy_ + 1) * i],
        nKz_, &Kz_[(nKz_ + 1) * i],
        nlx, nly, nlz, q, Kqi);
      Array::add(nl, Kqi, Kq);
    }

    real a = std::exp2(1.0 - l); // 2^(1-l)
    Array::multiply(nl, Kq, a);

    delete [] Kqi;
  }
/*
  real get_level_0_potential(real r) const{
    return std::erfc(alpha_ * r) / r;
  }

  real get_level_0_force(real r) const{
    real x = alpha_ * r;
    real r_inv = 1.0 / r;

    return r_inv * (r_inv * std::erfc(x) + beta_ * std::exp(- x * x));
  }

  real get_level_0_potential_force(real r, const real* dr, real* force) const{
    real r_inv = 1.0 / r;
    real x = alpha_ * r;
    real potential = r_inv * std::erfc(x);

    real fr = r_inv * (potential + beta_ * std::exp(- x * x));
    real fr_r = fr * r_inv; // F(r)/r

    force[0] = fr_r * dr[0]; // F(r) (x/r) = F(r)/r x
    force[1] = fr_r * dr[1];
    force[2] = fr_r * dr[2];

    return potential;
  }
*/

  real get_level_l_potential(int l, real r) const{
//    return get_multilevel_ewald_kernel(r, sigma_, 2.0, l);
    if(l == 0){
      return 1.0 / r - EwaldShort::get_ewald_kernel(r, sigma_);
    }
    else{
      real s1 = sigma_ * std::exp2(l - 1); // 2^(l-1)
      real s2 = 2 * s1;

      return EwaldShort::get_ewald_kernel(r, s1)
        - EwaldShort::get_ewald_kernel(r, s2);
    }
  }

  void print_parameter(std::ostream& o) const{
    o << "msm_kernel parameter:" << std::endl;
    o << "kernel " << "ewald_gaussian_mixture" << std::endl;
    o << "sigma " << sigma_ << std::endl;
    o << "cutoff " << cutoff_ << std::endl;
    o << "p " << p_ << std::endl;
    o << "quadrature " << nquadrature_ << std::endl;
//    real h_; // grid width
    o << "kernel_radius " << nKx_ << " " << nKy_ << " " << nKz_ << std::endl;
  }

};

////////////////////////////////////////////////////////////////////////////////
class MSMLong{
private:
  MSMKernel* kernel_;
  int p_; // B-spline order (even number)
  int p_2_; // p / 2
  int level_;

  int natom_;
  const real* q_;
  const real* r_;
  real L_[3];

  real self_potential_;

  real* J_;
  int nJeven_;
  int nJodd_;
  real* Jeven_;
  real* Jodd_;

  real* value_;

  int lmax_;
  int* grid_start_;
  int* nlx_;
  int* nly_;
  int* nlz_;
  int* nl_;
  int n_;
  real hx_;
  real hy_;
  real hz_;

  real* ql_;
  real* elplus_;

  TimeLogger log_;

  void set_Jeven_Jodd(){
    const int p_2 = p_ / 2;
    int odd_offset;
    int left;

    if(p_2 % 2 == 0){
      nJeven_ = p_2 + 1;
      nJodd_ = p_2;
//      odd_offset = 1;
//      left = p_2 / 2;
    }
    else{
      nJeven_ = p_2;
      nJodd_ = p_2 + 1;
//      odd_offset_ = 0;
//      left = (p_2 - 1) / 2;
    }

    Jeven_ = new real[nJeven_];
    Jodd_ = new real[nJodd_];

    if(p_2 % 2 == 0){
      for(int i = 0; i <= p_2; i++){
        Jeven_[i] = J_[2 * i];
      }

      for(int i = 0; i < p_2; i++){
        Jodd_[i] = J_[2 * i + 1];
      }
    }
    else{
      for(int i = 0; i < p_2; i++){
        Jeven_[i] = J_[2 * i + 1];
      }

      for(int i = 0; i <= p_2; i++){
        Jodd_[i] = J_[2 * i];
      }
    }
  }

  void set_grid(int n1x, int n1y, int n1z){
    grid_start_ = new int[lmax_];
    nlx_ = new int[lmax_];
    nly_ = new int[lmax_];
    nlz_ = new int[lmax_];
    nl_ = new int[lmax_];

    nlx_[0] = n1x;
    nly_[0] = n1y;
    nlz_[0] = n1z;

    for(int i = 1; i < lmax_; i++){
      nlx_[i] = nlx_[i - 1] / 2;
      nly_[i] = nly_[i - 1] / 2;
      nlz_[i] = nlz_[i - 1] / 2;
    }

    n_ = 0;

    for(int i = 0; i < lmax_; i++){
      grid_start_[i] = n_;
      nl_[i] = nlx_[i] * nly_[i] * nlz_[i];
      n_ += nl_[i];
    }

    ql_ = new real[n_];
    elplus_ = new real[n_];
  }

public:
  MSMLong(MSMKernel* kernel, int p, int level, int natom, const real* q){
    kernel_ = kernel;
    p_ = p;
    level_ = level;
    natom_ = natom;
    q_ = q;

//TODO
    lmax_ = level_;

//TODO
    // (1 - 2^-lmax) * k1(0) * sum(q^2)
//    self_potential_ = 1.0 - std::exp2(-lmax_);
    self_potential_ = 1.0;
    self_potential_ *= kernel_->get_level_l_potential(1, 0);
    self_potential_ *= std::inner_product(q_, q_ + natom, q_, 0.0);

    p_2_ = p / 2;
    int ngrid = std::exp2(level_); // 2^level

    // Current version only support cubic grid
    set_grid(ngrid, ngrid, ngrid);

    r_ = nullptr;

    J_ = new real[p_ + 1];
    get_J(p_, J_);
    set_Jeven_Jodd();

    value_ = new real[natom_];
  }

  ~MSMLong(){
    if(J_ != nullptr){delete [] J_;}
    if(Jeven_ != nullptr){delete [] Jeven_;}
    if(Jodd_ != nullptr){delete [] Jodd_;}
    if(value_ != nullptr){delete [] value_;}
    if(grid_start_ != nullptr){delete [] grid_start_;}
    if(nlx_ != nullptr){delete [] nlx_;}
    if(nly_ != nullptr){delete [] nly_;}
    if(nlz_ != nullptr){delete [] nlz_;}
    if(nl_ != nullptr){delete [] nl_;}
    if(ql_ != nullptr){delete [] ql_;}
    if(elplus_ != nullptr){delete [] elplus_;}
    if(kernel_ != nullptr){delete kernel_;}
  }

  const MSMKernel* get_kernel() const{
    return kernel_;
  }

  int get_Bspline_order() const{
    return p_;
  }

  int get_level() const{
    return level_;
  }

  int get_atom_number() const{
    return natom_;
  }

  const real* get_charge() const{
    return q_;
  }

  const real* get_coordinate() const{
    return r_;
  }

  const real* get_unit_cell_length() const{
    return L_;
  }

  void get_grid_number_per_axis(int l, int* nlx, int* nly, int* nlz) const{
    *nlx = nlx_[l - 1];
    *nly = nly_[l - 1];
    *nlz = nlz_[l - 1];
  }

  int get_grid_number(int l) const{
    return nl_[l - 1];
  }

  void set_coordinate(const real* r, const real* L){
    r_ = r;
    std::copy_n(L, 3, L_);

    hx_ = L_[0] / nlx_[0];
    hy_ = L_[1] / nly_[0];
    hz_ = L_[2] / nlz_[0];

    kernel_->set_grid_width(hx_, hy_, hz_);
  }

  void get_anterpolation(real* q1){
    const int n1x = nlx_[0];
    const int n1y = nly_[0];
    const int n1z = nlz_[0];
    const real hx_inv = 1.0 / hx_;
    const real hy_inv = 1.0 / hy_;
    const real hz_inv = 1.0 / hz_;
    const int n1yn1z = n1y * n1z;

    std::fill_n(q1, n1x * n1y * n1z, 0);

    real* Cx = new real[p_];
    real* Cy = new real[p_];
    real* Cz = new real[p_];

    int* xindex = new int[n1x + p_ - 1];
    int* yindex = new int[n1y + p_ - 1];
    int* zindex = new int[n1z + p_ - 1];

    Array::get_periodic_index(n1x, - p_ / 2 + 1, n1x + p_ - 1, xindex);
    Array::get_periodic_index(n1y, - p_ / 2 + 1, n1y + p_ - 1, yindex);
    Array::get_periodic_index(n1z, - p_ / 2 + 1, n1z + p_ - 1, zindex);

    for(int i = 0; i < natom_; i++){
      const real* ri = &r_[3 * i];
      real xi = ri[0] * hx_inv;
      real yi = ri[1] * hy_inv;
      real zi = ri[2] * hz_inv;

      int mx0 = std::floor(xi);
      int my0 = std::floor(yi);
      int mz0 = std::floor(zi);

      Bspline::get_reversed_Bspline(p_, xi - mx0, Cx);
      Bspline::get_reversed_Bspline(p_, yi - my0, Cy);
      Bspline::get_reversed_Bspline(p_, zi - mz0, Cz);

      const int* mxindex = &xindex[mx0];
      const int* myindex = &yindex[my0];
      const int* mzindex = &zindex[mz0];

      for(int jx = 0; jx < p_; jx++){
        real qiCx = q_[i] * Cx[jx];
        real* q1x = &q1[mxindex[jx] * n1yn1z];

        for(int jy = 0; jy < p_; jy++){
          real qiCxCy = qiCx * Cy[jy];
          real* q1xy = &q1x[myindex[jy] * n1z];

          for(int jz = 0; jz < p_; jz++){
            q1xy[mzindex[jz]] += qiCxCy * Cz[jz];
          }
        }
      }
    }

    delete [] Cx;
    delete [] Cy;
    delete [] Cz;
    delete [] xindex;
    delete [] yindex;
    delete [] zindex;
  }

  static void get_J(int p, real* J){
    Array::get_binomial_coefficient(p, J);
    real a = std::exp2(1.0 - p); // 2^(1-p)
    Array::multiply(p + 1, J, a);
  }

  // restriction : q^l -> q^(l+1)
  void get_restriction(int l, const real* ql, real* ql1){
    const int p_2 = p_ / 2;
    const int nlx = nlx_[l - 1];
    const int nly = nly_[l - 1];
    const int nlz = nlz_[l - 1];
    const int nl1x = nlx / 2;
    const int nl1y = nly / 2;
    const int nl1z = nlz / 2;
    const real zero = 0;

    // Jz * q^l : z-direction convolution
    const int nnzindex = 2 * (nl1z - 1) + p_ + 1;
    int* nzindex = new int[nnzindex];
    Array::get_periodic_index(nlz, - p_2, nnzindex, nzindex);
    real* qlxy = new real[nnzindex];
    const int nlxnly = nlx * nly;
    const int nlynlz = nly * nlz;
    real* Jzql = new real[nlx * nly * nl1z];

    for(int nx = 0; nx < nlx; nx++){
      const real* qlx = &ql[nx * nlynlz];
      real* Jzqlx = &Jzql[nx * nly];

      for(int ny = 0; ny < nly; ny++){
        Array::copy_index_value(nnzindex, nzindex, &qlx[ny * nlz], qlxy);
        real* Jzqlxy = &Jzqlx[ny];

        for(int mz = 0; mz < nl1z; mz++){
          // xyz-order -> zxy-order
          Jzqlxy[mz * nlxnly]
            = std::inner_product(J_, J_ + (p_ + 1), &qlxy[2 * mz], zero);
        }
      }
    }

    delete [] nzindex;
    delete [] qlxy;


    // Jy * (Jz * q^l)) : y-direction convolution
    const int nnyindex = 2 * (nl1y - 1) + p_ + 1;
    int* nyindex = new int[nnyindex];
    Array::get_periodic_index(nly, - p_2, nnyindex, nyindex);
    real* Jzqlzx = new real[nnyindex];
    const int nl1znlx = nl1z * nlx;
    real* JyJzql = new real[nlx * nl1y * nl1z];

    // zxy-order
    for(int mz = 0; mz < nl1z; mz++){
      const real* Jzqlz = &Jzql[mz * nlxnly];
      real* JyJzqlz = &JyJzql[mz * nlx];

      for(int nx = 0; nx < nlx; nx++){
        Array::copy_index_value(nnyindex, nyindex,
          &Jzqlz[nx * nly], Jzqlzx);
        real* JyJzqlzx = &JyJzqlz[nx];

        for(int my = 0; my < nl1y; my++){
          // zyx-order -> yzx-order
          JyJzqlzx[my * nl1znlx]
            = std::inner_product(J_, J_ + (p_ + 1), &Jzqlzx[2 * my], zero);
        }
      }
    }

    delete [] nyindex;
    delete [] Jzqlzx;

    // Jx * (Jy * (Jz * q^l)) : x-direction convolution
    const int nnxindex = 2 * (nl1x - 1) + p_ + 1;
    int* nxindex = new int[nnxindex];
    Array::get_periodic_index(nlx, - p_2, nnxindex, nxindex);
    real* JyJzqlyz = new real[nnxindex];
    const int nl1ynl1z = nl1y * nl1z;

    // yzx-order
    for(int my = 0; my < nl1y; my++){
      const real* JyJzqly = &JyJzql[my * nl1znlx];
      real* ql1y = &ql1[my * nl1z];

      for(int mz = 0; mz < nl1z; mz++){
        Array::copy_index_value(nnxindex, nxindex,
          &JyJzqly[mz * nlx], JyJzqlyz);
        real* ql1yz = &ql1y[mz];

        for(int mx = 0; mx < nl1x; mx++){
          // yzx-order -> xyz-order
          ql1yz[mx * nl1ynl1z]
            = std::inner_product(J_, J_ + (p_ + 1), &JyJzqlyz[2 * mx], zero);
        }
      }
    }

    delete [] nxindex;
    delete [] JyJzqlyz;

    delete [] Jzql;
    delete [] JyJzql;
  }

  // convolve(K^l, q^l)
  void get_kernel_convolution(int l, const real* ql, real* Klql){
    const int i = l - 1;
    kernel_->get_periodic_convolution(l, nlx_[i], nly_[i], nlz_[i], ql, Klql);
  }

  // convert level l coefficient e^l to level (l-1) coefficient
  void add_prolongation(int l, const real* el, real* el1){
    const int p_2 = p_ / 2;
    int odd_offset;
    int left;

    if(p_2 % 2 == 0){
      odd_offset = 1;
      left = p_2 / 2;
    }
    else{
      odd_offset = 0;
      left = (p_2 - 1) / 2;
    }

    const int nlx = nlx_[l - 1];
    const int nly = nly_[l - 1];
    const int nlz = nlz_[l - 1];
    const int nl1x = 2 * nlx;
    const int nl1y = 2 * nly;
    const int nl1z = 2 * nlz;
    const real zero = 0;

    // Jz * el : z-direction convolution
    const int nnzindex = nlz + p_2;
    int* nzindex = new int[nnzindex];
    Array::get_periodic_index(nlz, - left, nnzindex, nzindex);
    real* elxy = new real[nnzindex];
    const int nlxnly = nlx * nly;
    const int nlynlz = nly * nlz;
    real* Jzel = new real[nlx * nly * nl1z];

    for(int nx = 0; nx < nlx; nx++){
      const real* elx = &el[nx * nlynlz];
      real* Jzelx = &Jzel[nx * nly];

      for(int ny = 0; ny < nly; ny++){
        Array::copy_index_value(nnzindex, nzindex, &elx[ny * nlz], elxy);
        real* Jzelxy = &Jzelx[ny];

        for(int nz = 0; nz < nlz; nz++){
          // xyz-order -> zxy-order
          real* Jzelxyz = &Jzelxy[(2 * nz) * nlxnly];
          const real* elxyz = &elxy[nz];

          Jzelxyz[0] = std::inner_product(
            Jeven_, Jeven_ + nJeven_, elxyz, zero);
          Jzelxyz[nlxnly] = std::inner_product(
            Jodd_, Jodd_ + nJodd_, &elxyz[odd_offset], zero);
        }
      }
    }

    delete [] elxy;
    delete [] nzindex;

    // Jy * (Jz * el) : y-direction convolution
    const int nnyindex = nly + p_2;
    int* nyindex = new int[nnyindex];
    Array::get_periodic_index(nly, - left, nnyindex, nyindex);
    real* Jzelzx = new real[nnyindex];
    const int nl1znlx = nl1z * nlx;
    real* JyJzel = new real[nlx * nl1y * nl1z];

    // zxy-order
    for(int mz = 0; mz < nl1z; mz++){
      const real* Jzelz = &Jzel[mz * nlxnly];
      real* JyJzelz = &JyJzel[mz * nlx];

      for(int nx = 0; nx < nlx; nx++){
        Array::copy_index_value(nnyindex, nyindex, &Jzelz[nx * nly], Jzelzx);
        real* JyJzelzx = &JyJzelz[nx];

        for(int ny = 0; ny < nly; ny++){
          // zxy-order -> yzx-order
          real* JyJzelzxy = &JyJzelzx[(2 * ny) * nl1znlx];
          const real* Jzelzxy = &Jzelzx[ny];

          JyJzelzxy[0] = std::inner_product(
            Jeven_, Jeven_ + nJeven_, Jzelzxy, zero);
          JyJzelzxy[nl1znlx] = std::inner_product(
            Jodd_, Jodd_ + nJodd_, &Jzelzxy[odd_offset], zero);
        }
      }
    }

    delete [] Jzelzx;
    delete [] nyindex;


    // Jx * (Jy * Jz * el) : x-direction convolution
    const int nnxindex = nlx + p_2;
    int* nxindex = new int[nnxindex];
    Array::get_periodic_index(nlx, - left, nnxindex, nxindex);
    real* JyJzelyz = new real[nnxindex];
    const int nl1ynl1z = nl1y * nl1z;

    // yzx-order
    for(int my = 0; my < nl1y; my++){
      const real* JyJzely = &JyJzel[my * nl1znlx];
      real* el1y = &el1[my * nl1z];

      for(int mz = 0; mz < nl1z; mz++){
        Array::copy_index_value(nnxindex, nxindex,
          &JyJzely[mz * nlx], JyJzelyz);
        real* el1yz = &el1y[mz];

        for(int nx = 0; nx < nlx; nx++){
          // yzx-order -> xyz-order
          real* el1yzx =  &el1yz[(2 * nx) * nl1ynl1z];
          real* JyJzelyzx = &JyJzelyz[nx];

          el1yzx[0] += std::inner_product(
            Jeven_, Jeven_ + nJeven_, JyJzelyzx, zero);
          el1yzx[nl1ynl1z] += std::inner_product(
            Jodd_, Jodd_ + nJodd_, &JyJzelyzx[odd_offset], zero);
        }
      }
    }

    delete [] nxindex;
    delete [] JyJzelyz;

    delete [] Jzel;
    delete [] JyJzel;
  }

  void set_anterpolation(){
    log_.start("anterpolate");
    get_anterpolation(ql_);
    log_.stop();
  }

  void set_restriction(){
    for(int l = 1; l < lmax_; l++){
      log_.start("restrict:" + std::to_string(l)
        + "->" + std::to_string(l + 1));
      get_restriction(l, &ql_[grid_start_[l - 1]], &ql_[grid_start_[l]]);
      log_.stop();
    }
  }

  void set_kernel_convolution(){
    for(int l = 1; l <= lmax_; l++){
      log_.start("kernel_convolution:" + std::to_string(l));
      get_kernel_convolution(l, &ql_[grid_start_[l - 1]],
        &elplus_[grid_start_[l - 1]]);
      log_.stop();
    }
  }

  void set_prolongation(){
    for(int l = lmax_; l >= 2; l--){
      log_.start("prolongate:" + std::to_string(l)
        + "->" + std::to_string(l - 1));
      add_prolongation(l, &elplus_[grid_start_[l - 1]],
        &elplus_[grid_start_[l - 2]]);
      log_.stop();
    }
  }

  void set_grid_coefficient(){
    set_anterpolation();
    set_restriction();
    set_kernel_convolution();
    set_prolongation();
  }

  const real* get_grid_charge(int l) const{
    return &ql_[grid_start_[l - 1]];
  }

  const real* get_grid_coefficient(int l) const{
    return &elplus_[grid_start_[l - 1]];
  }

  void get_interpolation_value_gradient(int l, const real* el,
    int ntarget, const real* rtarget,
    real* value, real* gradient) const{
    const int nlx = nlx_[l - 1];
    const int nly = nly_[l - 1];
    const int nlz = nlz_[l - 1];
//TODO
    const real hlx = hx_ * std::exp2(l - 1);
    const real hly = hy_ * std::exp2(l - 1);
    const real hlz = hz_ * std::exp2(l - 1);

    Bspline::get_periodic_interpolation3d_value_gradient(p_,
      nlx, nly, nlz, hlx, hly, hlz, el, ntarget, rtarget, value, gradient);
  }

  void get_interpolation_value_gradient(int ntarget, const real* rtarget,
    real* value, real* gradient){
    Bspline::get_periodic_interpolation3d_value_gradient(p_,
      nlx_[0], nly_[0], nlz_[0], hx_, hy_, hz_, elplus_,
      ntarget, rtarget, value, gradient);
  }

  // return potential
  real get_potential_force(real* force){
    log_.start("interpolate");

    real potential = 0;

    get_interpolation_value_gradient(natom_, r_, value_, force);

    for(int i = 0; i < natom_; i++){
      real* fi = &force[3 * i];

      potential += q_[i] * value_[i];
      fi[0] *= - q_[i];
      fi[1] *= - q_[i];
      fi[2] *= - q_[i];
    }

    potential *= 0.5;
    potential -= self_potential_;

    log_.stop();

    return potential;
  }

  void print_parameter(std::ostream& o) const{
    kernel_->print_parameter(o);

    o << "msm_long parameter:" << std::endl;
    o << "p " << p_ << std::endl;
    o << "level " << get_level() << std::endl;
    o << "grid_width " << hx_ << " " << hy_ << " " << hz_ << std::endl;

    for(int i = 0; i < lmax_; i++){
      o << "l=" << i + 1
        << " nlx " << nlx_[i]
        << " nly " << nly_[i]
        << " nlz " << nlz_[i]
        << " nl " << nl_[i]
        << " grid_start " << grid_start_[i]
        << std::endl;
    }
  }

};

////////////////////////////////////////////////////////////////////////////////
class TestMSM{
public:
  // -neighbor <= (x/Lx,y/Ly,z/Lz) < neighbor + 1 cells are computed
  static real get_level_l_potential_neighbor(const MSMKernel* kernel, int l,
    int nsource, const real* qsource, const real* rsource, const real* L,
    int neighbor, const real* rtarget){
    real* sum = new real[nsource];
    std::fill_n(sum, nsource, 0);

    #pragma omp parallel for
    for(int i = 0; i < nsource; i++){
      const real* ri = &rsource[3 * i];
      real dx0 = rtarget[0] - ri[0];
      real dy0 = rtarget[1] - ri[1];
      real dz0 = rtarget[2] - ri[2];
      real sumi = 0;

      for(int jx = -neighbor; jx <= neighbor; jx++){
        real dx = dx0 + jx * L[0];
        real dx2 = dx * dx;

        for(int jy = -neighbor; jy <= neighbor; jy++){
          real dy = dy0 + jy * L[1];
          real dy2 = dy * dy;

          for(int jz = -neighbor; jz <= neighbor; jz++){
            real dz = dz0 + jz * L[2];
            real dz2 = dz * dz;

            real r = std::sqrt(dx2 + dy2 + dz2);
            sumi += kernel->get_level_l_potential(l, r);
          }
        }
      }

      sum[i] = qsource[i] * sumi;
    }

    real potential = Array::sum(nsource, sum);

    delete [] sum;

    return potential;
  }

  static void get_direct_potential_neighbor(const MSMKernel* kernel, int l,  
    int nsource, const real* qsource, const real* rsource, const real* L,
    int neighbor, int ntarget, const real* target, real* potential){
    for(int i = 0; i < ntarget; i++){
      potential[i] = get_level_l_potential_neighbor(kernel,
        l, nsource, qsource, rsource, L, neighbor, &target[3 * i]);
    }
  }

  static void get_random_coordinate(const real* L, int n, real* r){
//    std::random_device rd;
//    std::mt19937 mt(rd());
    std::mt19937 mt;
    std::uniform_real_distribution<real> u(0.0, 1.0);

    for(int i = 0; i < n; i++){
      real* ri = &r[3 * i];

      ri[0] = L[0] * u(mt);
      ri[1] = L[1] * u(mt);
      ri[2] = L[2] * u(mt);
    }
  }

  static void print_two_value(
    const std::string& xlabel, const std::string & ylabel,
    int n, const real* x, const real* y,
    std::ostream& o){
//TODO
    const int width = 15;

    o << std::left << std::setw(width) << "x=" + xlabel
      << std::left << std::setw(width) << "y=" + ylabel
      << std::left << std::setw(width) << "y-x"
      << std::left << std::setw(width) << "(y-x)/x"
      << std::endl;

    for(int i = 0; i < n; i++){
      o << std::left << std::setw(width) << x[i]
        << std::left << std::setw(width) << y[i]
        << std::left << std::setw(width) << y[i] - x[i]
        << std::left << std::setw(width) << (y[i] - x[i]) / x[i]
        << std::endl;
    }
  }

  static void test_msm_long_periodic(MSMLong* msm,
    int ntarget, const real* rtarget, int neighbor,
    EwaldLong* ewald,  std::ostream& o){
    msm->set_anterpolation();
    msm->set_restriction();

//TODO
const int lmax = msm->get_level();
    Array::print_separator(o);

    for(int l = 1; l <= lmax; l++){
      const real* ql = msm->get_grid_charge(l);
      const int nl = msm->get_grid_number(l);

      o << "sum(q" << l << ") " << Array::sum(nl, ql) << std::endl;
    }

    Array::print_separator(o);
    msm->set_kernel_convolution();

    real* msm_value = new real[ntarget];
    real* msm_gradient = new real[3 * ntarget];
    real* direct_value = new real[lmax * ntarget];

    const MSMKernel* kernel = msm->get_kernel();
    const int natom = msm->get_atom_number();
    const real* q = msm->get_charge();
    const real* r = msm->get_coordinate();
    const real* L = msm->get_unit_cell_length();

    for(int l = 1; l <= lmax; l++){
      Array::print_separator(o);
      const real* el = msm->get_grid_coefficient(l);

      msm->get_interpolation_value_gradient(l, el,
        ntarget, rtarget, msm_value, msm_gradient);

      o << "l=" << l << std::endl;
      TimeLogger log_direct;
      log_direct.start("direct");

      real* direct_valuel = &direct_value[(l - 1) * ntarget];
      get_direct_potential_neighbor(kernel, l,
        natom, q, r, L, neighbor, ntarget, rtarget, direct_valuel);

      log_direct.stop();

      o << std::endl;
      o << "value" << std::endl;
      print_two_value("direct", "msm", ntarget, direct_valuel, msm_value, o);
    }

    Array::print_separator(o);
    msm->set_prolongation();

    for(int l = lmax - 1; l >= 1; l--){
      Array::print_separator(o);
      o << "l=" << l << "~" << lmax << std::endl;
      const real* elplus = msm->get_grid_coefficient(l);

      msm->get_interpolation_value_gradient(l, elplus,
        ntarget, rtarget, msm_value, msm_gradient);

      const real* direct_valuel1 = &direct_value[l * ntarget]; // l+1
      real* direct_valuel = &direct_value[(l - 1) * ntarget]; // l

      // Accumulate direct value and gradient
      for(int i = 0; i < ntarget; i++){
        direct_valuel[i] += direct_valuel1[i];
      }

      o << std::endl;
      o << "value" << std::endl;
      print_two_value("direct", "msm", ntarget, direct_valuel, msm_value, o);
    }


    Array::print_separator(o);
    msm->get_interpolation_value_gradient(ntarget, rtarget,
      msm_value, msm_gradient);

    if(ewald != nullptr){
      o << "ewald long" << std::endl;
      ewald->print_parameter(o);

      ewald->set_coordinate(r, L);
      ewald->set_structure_factor();

      real* ewald_value = new real[ntarget];
      real* ewald_gradient = new real[3 * ntarget];

      ewald->get_value_gradient(ntarget, rtarget, ewald_value, ewald_gradient);

      o << std::endl;
      o << "value" << std::endl;
      print_two_value("ewald", "msm", ntarget, ewald_value, msm_value, o);

      o << std::endl;
      o << "gradient" << std::endl;
      print_two_value("ewald", "msm",
        3 * ntarget, ewald_gradient, msm_gradient, o);

      delete [] ewald_value;
      delete [] ewald_gradient;
    }

    delete [] msm_value;
    delete [] msm_gradient;
  }

};

#endif // MSM_LONG_H_
