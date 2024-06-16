#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <chrono>
using namespace std;

int main(int argc, const char **argv) {
  int n = 1 << 20;
  int Nt = 10;
  float alpha = 1.0;
  float *x, *y;
  cudaMallocManaged(&x, n * sizeof(float));
  cudaMallocManaged(&y, n * sizeof(float));
  for (int i=0; i<n; i++)
    x[i] = drand48();
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasSaxpy(cublas_handle,
		n,
		&alpha,
		x,
		1,
		y,
		1);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = 2 * int64_t(n);
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  printf("Saxpy: %.2f GFlops\n", cublas_flops);
  cudaFree(x);
  cudaFree(y);
  cublasDestroy(cublas_handle);
}
