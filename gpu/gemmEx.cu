#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <chrono>
using namespace std;

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasSgemmEx(cublas_handle,
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		m,
		n,
		k,
		&alpha,
		A,
		CUDA_R_16F,
		m,
		B,
		CUDA_R_16F,
		k,
		&beta,
		C,
		CUDA_R_32F,
		m);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e12;
  printf("CUBLAS: %.2f TFlops\n", cublas_flops);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cublasDestroy(cublas_handle);
}
