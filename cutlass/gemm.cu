#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#define DEBUG

#include <gemm/dispatch.h>
#include <gemm/epilogue_function.h>
#include "util/matrix.h"
#include "util/timer.h"

using namespace cutlass;

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 4096;
  float alpha = 1.0;
  float beta = 0.0;
  static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  typedef float value_t;
  typedef float accum_t;
  int g_timing_iterations = 10;
  cudaStream_t stream = 0;
  matrix<value_t> A(m, k);
  matrix<value_t> B(k, n);
  matrix<accum_t> C(m, n);
  matrix<accum_t> C2(m, n);
  A.random();
  B.random();
  C.fill_ramp(0,0);
  C2.fill_ramp(0,0);
  A.sync_device();
  B.sync_device();
  C.sync_device();
  C2.sync_device();
  cublasHandle_t g_cublas_handle;
  cublasCreate(&g_cublas_handle);
  gpu_timer timer;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    CUDA_PERROR(cublasSgemm(
                            g_cublas_handle,
                            (cublasOperation_t) TransformA,
                            (cublasOperation_t) TransformB,
                            m,
                            n,
                            k,
                            &alpha,
                            A.d_data(),
                            m,
                            B.d_data(),
                            k,
                            &beta,
                            C.d_data(),
                            m));
  }
  timer.stop();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = timer.elapsed_millis() / g_timing_iterations;
  double cublas_flops = double(num_flops) / tcublas / 1.0e6;
  typedef gemm::blas_scaled_epilogue<float, float, float> epilogue_op_t;
  epilogue_op_t epilogue(alpha, beta);
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    gemm::dispatch<epilogue_op_t>(
        m,
        n,
        k,
        alpha,
        beta,
        A.d_data(),
        B.d_data(),
        C2.d_data(),
        stream,
        false);
  }
  timer.stop();
  double tcutlass = timer.elapsed_millis() / g_timing_iterations;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e6;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  C.sync_host();
  C2.sync_host();
  double diff = 0, norm = 0;
  for (int i=0; i<n; i++) {
    //printf("%d %f %f\n",i,C.get(i,0), C2.get(i,0));
    for (int j=0; j<m; j++) {
      diff += (C.get(i,j) - C2.get(i,j)) * (C.get(i,j) - C2.get(i,j));
      norm += C.get(i,j) * C.get(i,j);
    }
  }
  printf("L2 Error: %lf\n", sqrtf(diff/norm));
  cublasDestroy(g_cublas_handle);
}
