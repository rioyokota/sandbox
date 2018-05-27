#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>

// CUBLAS GEMM API
#include <cublas_v2.h>

// Set Cutlass debug macro to enable console printing of library errors
#define DEBUG

#if defined(WMMA)
// Conditionally include WMMA headers (CUDA 9 Preview Feature)
#include <mma.h>
#endif

// Cutlass GEMM API
#include <util/util.h>
#include <gemm/dispatch.h>
#include <gemm/epilogue_function.h>

// Test utilities
#include "util/command_line.h"
#include "util/half.h"
#include "util/matrix.h"
#include "util/timer.h"
#include "util/type_conversion.h"

// Dispatch routines to CUBLAS and CUTLASS
#include "cublas_dispatch.h"
#include "cutlass_dispatch.h"

using namespace cutlass;

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 4096;
  float alpha = 1.0;
  float beta = 0.0;
  const math_operation_class_t math_op = math_operation_class_t::scalar;
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
  cublas_gemm<gemm::tiling_strategy::Unknown, math_op, TransformA, TransformB, value_t, accum_t> cublas;
  cutlass_gemm_dispatch<gemm::tiling_strategy::Large, math_op, TransformA, TransformB, value_t, accum_t> cutlass;
  cublasHandle_t g_cublas_handle;
  cublasCreate(&g_cublas_handle);
  gpu_timer timer;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    CUDA_PERROR(cublas(
                       g_cublas_handle,
                       m,
                       n,
                       k,
                       A.d_data(),
                       B.d_data(),
                       C.d_data(),
                       alpha,
                       beta,
                       stream,
                       false).result);
  }
  timer.stop();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = timer.elapsed_millis() / g_timing_iterations;
  double cublas_flops = double(num_flops) / tcublas / 1.0e6;
  for (int i = 0; i < g_timing_iterations+2; i++) {
    if (i == 2) timer.start();
    CUDA_PERROR(cutlass(
                        g_cublas_handle,
                        m,
                        n,
                        k,
                        A.d_data(),
                        B.d_data(),
                        C2.d_data(),
                        alpha,
                        beta,
                        stream,
                        false).result);
  }
  timer.stop();
  double tcutlass = timer.elapsed_millis() / g_timing_iterations;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e6;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  C.sync_host();
  C2.sync_host();
  double diff = 0, norm = 0;
  for (int i=0; i<n; i++) {
    printf("%d %f %f\n",i,C.get(i,0), C2.get(i,0));
    for (int j=0; j<m; j++) {
      diff += (C.get(i,j) - C2.get(i,j)) * (C.get(i,j) - C2.get(i,j));
      norm += C.get(i,j) * C.get(i,j);
    }
  }
  printf("L2 Error: %.2lf\n", sqrtf(diff/norm));
  cublasDestroy(g_cublas_handle);
}
