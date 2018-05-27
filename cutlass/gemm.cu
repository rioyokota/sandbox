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

/******************************************************************************
 * Globals, constants and typedefs
 ******************************************************************************/

using namespace cutlass;

/// CUBLAS handle
cublasHandle_t g_cublas_handle;

/// The device-id of the current device
int g_device_id = -1;

/// The number of timing iterations to invoke
int g_timing_iterations = -1;

/// The number of randomly-sized problems to schmoo
int g_schmoo = 0;


/******************************************************************************
 * Number generation
 ******************************************************************************/

/**
 * Simple low-integer generator
 */
struct simple_gen
{
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;

    /// Constructor
    simple_gen(int max) : distribution(max * -1, max)
    {}

    /// Functor
    int operator()()
    {
        return distribution(generator);
    }
};




/******************************************************************************
 * Test execution
 ******************************************************************************/


/**
 * Compute C = (alpha * A * B) + (beta * C)
 */
template <
    typename                    test_func_t,    ///< Test function type
    matrix_transform_t::kind_t  TransformA,     ///< Transformation op for matrix A
    matrix_transform_t::kind_t  TransformB,     ///< Transformation op for matrix B
    typename                    value_t,        ///< Multiplicand value type (matrices A and B)
    typename                    accum_t>        ///< Accumulator value type (matrix C and scalars)
bool test(
    int m,          ///< Height of C in rows
    int n,          ///< Width of C in columns
    int k,          ///< Width (height) of A (B)
    accum_t alpha,  ///< Multiplicand scalar
    accum_t beta)   ///< Addend scalar
{
    cudaStream_t stream = 0;

    //
    // Initialize matrices
    //

    matrix<value_t> A(
        (TransformA == matrix_transform_t::NonTranspose) ? m : k,
        (TransformA == matrix_transform_t::NonTranspose) ? k : m);

    matrix<value_t> B(
        (TransformB == matrix_transform_t::NonTranspose) ? k : n,
        (TransformB == matrix_transform_t::NonTranspose) ? n : k);

    matrix<accum_t> C(m, n);

    // initialized matrices with small values precisely representable as integers
    simple_gen a_gen(3);
    simple_gen b_gen(5);
    A.fill_random(a_gen);
    B.fill_random(b_gen);
    C.fill_ramp(0,0);

//    // Alternatively, initialize with procedural values to simplify debugging incorrect results
//    A.fill_ramp(1,2);
//    B.fill_ramp(1,1);

    // Sync to device
    A.sync_device();
    B.sync_device();
    C.sync_device();

    CUDA_PERROR(cudaPeekAtLastError());
    CUDA_PERROR(cudaDeviceSynchronize());

    //
    // Run test once with debug-synchronous enabled and check result
    //

    if (!g_schmoo) printf("\n");

    test_func_t test_func;

    C.fill_ramp(0, 0);
    C.sync_device();

    cudaError_t error = test_func(
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
        !g_schmoo).result;

    bool not_applicable = (error == cudaErrorInvalidValue);
    bool is_failed = false;
    if (not_applicable)
    {
        printf(", NA");
    }
    else
    {
        CUDA_PERROR(error);

        // Compute reference check if wont take too long on CPU
        if ((!g_schmoo) && (m * n <= 1024 * 1024))
        {
            matrix<accum_t> ref_C(m, n);
            ref_C.fill_ramp(0, 0);
            ref_C.gemm(TransformA, TransformB, alpha, A, B, beta);
            C.sync_host();

            is_failed = (C != ref_C);

            if (!g_schmoo)
            {
                if (is_failed)
                {
                    printf("FAIL, ");
                    std::ofstream file_a("a.csv");
                    A.write_matrix(file_a);
                    std::ofstream file_b("b.csv");
                    B.write_matrix(file_b);
                    std::ofstream file_d("gemm-REF.csv");
                    ref_C.write_matrix(file_d);
                    std::ofstream file_c("gemm-GPU.csv");
                    C.write_matrix(file_c);
                }
                else
                {
                    printf("PASS, ");
                }
            }
        }
        fflush(stdout);

        //
        // Warmup and timing iterations
        //

        if (g_timing_iterations > 0)
        {
            // Warmup for 1/100 of the timing iterations (minimum of 2)
            for (int i = 0; i < __NV_STD_MAX(2, (g_timing_iterations + 99) / 100); ++i)
            {
                CUDA_PERROR(test_func(
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
        }

        // Conduct timing iterations
        double elapsed_ms = 0;
        gpu_timer timer;
        timer.start();

        for (int i = 0; i < g_timing_iterations; i++)
        {
            CUDA_PERROR(test_func(
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
        elapsed_ms += timer.elapsed_millis();
        double avg_ms = elapsed_ms / g_timing_iterations;

        // Display performance
        if (g_timing_iterations > 0)
        {
            int64_t num_flops      = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
            double gflops_per_sec   = double(num_flops) / avg_ms / 1.0e6;

            if (g_schmoo)
            {
                if (is_failed)
                    printf("F");

                printf(", %.3f", gflops_per_sec);

                // Sleep for a few milliseconds to cool
                sleep_millis(10);
            }
            else
            {
                printf("Avg runtime: %.3f ms, total flops: %ld, GFLOP/s: %.2f\n",
                    avg_ms,
                    num_flops,
                    gflops_per_sec);
            }
            fflush(stdout);
        }
    }

    return is_failed;
}

/**
 * Compute C = (alpha * A * B) + (beta * C)
 */
bool test(int m, int n, int k, float alpha, float beta) {
  const math_operation_class_t math_op = math_operation_class_t::scalar;
  static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  typedef float value_t;
  typedef float accum_t;
  uint64_t flop_base = 1ull << 41;
  int max_timing_iterations = 10000;
  int min_timing_iterations = 10;
  bool test_error = false;
  uint64_t num_flops = (2 * uint64_t(m) * uint64_t(n) * uint64_t(k)) + (2 * uint64_t(m) * uint64_t(n));
  g_timing_iterations = (int) ((flop_base / sizeof(value_t)) / num_flops);
  g_timing_iterations = (int) __NV_STD_MIN(max_timing_iterations, g_timing_iterations);
  g_timing_iterations = (int) __NV_STD_MAX(min_timing_iterations, g_timing_iterations);

  printf("\n------------------------------------------------------------\n");
  printf("%dx%dx%d, GEMM_%c%c, %d C elements, %d timing iterations\n",
         m, n, k,
         (TransformA == matrix_transform_t::NonTranspose) ? 'n' : 't',
         (TransformB == matrix_transform_t::NonTranspose) ? 'n' : 't',
         m * n,
         g_timing_iterations);
  fflush(stdout);

  // CUBLAS
  test_error |= test<
    cublas_gemm<gemm::tiling_strategy::Unknown, math_op, TransformA, TransformB, value_t, accum_t>,
      TransformA,
      TransformB,
      value_t,
      accum_t>(m, n, k, accum_t(alpha), accum_t(beta));

  // CUTLASS
  test_error |= test<
    cutlass_gemm_dispatch<gemm::tiling_strategy::Large, math_op, TransformA, TransformB, value_t, accum_t>,
      TransformA,
      TransformB,
      value_t,
      accum_t>(m, n, k, accum_t(alpha), accum_t(beta));

    return test_error;
}

int main(int argc, const char **argv) {
    int m = 10240;
    int k = 4096;
    int n = 4096;
    float alpha = 1.0;
    float beta = 0.0;
    g_device_id = 0;
    cublasCreate(&g_cublas_handle);
    bool test_error = false;
    test_error |= test(m, n, k, alpha, beta);
    cublasDestroy(g_cublas_handle);
}
