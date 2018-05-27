#pragma once
#include <stdint.h>
#include "../util/util.h"
#include "block_task.h"
#include "grid_raster.h"
#include "k_split_control.h"

namespace cutlass {
namespace gemm {

  template <typename epilogue_op_t>
  __global__ void kernel(
                       int m,                      ///< Height in rows of op(A) and C
                       int n,                      ///< Width in columns of op(B) and C
                       int k,                      ///< Width in columns of op(A) and height in rows of op(B)
                       k_split_control k_split,    ///< Abstraction for controlling inter-block k-splitting
                       epilogue_op_t op,           ///< Epilogue operation to update matrix C
                       float *d_a,               ///< Pointer to matrix A array values
                       float *d_b,               ///< Pointer to matrix B array values
                       float *d_c)               ///< Pointer to matrix C array values
{
  typedef block_task<
    float,
    float,
    16,
    16,
    epilogue_op_t,
    4,
    false> block_task_t;

    // Declare statically-allocated shared storage
    __shared__ typename block_task_t::scratch_storage_t smem;

    // Construct and run the task
    block_task_t(
        &smem,
        d_a,
        d_b,
        d_c,
        op,
        m,
        n,
        k,
        k_split).run();
}


/******************************************************************************
 * Launch configuration description returned to the caller
 ******************************************************************************/

/// Return details about the launch configuration to the caller
struct launch_configuration
{
    //
    // Data members
    //

    /// cudaError_t resulting from grid launch
    cudaError_t result;

    /// Extent of a thread block's partition along the GEMM K-axis
    int split_k;

    /// Kernel grid extents in thread blocks
    dim3 grid;

    /// Thread block extents in threads
    dim3 block;

    //
    // Methods
    //

    /// Constructor
    launch_configuration():
        result(cudaSuccess),
        split_k(0),
        grid(0, 0, 0),
        block(0, 0, 0) {

    }

    /// Conversion from cudaError_t
    launch_configuration(cudaError_t result):
        result(result),
        split_k(1),
        grid(0, 0, 0),
        block(0, 0, 0) {

    }

    /// Launch configuration for Cutlass kernels
    launch_configuration(
        cudaError_t result,
        int split_k,
        dim3 grid,
        dim3 block
    ):
        result(result),
        split_k(split_k),
        grid(grid),
        block(block) {

    }
};


/******************************************************************************
 * Dispatch stub
 ******************************************************************************/

/**
 * GEMM dispatch stub
 *
 * This function also serves as the autotuning entrypoint to evaluate different
 * tuning parameterizations of kernel.
 */
template <typename epilogue_op_t>
launch_configuration dispatch(
    int             m,                              ///< Height in rows of op(A) and C
    int             n,                              ///< Width in columns of op(B) and C
    int             k,                              ///< Width in columns of op(A) and height in rows of op(B)
    float           alpha,
    float           beta,
    float         *d_a,                           ///< Device pointer to matrix A array values
    float         *d_b,                           ///< Device pointer to matrix B array values
    float         *d_c,                           ///< Device pointer to matrix C array values
    cudaStream_t    stream = 0,                     ///< CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool            debug_synchronous = true)       ///< Whether or not to synchronize the stream after every kernel launch
                                                    ///  to check for errors.  Also causes launch configurations to be printed
                                                    ///  to the console if DEBUG is defined.  Default is \p false.
{
    // Thread block rasterization type
  static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  epilogue_op_t epilogue(alpha, beta);
  typedef grid_raster<
    64,
    64,
    TransformA,
    TransformB,
    grid_raster_strategy::Default>
    grid_raster_t;
  launch_configuration config;
  config.block = dim3(64);
  int dynamic_smem_bytes = 0;
  int max_sm_occupancy = 8;
  config.grid = grid_raster_t::grid_dims(m, n);
  int sm_count;
  get_sm_count(sm_count);
  int *d_flags;
  cudaGetSymbolAddress((void**) &d_flags, d_flags_split_k);

  k_split_control k_split(
                          d_flags,
                          sm_count,
                          max_sm_occupancy,
                          k,
                          8,
                          config.block,
                          config.grid);
  config.split_k = k_split.split_k;
  gemm::kernel<epilogue_op_t>
    <<< config.grid,
    config.block,
    dynamic_smem_bytes,
    stream >>>(
               m,
               n,
               k,
               k_split,
               epilogue,
               d_a,
               d_b,
               d_c);
  return config;
}


} // namespace gemm
} // namespace cutlass
