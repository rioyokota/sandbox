/******************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

/**
 * \file
 * GEMM kernel entrypoint and dispatch stub
 */

#include <stdint.h>

#include "../util/util.h"
#include "block_task.h"
#include "block_task_wmma.h"
#include "grid_raster.h"
#include "dispatch_policies.h"
#include "k_split_control.h"

namespace cutlass {
namespace gemm {


/******************************************************************************
 * param_pack
 ******************************************************************************/

/**
 * Parameter-pack structure
 *
 * Kernel launch latency is reduced when kernel arguments are wrapped into
 * a single parameter
 */
template <
    typename value_t,
    typename accum_t,
    typename epilogue_op_t>
struct param_pack
{
    int m;                      ///< Height in rows of op(A) and C
    int n;                      ///< Width in columns of op(B) and C
    int k;                      ///< Width in columns of op(A) and height in rows of op(B)
    k_split_control k_split;    ///< Abstraction for controlling inter-block k-splitting
    value_t *d_a;               ///< Pointer to matrix A array values
    value_t *d_b;               ///< Pointer to matrix B array values
    accum_t *d_c;               ///< Pointer to matrix C array values
    epilogue_op_t epilogue_op;

    param_pack(
        int m,                      ///< Height in rows of op(A) and C
        int n,                      ///< Width in columns of op(B) and C
        int k,                      ///< Width in columns of op(A) and height in rows of op(B)
        k_split_control k_split,    ///< Abstraction for controlling inter-block k-splitting
        epilogue_op_t op,           ///< Epilogue operation to update matrix C
        value_t *d_a,               ///< Pointer to matrix A array values
        value_t *d_b,               ///< Pointer to matrix B array values
        accum_t *d_c)               ///< Pointer to matrix C array values
    :
        m(m),
        n(n),
        k(k),
        k_split(k_split),
        epilogue_op(op),
        d_a(d_a),
        d_b(d_b),
        d_c(d_c)
    {}

};


/******************************************************************************
 * Conditionally select the appropriate GEMM threadblock task
 ******************************************************************************/

/// Conditional selection for block task
template <
    math_operation_class_t      math_op,            ///<
    typename                    block_task_policy_t,  ///< Parameterization of block_task_policy
    typename                    value_t,            ///< Multiplicand value type (matrices A and B)
    typename                    accum_t,            ///< Accumulator value type (matrix C and scalars)
    matrix_transform_t::kind_t  TransformA,         ///< View transform enumerant for matrix A
    int                         LdgAlignA,          ///< Alignment (in bytes) for A operand
    matrix_transform_t::kind_t  TransformB,         ///< View transform enumerant for matrix B
    int                         LdgAlignB,          ///< Alignment (in bytes) for B operand
    typename                    epilogue_op_t,      ///< Epilogue operation applied to GEMM
    int                         LdgAlignC,          ///< Alignment (in bytes) for C operand
    bool                        AllowRaggedTiles    ///< Whether GEMM supports matrix sizes other than multiple of BlockItems{XY}
>
struct gemm_block_task;

/// Scalar math operations
template <
    typename                    block_task_policy_t,  ///< Parameterization of block_task_policy
    typename                    value_t,            ///< Multiplicand value type (matrices A and B)
    typename                    accum_t,            ///< Accumulator value type (matrix C and scalars)
    matrix_transform_t::kind_t  TransformA,         ///< View transform enumerant for matrix A
    int                         LdgAlignA,          ///< Alignment (in bytes) for A operand
    matrix_transform_t::kind_t  TransformB,         ///< View transform enumerant for matrix B
    int                         LdgAlignB,          ///< Alignment (in bytes) for B operand
    typename                    epilogue_op_t,      ///< Epilogue operation applied to GEMM
    int                         LdgAlignC,          ///< Alignment (in bytes) for C operand
    bool                        AllowRaggedTiles    ///< Whether GEMM supports matrix sizes other than multiple of BlockItems{XY}
>
struct gemm_block_task<
    math_operation_class_t::scalar,
    block_task_policy_t,
    value_t,
    accum_t,
    TransformA,
    LdgAlignA,
    TransformB,
    LdgAlignB,
    epilogue_op_t,
    LdgAlignC,
    AllowRaggedTiles
>
{
    // Parameterize task type
    typedef block_task<
            block_task_policy_t,
            value_t,
            accum_t,
            TransformA,
            LdgAlignA,
            TransformB,
            LdgAlignB,
            epilogue_op_t,
            LdgAlignC,
            AllowRaggedTiles> type;
};

/******************************************************************************
 * GEMM kernel entrypoint
 ******************************************************************************/

/**
 * GEMM kernel
 *
 * NB: Not sure why NVVM is doing stuff with "__launch_bounds__" instead of just
 * passing it along to PTXAS, but it is currently resulting in less optimal codegen
 */
template <
    matrix_transform_t::kind_t  TransformA,         ///< Transformation op for matrix A
    int                         LdgAlignA,          ///< Alignment of A matrix elements in bytes
    matrix_transform_t::kind_t  TransformB,         ///< Transformation op for matrix B
    int                         LdgAlignB,          ///< Alignment of B matrix elements in bytes
    typename                    value_t,            ///< Multiplicand value type (matrices A and B)
    typename                    accum_t,            ///< Accumulator value type (matrix C and scalars)
    typename                    epilogue_op_t,      ///< Epilogue operation applied to update matrix C
    int                         LdgAlignC,          ///< Alignment of C elements in bytes
    bool                        AllowRaggedTiles>   ///< Boolean to indicate whether AllowRaggedTiles handling is enabled
__global__ void kernel(param_pack<value_t, accum_t, epilogue_op_t> pack)
{
    // Parameterize task type
  typedef gemm::gemm_policy<value_t, accum_t, TransformA, TransformB, gemm::tiling_strategy::Large> block_task_policy_t;
    typedef block_task<
        block_task_policy_t,
        value_t,
        accum_t,
        TransformA,
        LdgAlignA,
        TransformB,
        LdgAlignB,
        epilogue_op_t,
        LdgAlignC,
        AllowRaggedTiles> block_task_t;

    // Declare statically-allocated shared storage
    __shared__ typename block_task_t::scratch_storage_t smem;

    // Construct and run the task
    block_task_t(
        &smem,
        pack.d_a,
        pack.d_b,
        pack.d_c,
        pack.epilogue_op,
        pack.m,
        pack.n,
        pack.k,
        pack.k_split).run();
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
template <
    typename                    value_t,            ///< Multiplicand value type (matrices A and B)
    typename                    accum_t,            ///< Accumulator value type (matrix C and scalars)
    typename                    epilogue_op_t,      ///< Epilogue operation
    int                         LdgAlignC,          ///< Alignment of C matrix elements in bytes
    bool                        AllowRaggedTiles,   ///< Boolean to indicate whether AllowRaggedTiles handling is enabled
    typename                    kernel_ptr_t>       ///< GEMM kernel function pointer type
launch_configuration dispatch(
    kernel_ptr_t    kernel_ptr,                     ///< GEMM kernel function pointer
    int             m,                              ///< Height in rows of op(A) and C
    int             n,                              ///< Width in columns of op(B) and C
    int             k,                              ///< Width in columns of op(A) and height in rows of op(B)
    epilogue_op_t   epilogue_op,                    ///< Epilogue operation to update matrix C
    value_t         *d_a,                           ///< Device pointer to matrix A array values
    value_t         *d_b,                           ///< Device pointer to matrix B array values
    accum_t         *d_c,                           ///< Device pointer to matrix C array values
    cudaStream_t    stream = 0,                     ///< CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
    bool            debug_synchronous = true)       ///< Whether or not to synchronize the stream after every kernel launch
                                                    ///  to check for errors.  Also causes launch configurations to be printed
                                                    ///  to the console if DEBUG is defined.  Default is \p false.
{
    // Thread block rasterization type
  static const matrix_transform_t::kind_t TransformA = matrix_transform_t::NonTranspose;
  static const matrix_transform_t::kind_t TransformB = matrix_transform_t::NonTranspose;
  typedef gemm::gemm_policy<value_t, accum_t, TransformA, TransformB, gemm::tiling_strategy::Large> block_task_policy_t;
  typedef grid_raster<
    block_task_policy_t::BlockItemsY,
    block_task_policy_t::BlockItemsX,
    TransformA,
    TransformB,
    block_task_policy_t::RasterStrategy>
    grid_raster_t;
  launch_configuration config;
  config.block = dim3(block_task_policy_t::BlockThreads);
  int dynamic_smem_bytes = 0;
  int max_sm_occupancy;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                                &max_sm_occupancy,
                                                kernel_ptr,
                                                config.block.x * config.block.y,
                                                dynamic_smem_bytes);
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
                          block_task_policy_t::BlockItemsK,
                          config.block,
                          config.grid);
  config.split_k = k_split.split_k;

  param_pack<value_t, accum_t, epilogue_op_t> pack(
                                                   m,
                                                   n,
                                                   k,
                                                   k_split,
                                                   epilogue_op,
                                                   d_a,
                                                   d_b,
                                                   d_c);

  kernel_ptr<<< config.grid, config.block, dynamic_smem_bytes, stream >>>(pack);
  return config;
}


} // namespace gemm
} // namespace cutlass
