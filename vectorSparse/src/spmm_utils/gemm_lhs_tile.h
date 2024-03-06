#ifndef GEMM_LHS_TILE_H
#define GEMM_LHS_TILE_H
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
namespace gemm {
    template <typename LoadType, int Block_M, int Block_K, int Padding_A>
    struct wmmaLHSTile {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//128/16=8
        static constexpr int kValuesPerLd_ = Block_K / kValuesPerLoad_;//32/8=4
        //
        // Member variables
        //

        // the number of load_inst per row in lhs
        const int lhs_columns_;
        // The number of columns in the rhs matrix
        const int tid_;
        // The lhs matrix pointer in global memory
        const LoadType *matrix_base_;
        // The lhs matrix pointer in shared memory
        LoadType *lhs_tile_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaLHSTile(
            const half* __restrict__ A,
            int row_offset,
            int lhs_columns,
            half * lhs_tile,
            int tid,
            int lane_id,
            int thread_group):
            lhs_columns_(lhs_columns / kValuesPerLoad_),
            tid_(tid),
            matrix_base_(reinterpret_cast<const LoadType *>(A + (row_offset + thread_group) * lhs_columns) + lane_id),
            // row_offsets_base_(row_offsets + lane_id),
            lhs_tile_(reinterpret_cast<LoadType *>(lhs_tile + thread_group * (Block_K+Padding_A)) + lane_id){}

        __device__ __forceinline__ void Load(){
            LoadType * lhs_value_tile = lhs_tile_;
            
            *(lhs_value_tile) = __ldg(matrix_base_);
            matrix_base_ += kValuesPerLd_;
        }

    };

    template <int Block_M, int Block_K, int Block_N, int Padding_A>
    struct wmmaLHSTile_colmajor {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(float4) / sizeof(half);//128/16=8
        // static constexpr int kValuesPerLd_ = Block_K / kValuesPerLoad_;//32/8=4
        //
        // Member variables
        //
        // the number of load_inst per row in lhs
        const int lhs_columns_;
        // thread id
        const int tid_;
        // The lhs matrix pointer in global memory
        const float4 *matrix_base_;
        // The lhs matrix pointer in shared memory
        float4 *lhs_tile_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaLHSTile_colmajor(
            const half* __restrict__ A,
            int row_offset,
            int lhs_columns,
            half * lhs_tile,
            int tid):
            lhs_columns_(lhs_columns / kValuesPerLoad_),
            tid_(tid),
            matrix_base_(reinterpret_cast<const float4 *>(A + row_offset * lhs_columns) + tid),
            // row_offsets_base_(row_offsets + lane_id),
            lhs_tile_(reinterpret_cast<float4 *>(lhs_tile) + tid){}

        __device__ __forceinline__ void Load(int step){
            float4 * lhs_value_tile = lhs_tile_ + (step % 2) * Block_K;
            
            *(lhs_value_tile) = __ldg(matrix_base_);
            matrix_base_ += 32;
        }

    };

    template <typename LoadType, int Block_M, int Block_K>
    struct mmaLHSTile_Volta {
        //
        // Member variables
        //
        // The lhs matrix pointer in global memory
        const LoadType *matrix_base_;
        // The lhs matrix pointer in shared memory
        LoadType *lhs_tile_;
        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ mmaLHSTile_Volta(
            const half* __restrict__ A,
            int row_offset,
            int lhs_columns,
            half * lhs_tile,
            int tid,
            int lane_id,
            int thread_group):
            matrix_base_(reinterpret_cast<const LoadType *>(A + row_offset * lhs_columns) + tid),
            // row_offsets_base_(row_offsets + lane_id),
            lhs_tile_(reinterpret_cast<LoadType *>(lhs_tile) + tid){}

        __device__ __forceinline__ void Load(){
            LoadType * lhs_value_tile = lhs_tile_;
            
            *(lhs_value_tile) = __ldg(matrix_base_);
            matrix_base_ += Block_K;
        }
    };

    template <typename LoadType, int Block_M, int Block_K>
    struct mmaLHSTile_Turing {
        //
        // Member variables
        //
        // The lhs matrix pointer in global memory
        const LoadType * matrix_base_;
        // The lhs matrix pointer in shared memory
        LoadType * lhs_tile_;
        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ mmaLHSTile_Turing(
            const half* __restrict__ A,
            int row_offset,
            int lhs_columns,
            half * lhs_tile,
            int tid):
            matrix_base_(reinterpret_cast<const LoadType *>(A + row_offset * lhs_columns) + tid),
            // row_offsets_base_(row_offsets + lane_id),
            lhs_tile_(reinterpret_cast<LoadType *>(lhs_tile) + tid){}

        __device__ __forceinline__ void Load(int row_group_idx){//row_group_idx = 0 or 1
            *(lhs_tile_ + row_group_idx * 32) = __ldg(matrix_base_);
            matrix_base_ += 32;
        }
    };

}
#endif