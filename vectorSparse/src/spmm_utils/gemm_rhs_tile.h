#ifndef GEMM_RHS_TILE_H
#define GEMM_RHS_TILE_H
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
namespace gemm {
    
    template <typename LoadType, int Block_K, int Block_N, int Padding_B>
    struct wmmaRHSTile {
        //
        // Static members
        //

        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//128/16=8//one inst load 8 half
        static constexpr int kValuesPerLd_ = (Block_N+Padding_B) / kValuesPerLoad_;//(64+8)/8=9


        //
        // Member variables
        //

        // the number of load_inst per row in lhs
        const int rhs_columns_;
        // The number of columns in the rhs matrix
        const int tid_;
        // The lhs matrix pointer in global memory
        const LoadType *matrix_base_;
        // The lhs matrix pointer in shared memory
        LoadType *rhs_tile_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaRHSTile(
            const half* __restrict__ B,
            int col_offset,
            int rhs_columns,
            half * rhs_tile,
            int tid,
            int lane_id,
            int thread_group):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            tid_(tid),
            matrix_base_(reinterpret_cast<const LoadType *>(B + col_offset + lane_id * 4 * rhs_columns) + thread_group),
            // row_offsets_base_(row_offsets + lane_id),
            rhs_tile_(reinterpret_cast<LoadType *>(rhs_tile + lane_id * 4 * (Block_N+Padding_B)) + thread_group){}

        __device__ __forceinline__ void LoadRow(int row_group_idx){
            LoadType * rhs_value_tile = rhs_tile_ + row_group_idx * 16 * kValuesPerLd_;
            
            *(rhs_value_tile) = __ldg(matrix_base_);
            *(rhs_value_tile + kValuesPerLd_) = __ldg(matrix_base_ + rhs_columns_);
            *(rhs_value_tile + 2*kValuesPerLd_) = __ldg(matrix_base_ + 2*rhs_columns_);
            *(rhs_value_tile + 3*kValuesPerLd_) = __ldg(matrix_base_ + 3*rhs_columns_);
            matrix_base_ += 16*rhs_columns_;
        }

    };

    template <int Block_M, int Block_K, int Block_N, int Padding_B>
    struct wmmaRHSTile_rowmajor {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(float4) / sizeof(half);//128/16=8//one inst load 8 half
        static constexpr int kValuesPerLd_ = (Block_N+Padding_B) / kValuesPerLoad_;//(64+8)/8=9
        //
        // Member variables
        //
        // the number of load_inst per row in lhs
        const int rhs_columns_;
        // thread id
        const int tid_;
        const int lane_id_;
        const int thread_group_;
        // The lhs matrix pointer in global memory
        const float4 *matrix_base_;
        // the lhs matrix pointer in reg
        float4 *rhs_reg_;
        // The lhs matrix pointer in shared memory
        float4 *rhs_tile_;

        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ wmmaRHSTile_rowmajor(
            const half* __restrict__ B,
            int col_offset,
            int rhs_columns,
            half * rhs_reg,
            half * rhs_tile,
            int tid):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            tid_(tid),
            lane_id_(tid%4),
            thread_group_(tid/4),
            matrix_base_(reinterpret_cast<const float4 *>(B + col_offset + lane_id_ * 4 * rhs_columns) + thread_group_),
            rhs_reg_(reinterpret_cast<float4 *>(rhs_reg)),//64 half -> 8 float4
            rhs_tile_(reinterpret_cast<float4 *>(rhs_tile + lane_id_ * 4 * (Block_N+Padding_B)) + thread_group_){}

        __device__ __forceinline__ void Prefetch(int row_group_idx){// 0 1
            // float4 * rhs_value_reg = ;
            
            *(rhs_reg_ + row_group_idx * 4    ) = __ldg(matrix_base_                   );
            *(rhs_reg_ + row_group_idx * 4 + 1) = __ldg(matrix_base_ + 1 * rhs_columns_);
            *(rhs_reg_ + row_group_idx * 4 + 2) = __ldg(matrix_base_ + 2 * rhs_columns_);
            *(rhs_reg_ + row_group_idx * 4 + 3) = __ldg(matrix_base_ + 3 * rhs_columns_);
            matrix_base_ += 16*rhs_columns_;
        }

        __device__ __forceinline__ void Load2Shared(int row_group_idx){// 0 1
            float4 * rhs_value_tile = rhs_tile_ + row_group_idx * 16 * kValuesPerLd_;

            *(rhs_value_tile                  ) = *(rhs_reg_ + row_group_idx * 4    );
            *(rhs_value_tile +   kValuesPerLd_) = *(rhs_reg_ + row_group_idx * 4 + 1);
            *(rhs_value_tile + 2*kValuesPerLd_) = *(rhs_reg_ + row_group_idx * 4 + 2);
            *(rhs_value_tile + 3*kValuesPerLd_) = *(rhs_reg_ + row_group_idx * 4 + 3);
        }

    };

    template <typename LoadType, int Block_K, int Block_N>
    struct mmaRHSFrag_Volta {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//128/16=8//one inst load 8 half
        //
        // Member variables
        //
        // the number of load_inst per row in lhs
        const int rhs_columns_;
        // const int tid_;
        // The lhs matrix pointer in global memory
        const LoadType *matrix_base_;
        // The lhs matrix pointer in reg
        LoadType *rhs_frag_;
        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ mmaRHSFrag_Volta(
            const half* __restrict__ B,
            int col_offset,
            int rhs_columns,
            half * rhs_frag,
            int tid,
            int lane_id,
            int thread_group):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            // tid_(tid),
            matrix_base_(reinterpret_cast<const LoadType *>(B + col_offset + lane_id * rhs_columns) + thread_group),
            // row_offsets_base_(row_offsets + lane_id),
            rhs_frag_(reinterpret_cast<LoadType *>(rhs_frag)){}

        __device__ __forceinline__ void LoadRow(int row_group_idx){
            
            *(rhs_frag_ + row_group_idx) = __ldg(matrix_base_);
            matrix_base_ += 4 * rhs_columns_;
        }
    };
    template <typename LoadType, int Block_K, int Block_N>
    struct mmaRHSFrag_Turing {
        //
        // Static members
        //
        static constexpr int kValuesPerLoad_ = sizeof(LoadType) / sizeof(half);//128/16=8//one inst load 8 half
        //
        // Member variables
        //
        // the number of load_inst per row in lhs
        const int rhs_columns_;
        const int tx_;
        const int ty_;
        const int tid_;
        // The lhs matrix pointer in global memory
        const LoadType *matrix_base_;
        // The lhs matrix pointer in reg
        LoadType *rhs_frag_;
        // Constructor. Set the initial pointer offsets
        __device__ __forceinline__ mmaRHSFrag_Turing(
            const half* __restrict__ B,
            int col_offset,
            int rhs_columns,
            half * rhs_frag,
            int tid):
            rhs_columns_(rhs_columns / kValuesPerLoad_),
            tid_(tid),
            tx_((tid%4) * 2 + (tid%8) / 4),// 0 4 1 5 2 6 3 7 -> 0 1 2 3 4 5 6 7
            ty_(tid / 8),
            matrix_base_(reinterpret_cast<const LoadType *>(B + col_offset + tx_ * rhs_columns) + ty_),
            // row_offsets_base_(row_offsets + lane_id),
            rhs_frag_(reinterpret_cast<LoadType *>(rhs_frag)){}

        __device__ __forceinline__ void LoadRow(int row_group_idx){// row_group_idx = 0 1 2 3
            // const LoadType * rhs_addr = matrix_base_ + ty_ + tx_ * rhs_columns_;
            *(rhs_frag_ + row_group_idx) = __ldg(matrix_base_);
            matrix_base_ += 8 * rhs_columns_;
            //shuffle
            half * rhs_half_reg = reinterpret_cast<half *>(rhs_frag_ + row_group_idx);//float4 -> 8*half
            int src_line = (tid_ + 4) % 8 + (tid_ / 8) * 8;
            #pragma unroll 4
            for(int i = 0; i < 4; i++){
                half temp;

                if(tid_ % 8 < 4) temp = rhs_half_reg[2*i+1];
                else temp = rhs_half_reg[2*i];

                temp = __shfl_sync(0xffffffff, temp, src_line, 32);

                if(tid_ % 8 < 4) rhs_half_reg[2*i+1] = temp;
                else rhs_half_reg[2*i] = temp;
            }
        }
    };


}

#endif