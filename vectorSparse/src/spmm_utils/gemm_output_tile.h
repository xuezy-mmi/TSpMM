#ifndef GEMM_OUTPUT_TILE_H
#define GEMM_OUTPUT_TILE_H
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
namespace gemm {
    template <typename OutType, int Block_M, int Block_K, int Block_N>
    struct wmma_OutputTile_83216{

        OutType * output_matrix_;
        const int columns_;
        wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c_;
        // int store_c_gmem_addr_;
        __device__ __forceinline__ wmma_OutputTile_83216 (
            int tid,
            int lane_id,
            int thread_group,
            const int N,
            int row_offset,
            int col_offset,
            wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c,
            OutType * output_matrix):
            columns_(N),
            // const int offset = col_offset + row_offset * N,
            output_matrix_(output_matrix + col_offset + row_offset * N),
            frag_c_(frag_c)
        {}

        __device__ __forceinline__ void Store(){
            // OutputTye * C = output_matrix_;
            wmma::store_matrix_sync(output_matrix_, frag_c_[0], columns_, wmma::mem_row_major);
            wmma::store_matrix_sync(output_matrix_ + 32, frag_c_[1], columns_, wmma::mem_row_major);

        }
    };
    
    template<typename OutType, typename StoreType>
    struct mmaOutput_Volta{
        //
        // Static members
        //

        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);

        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float2* output_fragment_;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ mmaOutput_Volta(
            int lane_id,
            int thread_group,
            int row_offset,
            int column_offset,
            int cols,
            float* output_fragment,//32
            OutType* output_matrix)//float
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            // const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
            const int output_offset = (row_offset * cols + column_offset) + (lane_id + (thread_group / 4)*4) * cols + (thread_group % 4) * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);//global addr
            rhs_columns_ = cols / kValuesPerStore_;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
        }

        // Store
        __device__ __forceinline__ void Store(){
            // Step 1: warp shuffle to align the memory access
            int src_line = (lane_id_ + 2) % 4 + thread_group_ * 4;//2 3 0 1 6 7 4 5
            #pragma unroll
            for (int i = 0; i < 4; i++){
                __align__(8) float temp[2];
                float2* temp_float2 = reinterpret_cast<float2 *>(temp);
//laneid=0 1的将output_fragment_[2,3]赋给temp[0,1] 6 7 10 11 14 15
//laneid=2 3的将output_fragment_[0,1]赋给temp[0,1] 4 5 8  9  12 13
                if (lane_id_ < 2) *(temp_float2) = output_fragment_[i * 2 + 1];
                else *(temp_float2) = output_fragment_[i * 2];
//获取线程src_line的temp[0,1]的值
                temp[0] = __shfl_sync(0xffffffff, temp[0], src_line, 32);
                temp[1] = __shfl_sync(0xffffffff, temp[1], src_line, 32);
//laneid=0 1的将线程src_line的output_fragment_[0,1]赋给自己的output_fragment_[2,3]
//laneid=2 3的将线程src_line的output_fragment_[2,3]赋给自己的output_fragment_[0,1]
                if (lane_id_ < 2) output_fragment_[i * 2 + 1] = *(temp_float2);
                else output_fragment_[i * 2] = *(temp_float2);
            }

            if (kTypeConvert != 1){
                float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
                OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
                #pragma unroll
                for(int i = 0; i < 16; i++){
                    output_fragment_outType[i] = (OutType)output_fragment_float[i];
                }
            }


            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            *(output_matrix_) = *(output_fragment_storetype);
            *(output_matrix_ + 1) = *(output_fragment_storetype + 2);
            *(output_matrix_ + 8) = *(output_fragment_storetype + 1);
            *(output_matrix_ + 9) = *(output_fragment_storetype + 3);

        }
    };

    template<typename OutType, typename StoreType>
    struct mmaOutput_Turing{
        //
        // Static members
        //
        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);// 2
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);
        //
        // Member variables
        //
        // The register file fragment with the results to store
        float2* output_fragment_;//2 output
        StoreType* output_matrix_;//half->float // float->float2
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ mmaOutput_Turing(
            int tid,
            int row_offset,
            int column_offset,
            int cols,
            float* output_fragment,//32
            OutType* output_matrix)//float
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            // const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
            const int output_offset = (row_offset * cols + column_offset) + (tid/4) * cols + (tid%4)*2;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);//global addr
            rhs_columns_ = cols / kValuesPerStore_;//cols / 2
        }

        // Store
        __device__ __forceinline__ void Store(){
            int store_num = 2;
            if (kTypeConvert != 1){//OutType = fp16
                float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
                OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
                #pragma unroll
                for(int i = 0; i < 16; i++){
                    output_fragment_outType[i] = (OutType)output_fragment_float[i];
                }
                store_num = 1;
            }
            // float4 *output_fragment_storetype = reinterpret_cast<float4 *>(output_fragment_);
            // // StoreType output_fragment_temp[8] = {};
            // // output_fragment_temp[0] = *(output_fragment_storetype    );
            // // output_fragment_temp[1] = *(output_fragment_storetype + 2);
            // // output_fragment_temp[2] = *(output_fragment_storetype + 4);
            // // output_fragment_temp[3] = *(output_fragment_storetype + 6);
            // // output_fragment_temp[4] = *(output_fragment_storetype + 1);
            // // output_fragment_temp[5] = *(output_fragment_storetype + 3);
            // // output_fragment_temp[6] = *(output_fragment_storetype + 5);
            // // output_fragment_temp[7] = *(output_fragment_storetype + 7);
            // // float4 * output_fragment_float4 = reinterpret_cast<float4 *>(output_fragment_storetype);
            // // float4 * output_matrix_float4 = reinterpret_cast<float4 *>(output_matrix_);
            // int load_num_per_row = rhs_columns_ / 2 / store_num;
            // #pragma unroll
            // for(int i = 0; i < store_num; i++){
            //     *(output_matrix_ + i) = *(output_fragment_storetype + i);

            //     *(output_matrix_ + 8*load_num_per_row + i) = *(output_fragment_storetype + store_num + i);
            // }
            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            *(output_matrix_     ) = *(output_fragment_storetype    );
            *(output_matrix_ +  1) = *(output_fragment_storetype + 2);
            *(output_matrix_ +  2) = *(output_fragment_storetype + 4);
            *(output_matrix_ +  3) = *(output_fragment_storetype + 6);

            *(output_matrix_ + 8 * rhs_columns_     ) = *(output_fragment_storetype + 1);
            *(output_matrix_ + 8 * rhs_columns_ +  1) = *(output_fragment_storetype + 3);
            *(output_matrix_ + 8 * rhs_columns_ +  2) = *(output_fragment_storetype + 5);
            *(output_matrix_ + 8 * rhs_columns_ +  3) = *(output_fragment_storetype + 7);

        }
    };

}
#endif