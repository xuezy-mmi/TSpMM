#ifndef GST_OUTPUT_Tile_H
#define GST_OUTPUT_Tile_H

namespace spmm{

    template<typename OutType, typename StoreType>
    struct wmmaOutputTile8{
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
        __device__ __forceinline__ wmmaOutputTile8(
            int lane_id,
            int thread_group,
            int tid,
            int row_offset_vec,
            int column_offset,
            int cols,
            float* output_fragment,//32
            OutType* output_matrix)//float
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            // const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
            const int output_offset = (row_offset_vec * 8 * cols + column_offset) + (lane_id + (thread_group / 4)*4) * cols + (thread_group % 4) * 8;
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
    struct wmmaOutputTile16{
        //
        // Static members
        //

        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);//4
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);

        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float2* output_fragment_1;
        float2* output_fragment_2;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile16(
            int lane_id, 
            int thread_group, 
            int row_offset_vec, 
            int column_offset,
            int cols, 
            float* output_fragment1,
            float* output_fragment2,
            OutType* output_matrix)//float
        {
            output_fragment_1 = reinterpret_cast<float2 *>(output_fragment1);
            output_fragment_2 = reinterpret_cast<float2 *>(output_fragment2);
            const int output_offset = (row_offset_vec * 16 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);
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
                __align__(16) float temp[2];
                __align__(16) float temp1[2];
                float2* temp_float2 = reinterpret_cast<float2 *>(temp);
                float2* temp1_float2 = reinterpret_cast<float2 *>(temp1);
                if (lane_id_ < 2){
                    *(temp_float2) = output_fragment_1[i * 2 + 1];
                    *(temp1_float2) = output_fragment_2[i * 2 + 1];
                }
                else{
                    *(temp_float2) = output_fragment_1[i * 2];
                    *(temp1_float2) = output_fragment_2[i * 2];
                }
                temp[0] = __shfl_sync(0xffffffff, temp[0], src_line, 32);
                temp[1] = __shfl_sync(0xffffffff, temp[1], src_line, 32);
                temp1[0] = __shfl_sync(0xffffffff, temp1[0], src_line, 32);
                temp1[1] = __shfl_sync(0xffffffff, temp1[1], src_line, 32);
                if (lane_id_ < 2){
                    output_fragment_1[i * 2 + 1] = *(temp_float2);
                    output_fragment_2[i * 2 + 1] = *(temp1_float2);
                }
                else{
                    output_fragment_1[i * 2] = *(temp_float2);
                    output_fragment_2[i * 2] = *(temp1_float2);
                }
            }

            if (kTypeConvert != 1){
                float* output_fragment1_float = reinterpret_cast<float *>(output_fragment_1);
                OutType* output_fragment1_outType = reinterpret_cast<OutType *>(output_fragment_1);
                float* output_fragment2_float = reinterpret_cast<float *>(output_fragment_2);
                OutType* output_fragment2_outType = reinterpret_cast<OutType *>(output_fragment_2);
                #pragma unroll
                for(int i = 0; i < 16; i++){
                    output_fragment1_outType[i] = (OutType)output_fragment1_float[i];
                    output_fragment2_outType[i] = (OutType)output_fragment2_float[i];
                }
            }


            StoreType *output_fragment_storetype1 = reinterpret_cast<StoreType *>(output_fragment_1);
            StoreType *output_fragment_storetype2 = reinterpret_cast<StoreType *>(output_fragment_2);
            *(output_matrix_) = *(output_fragment_storetype1);
            *(output_matrix_ + 1) = *(output_fragment_storetype1 + 2);
            *(output_matrix_ + 8) = *(output_fragment_storetype1 + 1);
            *(output_matrix_ + 9) = *(output_fragment_storetype1 + 3);
            *(output_matrix_ + rhs_columns_ * 8    ) = *(output_fragment_storetype2);
            *(output_matrix_ + rhs_columns_ * 8 + 1) = *(output_fragment_storetype2 + 2);
            *(output_matrix_ + rhs_columns_ * 8 + 8) = *(output_fragment_storetype2 + 1);
            *(output_matrix_ + rhs_columns_ * 8 + 9) = *(output_fragment_storetype2 + 3);
        }
    };

    template<typename OutType, typename StoreType>
    struct Transp_OutputTile8{
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
        int tid_;
        // The register file fragment with the results to store
        // float2* output_fragment_;
        float * output_fragment_;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ Transp_OutputTile8(
            int lane_id, 
            int thread_group,
            int tid,
            int row_offset_vec,
            int column_offset,
            int cols, 
            float* output_fragment,//32
            OutType* output_matrix)//float
        {
            // output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            output_fragment_ = output_fragment;//16 个 f32
            // const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
            const int output_offset = (row_offset_vec * 8 * cols + column_offset) + lane_id * cols + thread_group * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);//global addr
            rhs_columns_ = cols / kValuesPerStore_;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
            tid_ = tid;
        }

        // Store
        __device__ __forceinline__ void Store(){
            // Step 1: warp shuffle to align the memory access
            int src_line = (lane_id_ % 2) ? (tid_ - 1) : (tid_ + 1);//2 3 0 1 6 7 4 5
            #pragma unroll 8
            for (int i = 0; i < 8; i++){
                __align__(8) float temp;

                if (lane_id_ % 2) temp = output_fragment_[i * 2];
                else temp = output_fragment_[i * 2 + 1];

                temp = __shfl_sync(0xffffffff, temp, src_line, 32);

                if (lane_id_ % 2) output_fragment_[i * 2] = temp;
                else output_fragment_[i * 2 + 1] = temp;
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
            *(output_matrix_ + rhs_columns_ * 4) = *(output_fragment_storetype + 1);
            *(output_matrix_ + rhs_columns_ * 4 + 1) = *(output_fragment_storetype + 3);

        }
    };
    template<typename OutType, typename StoreType>
    struct mmaOutputTile8_Turing{
        //
        // Static members
        //

        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);//4
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);

        //
        // Member variables
        //
        int tid_;
        // The register file fragment with the results to store
        float* output_fragment_;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ mmaOutputTile8_Turing(
            int tid,
            int row_offset_vec,
            int column_offset,
            int cols,
            float* output_fragment,//8
            OutType* output_matrix)//
        {
            output_fragment_ = output_fragment;
            const int lane_id = threadIdx.x % 4;
            const int thread_group = threadIdx.x / 4;
            // const int output_offset = (row_offset_vec * 8 + lane_id + (thread_group / 4) * 4) * cols + column_offset + (thread_group % 4) * 8;
            const int output_offset = (row_offset_vec * 8 * cols + column_offset) + (lane_id) * 2 * cols + thread_group * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);//global addr
            rhs_columns_ = cols / kValuesPerStore_;
            tid_ = tid;
        }

        // Store
        __device__ __forceinline__ void Store(){
			float temp[16];
			#pragma unroll 16
			for(int i = 0; i < 16; i++){
				temp[i] = output_fragment_[i];
			}
			int row_id;//0 1
			int col_id;//0 - 7
			#pragma unroll 16
			for(int i = 0; i < 16; i++){
				row_id = i / 8;
				col_id = i % 8;
				output_fragment_[i] = temp[2*col_id + row_id];
			}
            if (kTypeConvert != 1){
                float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
                OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
                #pragma unroll 16
                for(int i = 0; i < 16; i++){
                    output_fragment_outType[i] = (OutType)output_fragment_float[i];
                }
            }

            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            *(output_matrix_) = *(output_fragment_storetype);
            *(output_matrix_ + 1) = *(output_fragment_storetype + 1);
            *(output_matrix_ + rhs_columns_) = *(output_fragment_storetype + 2);
            *(output_matrix_ + rhs_columns_ + 1) = *(output_fragment_storetype + 3);
        }
    };


}
#endif