#ifndef GEMM_COMPUTER_H
#define GEMM_COMPUTER_H
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
namespace gemm {
    template <typename OutType, int Block_M, int Block_K, int Block_N, int Padding_A, int Padding_B>
    struct wmma_ComputeUtils_83216{
		// Member variables
		// 
		const half * lhs_tile_;
		const half * rhs_tile_;
		wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> * frag_a_;
		wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> * frag_b_;
		wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c_;
		int tid_;
		int lane_id_;
		int thread_group_;
		
		__device__ __forceinline__ wmma_ComputeUtils_83216(
			const half * lhs_tile,//shared memory for lhs
			const half * rhs_tile,//shared memory for rhs
			wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> * frag_a,
			wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> * frag_b,
			wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c,
			int tid,
			int lane_id,
			int thread_group
			):
			lhs_tile_(lhs_tile),
			rhs_tile_(rhs_tile),
			frag_a_(frag_a),
			frag_b_(frag_b),
			frag_c_(frag_c),
			tid_(tid),
			lane_id_(lane_id),
			thread_group_(thread_group)
			
			{}
		__device__ __forceinline__ void wmmaLoadFragA(int row_group_idx){
			// const half * A_block = lhs_tile_ + row_group_idx * 16;
			wmma::load_matrix_sync(frag_a_[row_group_idx], lhs_tile_ + row_group_idx * 16, Block_K+Padding_A);
			// wmma::load_matrix_sync(frag_a_[1], &A_block[0][16], Block_K);
		}

		__device__ __forceinline__ void wmmaLoadFragB(int row_group_idx){
			// const half * B_block = rhs_tile_ + row_group_idx * 16 * (Block_N+Padding_B);
			// float r0, r1
			wmma::load_matrix_sync(frag_b_[2*row_group_idx  ], rhs_tile_ + row_group_idx * 16 * (Block_N+Padding_B), Block_N+Padding_B);
			wmma::load_matrix_sync(frag_b_[2*row_group_idx+1], rhs_tile_ + row_group_idx * 16 * (Block_N+Padding_B) + 32, Block_N+Padding_B);
		}

		__device__ __forceinline__ void wmmaCompute(int row_group_idx){
			wmma::mma_sync(frag_c_[0], frag_a_[row_group_idx], frag_b_[row_group_idx*2  ], frag_c_[0]);
			wmma::mma_sync(frag_c_[1], frag_a_[row_group_idx], frag_b_[row_group_idx*2+1], frag_c_[1]);
		}
    };
     
	template <typename OutType, int Block_M, int Block_K, int Block_N, int Padding_A, int Padding_B>
    struct wmma_ComputeUtils_83216_V2{
		// Member variables
		// 
		const half * lhs_tile_;
		const half * rhs_tile_;
		wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> * frag_a_;
		wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> * frag_b_;
		wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c_;
		int tid_;
		
		__device__ __forceinline__ wmma_ComputeUtils_83216_V2(
			const half * lhs_tile,//shared memory for lhs
			const half * rhs_tile,//shared memory for rhs
			wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> * frag_a,
			wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> * frag_b,
			wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c,
			int tid
			):
			lhs_tile_(lhs_tile),
			rhs_tile_(rhs_tile),
			frag_a_(frag_a),
			frag_b_(frag_b),
			frag_c_(frag_c),
			tid_(tid)
			
			{}
		__device__ __forceinline__ void wmmaLoadFragA(int step){// 0 1 
			const half * A_block = lhs_tile_ + (step%2) * 32 * 8;
			wmma::load_matrix_sync(frag_a_[0], A_block, Block_M);
			wmma::load_matrix_sync(frag_a_[1], A_block + 16 * 8, Block_M);
			// wmma::load_matrix_sync(frag_a_[1], &A_block[0][16], Block_K);
		}

		__device__ __forceinline__ void wmmaLoadFragB(int row_group_idx){
			const half * B_block = rhs_tile_ + row_group_idx * 16 * (Block_N+Padding_B);
			// float r0, r1
			wmma::load_matrix_sync(frag_b_[2*row_group_idx  ], B_block     , Block_N+Padding_B);
			wmma::load_matrix_sync(frag_b_[2*row_group_idx+1], B_block + 32, Block_N+Padding_B);
		}

		__device__ __forceinline__ void wmmaCompute(int row_group_idx){
			wmma::mma_sync(frag_c_[0], frag_a_[row_group_idx], frag_b_[row_group_idx*2  ], frag_c_[0]);
			wmma::mma_sync(frag_c_[1], frag_a_[row_group_idx], frag_b_[row_group_idx*2+1], frag_c_[1]);
		}
    };

    template <typename OutType, int Block_M, int Block_K, int Block_N, int Padding_A, int Padding_B>
    struct wmma_ComputeUtils_83216_colmajor{
		// Member variables
		// 
		const half * lhs_tile_;
		const half * rhs_tile_;
		wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> * frag_a_;
		wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> * frag_b_;
		wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c_;
		int tid_;
		int lane_id_;
		int thread_group_;
		
		__device__ __forceinline__ wmma_ComputeUtils_83216_colmajor(
			const half * lhs_tile,//shared memory for lhs
			const half * rhs_tile,//shared memory for rhs
			wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::col_major> * frag_a,
			wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> * frag_b,
			wmma::fragment<wmma::accumulator, 8, 32, 16, OutType> * frag_c,
			int tid,
			int lane_id,
			int thread_group
			):
			lhs_tile_(lhs_tile),
			rhs_tile_(rhs_tile),
			frag_a_(frag_a),
			frag_b_(frag_b),
			frag_c_(frag_c),
			tid_(tid),
			lane_id_(lane_id),
			thread_group_(thread_group)
			
			{}
		__device__ __forceinline__ void wmmaLoadFragA(int row_group_idx){
			// const half * A_block = lhs_tile_ + row_group_idx * 16;
			wmma::load_matrix_sync(frag_a_[row_group_idx], lhs_tile_ + row_group_idx * Block_M * Block_K, Block_M);
			// wmma::load_matrix_sync(frag_a_[1], &A_block[0][16], Block_K);
		}

		__device__ __forceinline__ void wmmaLoadFragB(int row_group_idx){
			// const half * B_block = rhs_tile_ + row_group_idx * 16 * (Block_N+Padding_B);
			// float r0, r1
			wmma::load_matrix_sync(frag_b_[2*row_group_idx  ], rhs_tile_ + row_group_idx * 16 * (Block_N+Padding_B), Block_N+Padding_B);
			wmma::load_matrix_sync(frag_b_[2*row_group_idx+1], rhs_tile_ + row_group_idx * 16 * (Block_N+Padding_B) + 32, Block_N+Padding_B);
		}

		__device__ __forceinline__ void wmmaCompute(int row_group_idx){
			wmma::mma_sync(frag_c_[0], frag_a_[row_group_idx], frag_b_[row_group_idx*2  ], frag_c_[0]);
			wmma::mma_sync(frag_c_[1], frag_a_[row_group_idx], frag_b_[row_group_idx*2+1], frag_c_[1]);
		}
    };

    struct mmaCompute_Volta {
        //
        // Static members
        //
        // static constexpr int kTotalStep = Tile_N / 4 - 1;//32/4-1=7
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const float2* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        // Constructor
        __device__ __forceinline__ mmaCompute_Volta(
            const half * lhs_tile,//shared float4 [Tile_N]
            const half* rhs_fragment,//reg half [64]
            float* output_fragment,//reg float [16]
            int tid, int lane_id, int thread_group):
            lhs_tile_(reinterpret_cast<const float2 *>(lhs_tile) + lane_id * 2 + thread_group / 4),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            float lhs_fragment[2];
            
            float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);
            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);
            
            #pragma unroll 2
            for (int i = 0; i < 2; i++){
                asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}, \t"
                    "{%8, %9}, \t"
                    "{%10, %11}, \t"
                    "{%0, %1, %2, %3, %4, %5, %6, %7}; ":
                    "+f"(output_fragment_[0 + 8 * i]), "+f"(output_fragment_[1 + 8 * i]),
                    "+f"(output_fragment_[2 + 8 * i]), "+f"(output_fragment_[3 + 8 * i]),
                    "+f"(output_fragment_[4 + 8 * i]), "+f"(output_fragment_[5 + 8 * i]),
                    "+f"(output_fragment_[6 + 8 * i]), "+f"(output_fragment_[7 + 8 * i]):
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[0 + 2 * i]), "r"(rhs_fragment_int[1 + 2 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };

	    struct mmaCompute_Turing {
        //
        // Static members
        //
        // static constexpr int kTotalStep = Tile_N / 4 - 1;//32/4-1=7
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const half* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        // Constructor 
        __device__ __forceinline__ mmaCompute_Turing(
            const half * lhs_tile,//shared half [bm*bk]
            const half * rhs_fragment,//reg half [4*8]
            float* output_fragment,//reg float [16]
            int tid):
            lhs_tile_(lhs_tile + tid/4 + (tid%4)*32),
			// tx=tid/4 // ty = (tid%4)*2
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){//n_group_idx = 0 1 2 3
            half lhs_fragment[4];
            lhs_fragment[0] = *(lhs_tile_      + n_group_idx * 128);
			lhs_fragment[1] = *(lhs_tile_ + 16 + n_group_idx * 128);
			lhs_fragment[2] = *(lhs_tile_ +  8 + n_group_idx * 128);
			lhs_fragment[3] = *(lhs_tile_ + 24 + n_group_idx * 128);
			int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);//4*half -> 2*int
            // float2 *lhs_fragment_float2 = reinterpret_cast<float2 *>(lhs_fragment);
            // *(lhs_fragment_float2) = *(lhs_tile_ + n_group_idx * 8);
            // int* lhs_fragment_int = reinterpret_cast<int *>(lhs_fragment);

            const int* rhs_fragment_int = reinterpret_cast<const int *>(rhs_fragment_ + 8 * n_group_idx);//8 half -> 4 int
            
            #pragma unroll 4
            for (int i = 0; i < 4; i++){
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%7, %8, %9, %10}; ":
                    "=f"(output_fragment_[0 + 4 * i]), "=f"(output_fragment_[1 + 4 * i]),
                    "=f"(output_fragment_[2 + 4 * i]), "=f"(output_fragment_[3 + 4 * i]):
                    "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1]),
                    "r"(rhs_fragment_int[i]),
					"f"(output_fragment_[0 + 4 * i]), "f"(output_fragment_[1 + 4 * i]),
					"f"(output_fragment_[2 + 4 * i]), "f"(output_fragment_[3 + 4 * i])
                    // "r"(lhs_fragment_int[0]), "r"(lhs_fragment_int[1])
                );
            }
        }
    };

}
#endif