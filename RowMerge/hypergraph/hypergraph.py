#!/usr/bin/python3
import os
import random
from math import log
from copy import deepcopy
import matplotlib.pyplot as plt

transformer_matrices_data_dir = "dlmc/transformer_matrices.txt"



def log2(num):
    return log(num, 2)

def binary(num):
    return bin(num).split('0b')[1]

def binary_align32(num):
    return bin(num).split('0b')[1].rjust(32,'0')

def Processing_Matrices_Transformer(data_dir, i, inst_M):
    transformer_matrices_file_list = open(data_dir, 'r')
    transformer_matrices_files = transformer_matrices_file_list.readlines()
    transformer_matrices_file_list.close()

    for file_name in transformer_matrices_files[i+2270:i+2271]:
        file = os.path.join('dlmc', file_name.split('\n')[0])
        print(file)

        # Line 0: ROW, COL, NNZ
        # CSR Format:
        #       Line 1: ROW_INDEXES
        #       LINE 2: COL_INDEXES
        ROW = 0
        COL = 0
        NNZ = 0
        ROW_INDEXES = []
        COL_INDEXES = []

        data_of_this_file = open(file, 'r')
        data = data_of_this_file.readlines()
        data_of_this_file.close()

        ROW_COL_NNZ = data[0].split('\n')[0].split()
        ROW, COL, NNZ = int(ROW_COL_NNZ[0].split(',')[0]), \
                        int(ROW_COL_NNZ[1].split(',')[0]), \
                        int(ROW_COL_NNZ[2].split(',')[0])
        
        hypergraph_patitioning_parts = int(ROW / inst_M)
        origin_A_matrix = [[0 for _ in range(COL)] for __ in range(ROW)]
        
        if NNZ > 0:
            ROW_INDEXES = data[1].split('\n')[0].split()
            for _ in range(len(ROW_INDEXES)):
                ROW_INDEXES[_] = int(ROW_INDEXES[_])

            COL_INDEXES = data[2].split('\n')[0].split()
            for _ in range(len(COL_INDEXES)):
                COL_INDEXES[_] = int(COL_INDEXES[_])

            ### restore A matrix from CSR format
            
            for _ in range(ROW):
                for __ in COL_INDEXES[ROW_INDEXES[_]:ROW_INDEXES[_+1]]:
                    origin_A_matrix[_][__] = 1#abs(float("%.1f" % random.random())) + 0.1
            # print(origin_A_matrix)
            # return

            ######################################################################
            ######################################################################
            ######################################################################
            ### error check by generating CRS format
            error_check_indexes = []
            for _ in range(len(origin_A_matrix)):
                for __ in range(len(origin_A_matrix[0])):
                    if not origin_A_matrix[_][__] == 0:
                        error_check_indexes.append(__)
            if not len(error_check_indexes) == len(COL_INDEXES):
                print("ERROR: len is not equal")
            for _ in range(len(error_check_indexes)):
                if not error_check_indexes[_] == COL_INDEXES[_]:
                    print("ERROR: ", error_check_indexes[_], '!=', COL_INDEXES[_])
            print("PASS")

            ######################################################################
            ######################################################################
            ######################################################################
            ### restore hypergraph
            vertices_weight = [0 for _ in range(ROW)] # A rows
            nets_vertices_list = [[] for _ in range(COL)]

            for _ in range(ROW):
                vertices_weight[_] = ROW_INDEXES[_+1] - ROW_INDEXES[_]

            for _ in range(ROW):
                column_indexes_of_this_row = COL_INDEXES[ROW_INDEXES[_]:ROW_INDEXES[_+1]]
                for column in column_indexes_of_this_row:
                    if not _ in nets_vertices_list[column]:
                        nets_vertices_list[column].append(_)
                # print(column_indexes_of_this_row)

            pins_num = 0
            for _ in range(len(nets_vertices_list)):
                for __ in range(len(nets_vertices_list[_])):
                    pins_num += 1

            f_myken_u = open("./myken.u", 'w')
            f_myken_u.writelines(str(0)+' '+str(ROW)+' '+str(COL)+' '+str(pins_num)+'\n')
            for _ in range(len(nets_vertices_list)):
                string = ''
                for __ in nets_vertices_list[_]:
                    string += str(__)+' '
                string += '\n'
                f_myken_u.writelines(string)
            for _ in vertices_weight:
                f_myken_u.writelines(str(_)+'\n')
            f_myken_u.close()

            ######################################################################
            ######################################################################
            ######################################################################
            ### do hypergraph patitioning


            os.system('cd PaToH/linux/ && ./patoh ../../myken.u '+str(hypergraph_patitioning_parts))
            os.system('cd ../../')
            # os.system('cat myken.u.part.'+str(hypergraph_patitioning_parts))

            ### reorgnize A matrix
            file_ordering_list = open('myken.u.part.'+str(hypergraph_patitioning_parts), 'r')
            ordering_list = file_ordering_list.readlines()
            file_ordering_list.close()

            for _ in range(len(ordering_list)):
                ordering_list[_] = int(ordering_list[_].split('\n')[0])
            # print(ordering_list)

            reorgnize_A_matrix = [[0 for _ in range(COL)] for __ in range(ROW)]

            rows_haved_in_single_part_of = [0 for i in range(hypergraph_patitioning_parts)]
            for _ in range(len(ordering_list)):
                # _ is row number, ordering_list[_] is the reorgnized_part of current row.

                reorgnize_A_matrix[ordering_list[_]*int(ROW/hypergraph_patitioning_parts)+\
                rows_haved_in_single_part_of[ordering_list[_]]] = deepcopy(origin_A_matrix[_])
                rows_haved_in_single_part_of[ordering_list[_]] += 1
            
            ####################################################################################
            # origin_A_matrix, reorgnize_A_matrix
            return origin_A_matrix, reorgnize_A_matrix, NNZ
        
        else:
            return origin_A_matrix, origin_A_matrix, NNZ

# def plot_dot_figure(origin_A_matrix, reorgnize_A_matrix):
#     origin_A_matrix_rows = []
#     origin_A_matrix_cols = []
#     for _ in range(len(origin_A_matrix)):
#         for __ in range(len(origin_A_matrix[0])):
#             if not origin_A_matrix[_][__] == 0:
#                 origin_A_matrix_rows.append(_)
#                 origin_A_matrix_cols.append(__)
    
#     plt.figure(figsize=(10, 10), dpi=100)
#     plt.scatter(origin_A_matrix_cols, origin_A_matrix_rows, s=0.1)
#     plt.savefig('origin_A_matrix.pdf', bbox_inches='tight')
#     #plt.show()
#     # ax = plt.gca()
#     # ax.xaxis.set_ticks_position('top')
#     # ax.invert_yaxis()
#     # plt.scatter(origin_A_matrix_cols, origin_A_matrix_rows, s=0.1)
#     # plt.show()
#     # plt.savefig('origin_A_matrix.pdf', bbox_inches='tight')
#     plt.close()
    
#     reorgnize_A_matrix_rows = []
#     reorgnize_A_matrix_cols = []
#     for _ in range(len(reorgnize_A_matrix)):
#         for __ in range(len(reorgnize_A_matrix[0])):
#             if not reorgnize_A_matrix[_][__] == 0:
#                 reorgnize_A_matrix_rows.append(_)
#                 reorgnize_A_matrix_cols.append(__)
    
#     plt.figure(figsize=(10, 10), dpi=100)
    
#     plt.scatter(reorgnize_A_matrix_cols, reorgnize_A_matrix_rows, s=0.1)
#     plt.savefig('reorgnize_A_matrix.pdf', bbox_inches='tight')
#     plt.show()
#     plt.close()
#     # ax = plt.gca()
#     # ax.xaxis.set_ticks_position('top')
#     # ax.invert_yaxis()
#     # plt.scatter(reorgnize_A_matrix_cols, reorgnize_A_matrix_rows, s=0.1)
#     # plt.show()
#     # plt.savefig('reorgnize_A_matrix.pdf', bbox_inches='tight')
#     # plt.close()

# def calculate_reuse_bak(origin_A_matrix, reorgnize_A_matrix):
    
#     # print(origin_A_matrix)
#     # print(reorgnize_A_matrix)
#     # return 
#     # for i in range(len(origin_A_matrix)):
#     #     for j in range(len(origin_A_matrix[0])):
#     #         if origin_A_matrix[i][j] == 0:
#     #             print(0, end="")
#     #         elif origin_A_matrix[i][j] != 0:
#     #             print(1, end="")
#     #     print()
#     # print("================")
#     # return 

#     # for _ in range(len(origin_A_matrix[0])):
#     #     true_all_row_is_zero = True
#     #     non_zero_num = 0
#     #     for __ in range(len(origin_A_matrix)):
#     #         if not origin_A_matrix[__][_] == 0:
#     #             true_all_row_is_zero = False
#     #             non_zero_num += 1
#     #             continue
#     #     print(true_all_row_is_zero, non_zero_num)
#     # return

#     A_row = len(origin_A_matrix)
#     A_col = len(origin_A_matrix[0])
#     B_row = A_col
#     B_col = 512
#     Parts = [16, 16, 16]
    
#     A_block_nonzero_num = 0
#     for _ in range(int(A_row/Parts[0])):
#         for __ in range(int(A_col/Parts[1])):
#             for ___ in range(int(B_col/Parts[2])):
#                 origin_A_block = deepcopy(origin_A_matrix[_*Parts[0]:(_+1)*Parts[0]])
#                 for row in range(len(origin_A_block)):
#                     origin_A_block[row] = origin_A_block[row][__*Parts[1]:(__+1)*Parts[1]]
#                 flag_has_cached = [0 for i in range(len(origin_A_block[0]))]
#                 for row in range(len(origin_A_block)):
#                     for col in range(len(origin_A_block[0])):
#                         # if not origin_A_block[row][col] == 0 and flag_has_cached[col] == 0:
#                         #     B_cache_num += B_col
#                         #     flag_has_cached[col] = 1

#                         if not origin_A_block[row][col] == 0 and flag_has_cached[col] == 0:
#                             flag_has_cached[col] = 1
#                             A_block_nonzero_num += 1
#                 # print(flag_has_cached, A_block_nonzero_num)
#     print("A_block_nonzero_num:", A_block_nonzero_num)
    
#     A_block_nonzero_num = 0
#     for _ in range(int(A_row/Parts[0])):
#         for __ in range(int(A_col/Parts[1])):
#             for ___ in range(int(B_col/Parts[2])):
#                 reorgnize_A_block = deepcopy(reorgnize_A_matrix[_*Parts[0]:(_+1)*Parts[0]])
#                 for row in range(len(reorgnize_A_block)):
#                     reorgnize_A_block[row] = reorgnize_A_block[row][__*Parts[1]:(__+1)*Parts[1]]
#                 flag_has_cached = [0 for i in range(len(reorgnize_A_block[0]))]
#                 for row in range(len(reorgnize_A_block)):
#                     for col in range(len(reorgnize_A_block[0])):
#                         # if not reorgnize_A_block[row][col] == 0 and flag_has_cached[col] == 0:
#                         #     B_cache_num += B_col
#                         #     flag_has_cached[col] = 1

#                         if not reorgnize_A_block[row][col] == 0 and flag_has_cached[col] == 0:
#                             flag_has_cached[col] = 1
#                             A_block_nonzero_num += 1
#                 # print(flag_has_cached, A_block_nonzero_num)
#     print("A_block_nonzero_num:", A_block_nonzero_num)


# class L2_cache_block():
#     # One cache block consists of four parts: valid, U_flag, tag, data
#     def __init__(self, bsize):
#         self.valid = 0
#         self.U_flag = 0
#         self.tag = None
#         self.bsize = bsize
#         self.data = [None for _ in range(self.bsize)]
    
# class memory_partition_L2_cache():
#     # V100 has 64 memory partitions (32 memory channels * 2 partitions per channel)
#     def __init__(self, partition_id, nsets, bsize, assoc):
#         self.partition_id = partition_id
#         self.nsets = nsets
#         self.bsize = bsize
#         self.assoc = assoc
#         self.cache_blocks = [[L2_cache_block(self.bsize) for i in range(assoc)] for j in range(nsets)]

#         # [31, log2(self.nsets)+log2(self.bsize)]
#         self.addr_tag_bits = (31, int(log2(self.nsets)+log2(self.bsize)))
#         # (log2(self.nsets)+log2(self.bsize), log2(self.bsize)]
#         self.addr_set_bits = (int(log2(self.nsets)+log2(self.bsize)), int(log2(self.bsize)))
#         # (log2(self.bsize), 0]
#         self.addr_byte_offset_bits = (int(log2(self.bsize)), 0)

#         self.addr_tag_bits_indices = (0, 32-int(log2(self.nsets)+log2(self.bsize)))
#         self.addr_set_bits_indices = (32-int(log2(self.nsets)+log2(self.bsize)), 32-int(log2(self.bsize)))
#         self.addr_byte_offset_bits_indices = (32-int(log2(self.bsize)), 32)
        
#         # statistics
#         self.num_total_reads = 0
#         self.read_hit = 0
#         self.read_miss = 0
        
#     def read(self, addr):
#         self.num_total_reads += 1

#         ## Process the addr, get the tag, set, byte_offset.
#         addr = binary_align32(addr)
#         addr_tag = addr[self.addr_tag_bits_indices[0] : self.addr_tag_bits_indices[1]]
#         addr_set = addr[self.addr_set_bits_indices[0] : self.addr_set_bits_indices[1]]
#         addr_byte_offset = addr[self.addr_byte_offset_bits_indices[0] : self.addr_byte_offset_bits_indices[1]]
#         # print(addr_tag, addr_set, addr_byte_offset)

#         ## Get the corresponding cache line by the addr_set.
#         found_cache_line = self.cache_blocks[int(addr_set, 2)]
#         for i in range(self.assoc):
#             if found_cache_line[i].valid == 1 and found_cache_line[i].tag == addr_tag:
#                 # if the loop returns, it means that the cache status is hit.
#                 self.read_hit += 1
#                 ## Here, when a cache block is used, we should set the U_flag to 0, and add the U_flag bit of other cache 
#                 ## blocks in the same line by 1. For example, when a cache block is used, the U_flag of this cache block 
#                 ## is 0, and other cache blocks in the same line are 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, respectively, and when 
#                 ## we want to find a cache block to replace, we will find the cache block with the biggest U_flag. 
#                 found_cache_line[i].U_flag = 0
#                 for j in range(self.assoc):
#                     if not j == i:
#                         found_cache_line[j].U_flag += 1
#                 # return found_cache_line[i].data[int(addr_byte_offset, 2)]
#                 return 'hit'
        
#         ## If the upper loop does not return and exit, it means that the cache status is miss.
#         self.read_miss += 1

#         ## Then we should get data from the dram by addr and update the cache.
#         ## For ease of simulation, we here do not simulate the data path here, only simulate the status of the cache.
        
#         ## Next, we will update the status of cache by addr.
#         ## 1 st step: find an invalid cache block to place the data block where addr represents.
#         for i in range(self.assoc):
#             if found_cache_line[i].valid == 0:
#                 ## We should the data block here, the data block in dram is represented by addr.
#                 found_cache_line[i].valid = 1
#                 found_cache_line[i].tag = addr_tag
#                 ## Here, when a cache block is used, we should set the U_flag to 0, and add the U_flag bit of other cache 
#                 ## blocks in the same line by 1. For example, when a cache block is used, the U_flag of this cache block 
#                 ## is 0, and other cache blocks in the same line are 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, respectively, and when 
#                 ## we want to find a cache block to replace, we will find the cache block with the biggest U_flag. 
#                 found_cache_line[i].U_flag = 0
#                 for j in range(self.assoc):
#                     if not j == i:
#                         found_cache_line[j].U_flag += 1
#                 return 'miss'
#         ## 2 nd step: If the upper loop does not return and exit, it means there is no invalid cache block, 
#         ##            we should find another cache block to replace, here we adopt the LRU strategy.
#         ##          2.1 find the cache block with the biggest U_flag.
#         should_replace_cache_block_index = 0
#         for i in range(self.assoc):
#             if found_cache_line[i].U_flag > found_cache_line[should_replace_cache_block_index].U_flag:
#                 should_replace_cache_block_index = i
#         ##          2.2 replace the cache block with the biggest U_flag by the data block in dram represented by addr.
#         found_cache_line[should_replace_cache_block_index].valid = 1
#         found_cache_line[should_replace_cache_block_index].tag = addr_tag
#         ## Here, when a cache block is used, we should set the U_flag to 0, and add the U_flag bit of other cache
#         ## blocks in the same line by 1. For example, when a cache block is used, the U_flag of this cache block
#         ## is 0, and other cache blocks in the same line are 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, respectively, and when
#         ## we want to find a cache block to replace, we will find the cache block with the biggest U_flag.
#         found_cache_line[should_replace_cache_block_index].U_flag = 0
#         for j in range(self.assoc):
#             if not j == should_replace_cache_block_index:
#                 found_cache_line[j].U_flag += 1
#         return 'miss'

# def test_single_partition_L2_cache():
#     single_partition_L2_cache = memory_partition_L2_cache(partition_id=0, nsets=32, bsize=128, assoc=24)
#     print(single_partition_L2_cache.read(addr=0x1342abcd))
#     print(single_partition_L2_cache.read(addr=0x1342abce))
#     print(single_partition_L2_cache.read(addr=0x1342abcf))
#     print(single_partition_L2_cache.read(addr=0x1342abd0))
#     print(single_partition_L2_cache.read(addr=0x1342abd1))
#     print(single_partition_L2_cache.read(addr=0x1342abd2))
#     print(single_partition_L2_cache.read(addr=0x1342abd3))
#     print(single_partition_L2_cache.read(addr=0x1342abd4))
#     print(single_partition_L2_cache.read(addr=0x1342abd5))
#     print(single_partition_L2_cache.read(addr=0x1342abd6))
#     print(single_partition_L2_cache.read(addr=0x1342abd7))
#     print(single_partition_L2_cache.read(addr=0x1342abd8))
#     print(single_partition_L2_cache.read(addr=0x1342abd9))
#     print(single_partition_L2_cache.read(addr=0x1342abda))
#     print(single_partition_L2_cache.read(addr=0x1342abdb))
#     print(single_partition_L2_cache.read(addr=0x1342abdc))
#     print(single_partition_L2_cache.read(addr=0x1342abdd))
#     print(single_partition_L2_cache.read(addr=0x1342abde))
#     print(single_partition_L2_cache.read(addr=0x1342abdf))
#     print(single_partition_L2_cache.read(addr=0x1342abe0))
#     print(single_partition_L2_cache.read(addr=0x1342abe0))
#     print(single_partition_L2_cache.read(addr=0x1342ab4d))

# # Simulation of the dram.
# def DRAM():
#     def __init__(self):
#         pass

# def generate_cache_traces(origin_A_matrix, reorgnize_A_matrix):
#     A_row = len(origin_A_matrix)
#     A_col = len(origin_A_matrix[0])
#     B_row = A_col
#     B_col = 512

#     # ## 1. Generate the cache traces of Matrix B of the origin A matrix.
#     # origin_A_matrix_traces = []
#     # for i in range(len(origin_A_matrix)):
#     #     for j in range(len(origin_A_matrix[i])):
#     #         if origin_A_matrix[i][j] != 0:
#     #             for k in range(B_col):
#     #                 origin_A_matrix_traces.append(j*B_col+k)

#     # ## 2. Generate the cache traces of Matrix B of the reorgnize A matrix.
#     # reorgnize_A_matrix_traces = []
#     # for i in range(len(reorgnize_A_matrix)):
#     #     for j in range(len(reorgnize_A_matrix[i])):
#     #         if reorgnize_A_matrix[i][j] != 0:
#     #             for k in range(B_col):
#     #                 reorgnize_A_matrix_traces.append(j*B_col+k)
    
#     # return origin_A_matrix_traces, reorgnize_A_matrix_traces

#     origin_A_matrix_traces = []
#     reorgnize_A_matrix_traces = []

#     ## 1. Generate the cache traces of Matrix A of the origin A matrix.
#     for k in range(int(B_row/hypergraph_patitioning_parts)):
#         for m in range(int(A_row/hypergraph_patitioning_parts)):
#             for n in range(int(B_col/hypergraph_patitioning_parts)):
#                 for _ in range(hypergraph_patitioning_parts):
#                     for __ in range(hypergraph_patitioning_parts):
#                         for ___ in range(hypergraph_patitioning_parts):
#                             if origin_A_matrix[m*hypergraph_patitioning_parts+_][k*hypergraph_patitioning_parts+__] != 0:
#                                 origin_A_matrix_traces.append(((k*hypergraph_patitioning_parts+_)*B_col+n*hypergraph_patitioning_parts+__)*8)
    
#     ## 2. Generate the cache traces of Matrix A of the reorgnize A matrix.
#     for k in range(int(B_row/hypergraph_patitioning_parts)):
#         for m in range(int(A_row/hypergraph_patitioning_parts)):
#             for n in range(int(B_col/hypergraph_patitioning_parts)):
#                 for _ in range(hypergraph_patitioning_parts):
#                     for __ in range(hypergraph_patitioning_parts):
#                         for ___ in range(hypergraph_patitioning_parts):
#                             if reorgnize_A_matrix[m*hypergraph_patitioning_parts+_][k*hypergraph_patitioning_parts+__] != 0:
#                                 reorgnize_A_matrix_traces.append(((k*hypergraph_patitioning_parts+_)*B_col+n*hypergraph_patitioning_parts+__)*8)
    
#     return origin_A_matrix_traces, reorgnize_A_matrix_traces


# def calculate_reuse_cache(origin_A_matrix, reorgnize_A_matrix):
#     origin_A_matrix_traces, reorgnize_A_matrix_traces = generate_cache_traces(origin_A_matrix, reorgnize_A_matrix)

#     # single_partition_L2_cache = memory_partition_L2_cache(partition_id=0, nsets=32, bsize=128, assoc=24)
#     single_partition_L2_cache = memory_partition_L2_cache(partition_id=0, nsets=32, bsize=16, assoc=24)
#     # test_single_partition_L2_cache()
    
#     for trace in origin_A_matrix_traces:
#         single_partition_L2_cache.read(addr=trace)
#     print(single_partition_L2_cache.num_total_reads, single_partition_L2_cache.read_hit, single_partition_L2_cache.read_miss)
    
#     # single_partition_L2_cache = memory_partition_L2_cache(partition_id=0, nsets=32, bsize=128, assoc=24)
#     single_partition_L2_cache = memory_partition_L2_cache(partition_id=0, nsets=32, bsize=16, assoc=24)

#     for trace in reorgnize_A_matrix_traces:
#         single_partition_L2_cache.read(addr=trace)
#     print(single_partition_L2_cache.num_total_reads, single_partition_L2_cache.read_hit, single_partition_L2_cache.read_miss)


if __name__ == '__main__':
    print('Start Processing Matrices...')
    origin_A_matrix, reorgnize_A_matrix = Processing_Matrices_Transformer(data_dir=transformer_matrices_data_dir)
    # plot_dot_figure(origin_A_matrix, reorgnize_A_matrix)
    # calculate_reuse_bak(origin_A_matrix, reorgnize_A_matrix)
    # calculate_reuse_cache(origin_A_matrix, reorgnize_A_matrix)