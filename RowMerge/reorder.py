import os
import random
from math import log
from copy import deepcopy
import matplotlib.pyplot as plt

import sparse_matrix_format

def log2(num):
    return log(num, 2)

def binary(num):
    return bin(num).split('0b')[1]

def binary_align32(num):
    return bin(num).split('0b')[1].rjust(32,'0')

def Processing_Matrices_Transformer(root_dir, num, inst_M):
    datapath_dir = root_dir + "dlmc/transformer_matrices.txt"
    dataset_dir = root_dir + "dlmc/"
    transformer_matrices_file_list = open(datapath_dir, 'r')
    transformer_matrices_files = transformer_matrices_file_list.readlines()
    transformer_matrices_file_list.close()

    for file_name in transformer_matrices_files[1213:1214]:
        file = os.path.join(dataset_dir, file_name.split('\n')[0])
        print(file)
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
        print(ROW, COL, NNZ)
        hypergraph_patitioning_parts = int(ROW / inst_M)
        origin_A_matrix = [[0 for i in range(COL)] for j in range(ROW)]

        if NNZ > 0:
            ROW_INDEXES = data[1].split('\n')[0].split()
            # print(len(ROW_INDEXES))
            for i in range(len(ROW_INDEXES)):
                ROW_INDEXES[i] = int(ROW_INDEXES[i])
            COL_INDEXES = data[2].split('\n')[0].split()
            # print(len(COL_INDEXES))
            for i in range(len(COL_INDEXES)):
                COL_INDEXES[i] = int(COL_INDEXES[i])
                
            for i in range(ROW):
                for j in COL_INDEXES[ROW_INDEXES[i]:ROW_INDEXES[i+1]]:
                    origin_A_matrix[i][j] = 1
            ######################################################################
            ######################################################################
            ######################################################################
            ### error check by generating CRS format
            error_check_indexes = []
            for i in range(len(origin_A_matrix)):
                for j in range(len(origin_A_matrix[0])):
                    if not origin_A_matrix[i][j] == 0:
                        error_check_indexes.append(j)
            if not len(error_check_indexes) == len(COL_INDEXES):
                print("ERROR: len is not equal")
            for i in range(len(error_check_indexes)):
                if not error_check_indexes[i] == COL_INDEXES[i]:
                    print("ERROR: ", error_check_indexes[i], '!=', COL_INDEXES[i])
            print("PASS")
            ######################################################################
            ######################################################################
            ######################################################################
            ### restore hypergraph
            vertices_weight = [0 for i in range(ROW)] # A rows
            nets_vertices_list = [[] for i in range(COL)]

            for i in range(ROW):
                vertices_weight[i] = ROW_INDEXES[i+1] - ROW_INDEXES[i]

            for i in range(ROW):
                column_indexes_of_this_row = COL_INDEXES[ROW_INDEXES[i]:ROW_INDEXES[i+1]]
                for column in column_indexes_of_this_row:
                    if not i in nets_vertices_list[column]:
                        nets_vertices_list[column].append(i)
                        
            pins_num = 0
            for i in range(len(nets_vertices_list)):
                for j in range(len(nets_vertices_list[i])):
                    pins_num += 1
                    
            # myken_file_name = "./part." + str(hypergraph_patitioning_parts) + ".txt"
            f_myken_u = open("myken.u", 'w')
            # f_myken_u = open(myken_file_name, 'w')
            f_myken_u.writelines(str(0)+' '+str(ROW)+' '+str(COL)+' '+str(pins_num)+'\n')
            for i in range(len(nets_vertices_list)):
                string = ''
                for j in nets_vertices_list[i]:
                    string += str(j)+' '
                string += '\n'
                f_myken_u.writelines(string)
            for i in vertices_weight:
                f_myken_u.writelines(str(i)+'\n')
            f_myken_u.close()
            ######################################################################
            ######################################################################
            ######################################################################
            ### do hypergraph patitioning
            os.system('cd hypergraph/PaToH/linux/ && ./patoh ../../../myken.u '+ str(hypergraph_patitioning_parts))
            os.system('cd ../../../')
            # os.system('cat myken.u.part.'+str(hypergraph_patitioning_parts))

            ### reorgnize A matrix
            file_ordering_list = open('./myken.u.part.'+str(hypergraph_patitioning_parts), 'r')
            ordering_list = file_ordering_list.readlines()
            file_ordering_list.close()

            for i in range(len(ordering_list)):
                ordering_list[i] = int(ordering_list[i].split('\n')[0])
            # print(ordering_list)

            reorgnize_A_matrix = [[0 for i in range(COL)] for j in range(ROW)]

            rows_haved_in_single_part_of = [0 for i in range(hypergraph_patitioning_parts)]
            for i in range(len(ordering_list)):
                # _ is row number, ordering_list[_] is the reorgnized_part of current row.

                reorgnize_A_matrix[ordering_list[i]*int(ROW/hypergraph_patitioning_parts)+\
                rows_haved_in_single_part_of[ordering_list[i]]] = deepcopy(origin_A_matrix[i])
                rows_haved_in_single_part_of[ordering_list[i]] += 1
            
            ####################################################################################
            # origin_A_matrix, reorgnize_A_matrix
            return origin_A_matrix, reorgnize_A_matrix, NNZ
        else:#NNZ = 0
            return origin_A_matrix, origin_A_matrix, NNZ

def file_to_sparse_matrix(filename):
    # data_of_this_file = open(filename, 'r')
    # data = data_of_this_file.readlines()
    # data_of_this_file.close()
    
    # ROW_COL_NNZ = data[0].split('\n')[0].split()
    # ROW, COL, NNZ = int(ROW_COL_NNZ[0].split(',')[0]), \
    #                 int(ROW_COL_NNZ[1].split(',')[0]), \
    #                 int(ROW_COL_NNZ[2].split(',')[0])
    # # print(ROW, COL, NNZ)
    # ROW_PTR = data[1].split('\n')[0].split()
    # # print(len(ROW_PTR))
    # for i in range(len(ROW_PTR)):
    #     ROW_PTR[i] = int(ROW_PTR[i])
    # COL_IND = data[2].split('\n')[0].split()
    # # print(len(COL_IND))
    # for i in range(len(COL_IND)):
    #     COL_IND[i] = int(COL_IND[i])

    sm = sparse_matrix_format.sparse_matrix(filename)

    return sm#ROW, COL, NNZ, ROW_PTR, COL_IND

def return_2_ARRAY(ROW, COL, ROW_PTR, COL_IND):
    origin_matrix = [[0 for i in range(COL)] for j in range(ROW)]
    colInd_list_array = []
    for i in range(ROW):
        this_row_colind = []
        for j in COL_IND[ROW_PTR[i]:ROW_PTR[i+1]]:
            origin_matrix[i][j] = 1
            this_row_colind.append(j)
        colInd_list_array.append(this_row_colind)
    
    return origin_matrix, colInd_list_array
    
def return_all_zero_col(vec_len, ROW, COL, origin_A_matrix):
    num_vec_row = int(ROW / vec_len)
    cnt = 0
    for i in range(num_vec_row):
        for j in range(COL):
            flag = 0
            for k in range(vec_len):
                flag += origin_A_matrix[i*vec_len+k][j]
            if(flag == 0):
                cnt += 1
    return cnt
    
def read_files(datapath_dir, root_dir):
    # datapath_dir = "/home/xuezeyu/dlmc/transformer_matrices.txt"
    matrices_file_list = open(datapath_dir, 'r')
    matrices_files = matrices_file_list.readlines()
    matrices_file_list.close()
    file_num = len(matrices_files)
    matrices_paths = []
    for filename in matrices_files:
        file = os.path.join(root_dir, filename.split('\n')[0])
        matrices_paths.append(file)
    return matrices_paths, file_num

def hypergraph_reorder(ROW, COL, NNZ, ROW_PTR, COL_IND, origin_matrix, inst_M):
    if(NNZ > 0):
        hypergraph_patitioning_parts = int(ROW / inst_M)
        ### error check by generating CRS format
        error_check_indexes = []
        for i in range(len(origin_matrix)):
            for j in range(len(origin_matrix[0])):
                if not origin_matrix[i][j] == 0:
                    error_check_indexes.append(j)
        if not len(error_check_indexes) == len(COL_IND):
            print("ERROR: len is not equal")
            return
        for i in range(len(error_check_indexes)):
            if not error_check_indexes[i] == COL_IND[i]:
                print("ERROR: ", error_check_indexes[i], '!=', COL_IND[i])
                return
        print("PASS")
        ######################################################################
        ######################################################################
        ######################################################################
        ### restore hypergraph
        vertices_weight = [0 for i in range(ROW)] # A rows
        nets_vertices_list = [[] for i in range(COL)]

        for i in range(ROW):
            vertices_weight[i] = ROW_PTR[i+1] - ROW_PTR[i]

        for i in range(ROW):
            column_indexes_of_this_row = COL_IND[ROW_PTR[i]:ROW_PTR[i+1]]
            for column in column_indexes_of_this_row:
                if not i in nets_vertices_list[column]:
                    nets_vertices_list[column].append(i)
                            
        pins_num = 0
        for i in range(len(nets_vertices_list)):
            for j in range(len(nets_vertices_list[i])):
                pins_num += 1
                
        f_myken_u = open("myken.u", 'w')
        f_myken_u.writelines(str(0)+' '+str(ROW)+' '+str(COL)+' '+str(pins_num)+'\n')
        for i in range(len(nets_vertices_list)):
            string = ''
            for j in nets_vertices_list[i]:
                string += str(j)+' '
            string += '\n'
            f_myken_u.writelines(string)
        for i in vertices_weight:
            f_myken_u.writelines(str(i)+'\n')
        f_myken_u.close()
        ######################################################################
        ######################################################################
        ######################################################################
        ### do hypergraph patitioning
        os.system('cd hypergraph/PaToH/linux/ && ./patoh ../../../myken.u '+ str(hypergraph_patitioning_parts))
        os.system('cd ../../../')
        
        ### reorgnize A matrix
        file_ordering_list = open('./myken.u.part.'+str(hypergraph_patitioning_parts), 'r')
        ordering_list = file_ordering_list.readlines()
        file_ordering_list.close()
        
        for i in range(len(ordering_list)):
            ordering_list[i] = int(ordering_list[i].split('\n')[0])
        
        reorgnize_matrix = [[0 for i in range(COL)] for j in range(ROW)]
        
        rows_haved_in_single_part_of = [0 for i in range(hypergraph_patitioning_parts)]
        for i in range(len(ordering_list)):
        # _ is row number, ordering_list[_] is the reorgnized_part of current row.

            reorgnize_matrix[ordering_list[i]*int(ROW/hypergraph_patitioning_parts)+\
            rows_haved_in_single_part_of[ordering_list[i]]] = deepcopy(origin_matrix[i])
            rows_haved_in_single_part_of[ordering_list[i]] += 1
        return reorgnize_matrix
    else:#NNZ = 0
        return origin_matrix
    
def dense_to_colind_list_array(dense_matrix):
    row = len(dense_matrix)
    colInd_list_array = []
    for i in range(row):
        col = len(dense_matrix[i])
        this_row_colind = []
        for j in range(col):
            if(dense_matrix[i][j] != 0):
                this_row_colind.append(j)
        colInd_list_array.append(this_row_colind)
    return colInd_list_array

if __name__ == '__main__':
    print('Start Processing Matrices...\n\n')
    
    
    new_file_name = "../dataset-v8/"
    datapath_dir = "../dlmc/transformer_matrices.txt"
    dataset_dir = "../dlmc/"
    # a, b, c = Processing_Matrices_Transformer(root_dir, 1, 8)

    matrices_paths, file_num = read_files(datapath_dir, dataset_dir)
    # print(matrices_paths[0])
    # print(file_num)
    for i in range(1):
        print("Matrix: ",matrices_paths[i].split('/')[4]+'/'+matrices_paths[i].split('/')[5]+'/'+matrices_paths[i].split('/')[6])
        sm = file_to_sparse_matrix(matrices_paths[i])
        dest_path = new_file_name + matrices_paths[i].split('/')[3]+'/'+matrices_paths[i].split('/')[4]+'/'+matrices_paths[i].split('/')[5]+'/'+matrices_paths[i].split('/')[6]
        # print(dest_path)
        # ROW, COL, NNZ, ROW_PTR, COL_IND
        ROW = sm.M
        COL = sm.N
        NNZ = sm.nnz
        ROW_PTR = sm.rowPtr
        COL_IND = sm.colInd
        sparsity_ratio = sm.sparsity()
        colind_list_array = sm.colInd_list_array()
        matrix, array = return_2_ARRAY(ROW, COL, ROW_PTR, COL_IND)
        reordered_matrix = hypergraph_reorder(ROW, COL, NNZ, ROW_PTR, COL_IND, matrix, 8)
        reordered_colind_list_array = dense_to_colind_list_array(reordered_matrix)
        CVSE_matrix1 = sparse_matrix_format.CVSE(ROW, COL, NNZ, colind_list_array, 8)
        CVSE_matrix2 = sparse_matrix_format.CVSE(ROW, COL, NNZ, reordered_colind_list_array, 8)
        total_count = int(ROW * COL / 8)
        vec_num1 = CVSE_matrix1.num_vec
        vec_num2 = CVSE_matrix2.num_vec
        all_zero_col_cnt1 = return_all_zero_col(8, ROW, COL, matrix)
        all_zero_col_cnt2 = return_all_zero_col(8, ROW, COL, reordered_matrix)
        new_sparsity_ratio1 = 1.0 - NNZ / (vec_num1 * 8)
        new_sparsity_ratio2 = 1.0 - NNZ / (vec_num2 * 8)
        # print("total vec num: ", total_count)
        # print("all-zero vec num: ", all_zero_col_cnt1, all_zero_col_cnt2)
        # print("non-zero vec num: ", vec_num1, vec_num2)
        print("sparsity ratio: ", sparsity_ratio, new_sparsity_ratio1, new_sparsity_ratio2)
        CVSE_matrix2.CVSE_out(dest_path)
        # for j in range(len(reordered_colind_list_array)):
        #     print(len(reordered_colind_list_array[j]))
    


    
