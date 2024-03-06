import csv
import os
import random
from math import log
import matplotlib.pyplot as plt

import sparse_matrix_format
import reorder

    
if __name__ == "__main__":
    #################################################################################
    #############################directly merge######################################
    #################################################################################
    ##v = 8
    # os.system("cd ../")
    # os.system("mkdir ./dlmc-v8")
    # os.system("cd ./RowMerge/")
    sparse_matrix_format.generate_CVSE_file("../dlmc/", "../dlmc-v8/", 8)
    ##v = 16
    # os.system("cd ../")
    # os.system("mkdir ./dlmc-v16")
    # os.system("cd cd ./RowMerge/")
    sparse_matrix_format.generate_CVSE_file("../dlmc/", "../dlmc-v16/", 16)
    #################################################################################
    #################################################################################
    #################################################################################
    print('Start Reorder Matrices...\n\n')
    new_file_name = "../dataset-v8/"
    datapath_dir = "../dlmc/transformer_matrices.txt"
    dataset_dir = "../dlmc/"

    matrices_paths, file_num = reorder.read_files(datapath_dir, dataset_dir)
    print(file_num)
    
    header = ["NO.", "ROW", "COL", "NNZ", "cnt0-8", "cnt1-8",  "new_sparsity_ratio1", "new_sparsity_ratio2"]
    f = open('./csv_data/all-zero_col_num_8.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(header)
        # writer.writerows(rows)
    for i in range(file_num):
        print("Matrix: ",matrices_paths[i].split('/')[4]+'/'+matrices_paths[i].split('/')[5]+'/'+matrices_paths[i].split('/')[6])
        sm = reorder.file_to_sparse_matrix(matrices_paths[i])
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
        matrix, array = reorder.return_2_ARRAY(ROW, COL, ROW_PTR, COL_IND)
        reordered_matrix = reorder.hypergraph_reorder(ROW, COL, NNZ, ROW_PTR, COL_IND, matrix, 8)
        reordered_colind_list_array = reorder.dense_to_colind_list_array(reordered_matrix)
        CVSE_matrix1 = sparse_matrix_format.CVSE(ROW, COL, NNZ, colind_list_array, 8)
        CVSE_matrix2 = sparse_matrix_format.CVSE(ROW, COL, NNZ, reordered_colind_list_array, 8)
        total_count = int(ROW * COL / 8)
        vec_num1 = CVSE_matrix1.num_vec
        vec_num2 = CVSE_matrix2.num_vec
        all_zero_col_cnt1 = reorder.return_all_zero_col(8, ROW, COL, matrix)
        all_zero_col_cnt2 = reorder.return_all_zero_col(8, ROW, COL, reordered_matrix)
        if (vec_num1 == 0):
            new_sparsity_ratio1 = 1.0
            new_sparsity_ratio2 = 1.0
        else:
            new_sparsity_ratio1 = 1.0 - NNZ / (vec_num1 * 16)
            new_sparsity_ratio2 = 1.0 - NNZ / (vec_num2 * 16)
        print("ALL-ZERO-8 COL NUM| before reordering: ", all_zero_col_cnt1, "| after reordering: ", all_zero_col_cnt2)
        print("sparsity ratio: ", sparsity_ratio, new_sparsity_ratio1, new_sparsity_ratio2)
        CVSE_matrix2.CVSE_out(dest_path)
        this_csv_row = [str(i), str(ROW), str(COL), str(NNZ), str(all_zero_col_cnt1), str(all_zero_col_cnt2), str(new_sparsity_ratio1), str(new_sparsity_ratio2)]
        
        writer.writerow(this_csv_row)
    f.close()
    
    new_file_name1 = "../dataset-v16/"
    
    header1 = ["NO.", "ROW", "COL", "NNZ", "cnt0-16", "cnt1-16",  "new_sparsity_ratio1", "new_sparsity_ratio2"]
    f1 = open('./csv_data/all-zero_col_num_16.csv', 'w')
    writer1 = csv.writer(f1)
    writer1.writerow(header1)
        # writer.writerows(rows)
    for i in range(file_num):
        print("Matrix: ",matrices_paths[i].split('/')[4]+'/'+matrices_paths[i].split('/')[5]+'/'+matrices_paths[i].split('/')[6])
        sm = reorder.file_to_sparse_matrix(matrices_paths[i])
        dest_path = new_file_name1 + matrices_paths[i].split('/')[3]+'/'+matrices_paths[i].split('/')[4]+'/'+matrices_paths[i].split('/')[5]+'/'+matrices_paths[i].split('/')[6]
        # print(dest_path)
        # ROW, COL, NNZ, ROW_PTR, COL_IND
        ROW = sm.M
        COL = sm.N
        NNZ = sm.nnz
        ROW_PTR = sm.rowPtr
        COL_IND = sm.colInd
        sparsity_ratio = sm.sparsity()
        colind_list_array = sm.colInd_list_array()
        matrix, array = reorder.return_2_ARRAY(ROW, COL, ROW_PTR, COL_IND)
        reordered_matrix = reorder.hypergraph_reorder(ROW, COL, NNZ, ROW_PTR, COL_IND, matrix, 16)
        reordered_colind_list_array = reorder.dense_to_colind_list_array(reordered_matrix)
        CVSE_matrix1 = sparse_matrix_format.CVSE(ROW, COL, NNZ, colind_list_array, 16)
        CVSE_matrix2 = sparse_matrix_format.CVSE(ROW, COL, NNZ, reordered_colind_list_array, 16)
        total_count = int(ROW * COL / 16)
        vec_num1 = CVSE_matrix1.num_vec
        vec_num2 = CVSE_matrix2.num_vec
        all_zero_col_cnt1 = reorder.return_all_zero_col(16, ROW, COL, matrix)
        all_zero_col_cnt2 = reorder.return_all_zero_col(16, ROW, COL, reordered_matrix)
        if (vec_num1 == 0):
            new_sparsity_ratio1 = 1.0
            new_sparsity_ratio2 = 1.0
        else:
            new_sparsity_ratio1 = 1.0 - NNZ / (vec_num1 * 16)
            new_sparsity_ratio2 = 1.0 - NNZ / (vec_num2 * 16)
        print("ALL-ZERO-16 COL NUM| before reordering: ", all_zero_col_cnt1, "| after reordering: ", all_zero_col_cnt2)
        print("sparsity ratio: ", sparsity_ratio, new_sparsity_ratio1, new_sparsity_ratio2)
        CVSE_matrix2.CVSE_out(dest_path)
        this_csv_row = [str(i), str(ROW), str(COL), str(NNZ), str(all_zero_col_cnt1), str(all_zero_col_cnt2), str(new_sparsity_ratio1), str(new_sparsity_ratio2)]
        
        writer1.writerow(this_csv_row)
    f1.close()
